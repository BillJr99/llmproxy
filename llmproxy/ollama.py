"""Ollama-protocol endpoints.

Why this module exists
----------------------
``_StripApiPrefix`` (see ``server.py``) already maps ``/api/<path>`` onto the
bare ``/<path>``, so clients that assume an Ollama base URL reach the OpenAI
routes. What it could not do is answer the Ollama *protocol*: ``/api/show`` and
``/api/tags`` had no route behind them at all, so they fell through to Flask's
default HTML 404. A JSON client reports that as a parse error rather than as an
unsupported operation, and a client asking ``/api/show`` for a context window
silently fell back to a hardcoded default instead of using the real number this
proxy already knows.

Everything here is a thin projection of machinery that lives in ``server.py``.
No routing, discovery or capability logic is duplicated: this module resolves a
model id the same way ``/v1/models/<id>`` does, then renames the fields.

Honesty rules
-------------
Two facts Ollama's schema asks for do not exist in llmproxy, and both are
**omitted rather than faked**: ``details.quantization_level`` (nothing in this
codebase tracks quantization) and ``context_length`` when no candidate reports
one. A missing key lets the client apply its own default knowingly; an invented
one is a wrong answer wearing a right answer's clothes.

Routes are registered at the bare path only (``/show``, ``/tags``, ...). The
prefix shim turns ``/api/show`` into ``/show``, and clients are observed to hit
both spellings, so one route serves both.
"""

from __future__ import annotations

import datetime

from flask import Blueprint, jsonify, request

bp = Blueprint("ollama", __name__)


# Ollama's capability vocabulary differs from this codebase's. Only the
# overlapping ones are translated; llmproxy's "json" has no Ollama counterpart
# and is dropped rather than mapped onto something it is not.
_CAPABILITY_NAMES: dict[str, str] = {
    "tools": "tools",
    "vision": "vision",
    "reasoning": "thinking",
}

_NOT_A_MODEL_STORE = (
    "llmproxy is a routing proxy, not a model store: it has no local models to "
    "pull, create, copy or delete. Configure upstream providers instead "
    "(llmproxy --setup), then list what is reachable with /api/tags."
)


def _err(message: str, status: int):
    """An Ollama-shaped error: a bare ``{"error": ...}`` object.

    Deliberately NOT ``server._error``, whose OpenAI envelope
    (``{"error": {"message", "type", "code"}}``) is a different schema. A client
    speaking Ollama reads ``.error`` as a string.
    """
    return jsonify({"error": message}), status


def _resolve(model_id: str):
    """Resolve a client-supplied id to ``(kind, payload)``.

    Returns ``("virtual", [(provider, cfg, upstream), ...])`` for a pool,
    ``("real", (provider, upstream))`` for a single model, or ``(None, None)``
    when the id names nothing this deployment can route to.

    Mirrors the resolution order in ``server.get_model`` so that an id which
    works against ``/v1/models/<id>`` works here too, in every spelling
    ``_canonicalize_model_id`` accepts.
    """
    from . import server

    canonical = server._canonicalize_model_id(model_id, server.load_config())

    if server._is_virtual_model(canonical):
        candidates = server._get_virtual_candidates(canonical)
        if server._is_flagship_virtual_model(canonical):
            candidates = server._flagship_ordered_candidates(
                candidates,
                server._get_flagship_scores(),
                server._get_normalized_free_limits(server.load_config()),
            )
        return "virtual", candidates

    with server._model_route_cache_lock:
        cached = server._model_route_cache.get(canonical)
    if cached:
        return "real", cached

    # Fall back to parsing the canonical form, so a model the route cache has
    # not warmed yet still answers rather than 404-ing (same tolerance as
    # server.get_model).
    provider, _sep, upstream = canonical.partition("__")
    if _sep and server.get_provider(server.load_config(), provider):
        return "real", (provider, upstream)
    return None, None


def _context_length(kind: str, payload) -> int | None:
    """The context window to advertise, or None when nothing is known.

    For a pool this is the **minimum** across current candidates. The router may
    dispatch a request to any member, and ``_order_by_context_fit`` only demotes
    an ill-fitting candidate rather than dropping it, so advertising anything
    larger than the smallest member can hand the client a prompt an upstream
    will reject.
    """
    from . import server

    config = server.load_config()
    ctx_map = server._get_model_context(config)
    discovered = server._get_model_context_snapshot()

    if kind == "real":
        provider, upstream = payload
        return server._model_context_window(provider, upstream, ctx_map, discovered)

    windows = [
        w for w in (
            server._model_context_window(pn, um, ctx_map, discovered)
            for pn, _cfg, um in payload
        ) if w is not None
    ]
    return min(windows) if windows else None


def _capabilities(kind: str, payload) -> list[str]:
    """Ollama capability names for this model or pool.

    For a pool this is the **union** across candidates, not the intersection.
    Unlike context fit, capability gating actively steers: ``_apply_capability_gate``
    reorders toward capable candidates and drops known-incapable ones, so a
    capability any member has is one the router can honour. Reporting the
    intersection would hide tool support that works.
    """
    from . import server

    cap_map = server._model_capabilities(server.load_config())
    targets = [payload] if kind == "real" else [(pn, um) for pn, _c, um in payload]

    found: set[str] = set()
    for provider, upstream in targets:
        for cap, ollama_name in _CAPABILITY_NAMES.items():
            if server._model_has_capability(provider, upstream, cap, cap_map):
                found.add(ollama_name)

    # Every routing target this proxy serves does chat completion.
    return ["completion"] + sorted(found)


def _details(kind: str, payload) -> dict:
    """Ollama's ``details`` block, with unknowable fields left out.

    ``quantization_level`` is never emitted: llmproxy holds no quantization data
    for any model, and a placeholder would be a fabricated hardware claim.
    ``parameter_size`` is emitted only when the id actually carries a size hint.
    """
    from . import server
    from .providers import family_key

    if kind == "real":
        provider, upstream = payload
        ids = [upstream]
    else:
        ids = [um for _pn, _c, um in payload]

    details: dict = {"parent_model": "", "format": "", "families": []}

    families = sorted({f for f in (family_key(i) for i in ids) if f})
    if families:
        details["families"] = families
        # Ollama's singular `family` describes one model. A pool spanning
        # several has no single family, and picking one (the first, the largest,
        # whichever) would assert something untrue of most of its members — so
        # the key is set only when the answer is unambiguous.
        if len(families) == 1:
            details["family"] = families[0]

    # For a pool, report the smallest member for the same reason context uses
    # the minimum: it is the only value true of every candidate.
    sizes = [s for s in (server._param_count(i) for i in ids) if s > 0.0]
    if sizes:
        smallest = min(sizes)
        trimmed = int(smallest) if smallest == int(smallest) else smallest
        details["parameter_size"] = f"{trimmed}B"

    return details


@bp.route("/show", methods=["POST", "GET"])
def show():
    """Ollama ``/api/show``: metadata for one model or virtual pool.

    Accepts ``{"model": ...}`` and the older ``{"name": ...}``. GET with a
    ``?model=`` query is accepted too, for clients that probe before POSTing.
    """
    from . import server

    body = request.get_json(silent=True) or {}
    model_id = (
        body.get("model")
        or body.get("name")
        or request.args.get("model")
        or request.args.get("name")
    )
    if not model_id or not isinstance(model_id, str):
        return _err("Supply a model name, e.g. {\"model\": \"llmproxy/free\"}.", 400)

    kind, payload = _resolve(model_id)
    if kind is None:
        return _err(
            f"Model '{model_id}' is not served by this proxy. "
            f"List what is available at /api/tags.",
            404,
        )
    if kind == "virtual" and not payload:
        # The pool exists but is empty — the same condition that makes a chat
        # request 503. Reuse that hint so both surfaces name the real cause.
        return _err(
            f"No '{model_id}' models are currently available. "
            + server._virtual_model_hint(
                server._canonicalize_model_id(model_id, server.load_config())
            ),
            503,
        )

    arch = "llmproxy"
    model_info: dict = {"general.architecture": arch}
    window = _context_length(kind, payload)
    if window is not None:
        # Keyed to match general.architecture, which is how Ollama clients
        # resolve the field: model_info[f"{arch}.context_length"].
        model_info[f"{arch}.context_length"] = window

    return jsonify({
        # No Modelfile concept exists here; empty strings keep the schema's
        # shape without asserting content.
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": _details(kind, payload),
        "model_info": model_info,
        "capabilities": _capabilities(kind, payload),
    })


@bp.route("/tags", methods=["GET"])
def tags():
    """Ollama ``/api/tags``: everything this proxy can route to.

    Projected from ``/v1/models`` so the two never disagree and the model-list
    cache is shared. ``size`` and ``digest`` have no meaning for a proxied
    model and are reported as ``0`` / ``""`` rather than invented.
    """
    from . import server

    listing = server.list_models().get_json() or {}
    now = datetime.datetime.now(datetime.UTC).isoformat()

    models = []
    for entry in listing.get("data", []):
        mid = entry.get("id")
        if not mid:
            continue
        model: dict = {
            "name": mid,
            "model": mid,
            "modified_at": now,
            "size": 0,
            "digest": "",
            "details": {"parent_model": "", "format": "", "families": []},
        }
        window = entry.get("context_length")
        if isinstance(window, int) and window > 0:
            # Not part of Ollama's /api/tags schema, but harmless to include and
            # it saves a client an /api/show round trip per model.
            model["context_length"] = window
        models.append(model)

    out: dict = {"models": models}
    if "_warning" in listing:
        out["_warning"] = listing["_warning"]
    return jsonify(out)


@bp.route("/ps", methods=["GET"])
def ps():
    """Ollama ``/api/ps``: models loaded in memory.

    Always empty, and truthfully so: llmproxy loads nothing locally, it forwards
    to upstream providers.
    """
    return jsonify({"models": []})


@bp.route("/pull", methods=["POST", "GET"])
@bp.route("/push", methods=["POST", "GET"])
@bp.route("/create", methods=["POST", "GET"])
@bp.route("/copy", methods=["POST", "GET"])
@bp.route("/delete", methods=["POST", "DELETE", "GET"])
@bp.route("/blobs/<path:digest>", methods=["POST", "HEAD", "GET"])
def unsupported(digest: str | None = None):
    """Model-management endpoints, answered as JSON rather than an HTML 404.

    These exist purely so an Ollama client fails legibly. Without them Flask
    returns its HTML 404 page, which a JSON client surfaces as a parse error
    that says nothing about what went wrong.
    """
    return _err(_NOT_A_MODEL_STORE, 404)


def register_ollama(app) -> None:
    """Attach the Ollama-protocol routes to *app*."""
    app.register_blueprint(bp)
