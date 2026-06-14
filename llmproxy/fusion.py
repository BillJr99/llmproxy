"""Multi-model deliberation ("fusion") for llmproxy.

This module holds the dialect-agnostic, HTTP-free pieces of the fusion pipeline:
config parsing, diversity-aware panel selection, judge and synthesizer prompt
construction, tolerant parsing of the judge's structured analysis, and the
additive ``llmproxy_fusion`` report block. The orchestration that actually
issues upstream calls (parallel panel fan-out, judge pass, streamed synthesis)
lives in :mod:`llmproxy.server`, which imports these helpers; keeping the
deliberation logic here makes it unit-testable without a running upstream.

Pipeline (see ``server._proxy_fusion``):

  1. Select a panel of N models, preferring distinct providers for diversity.
  2. Fan the prompt out to the panel in parallel (non-streaming).
  3. A judge model compares the panel responses and emits structured analysis
     (consensus, contradictions, coverage gaps, unique insights, blind spots).
  4. A synthesizer model writes the final answer grounded in that analysis.

Graceful degradation mirrors OpenRouter's fusion: the request proceeds when at
least one panel member answers; the synthesizer falls back to the first
successful panel response if the judge or synthesizer call fails; and the
request errors only when every panel member fails.
"""

from __future__ import annotations

import json
import traceback

# Default config used when keys are absent. Kept in sync with
# config.DEFAULT_FUSION_CONFIG; duplicated here so this module has no import
# dependency on the config module (which imports providers, etc.).
_FUSION_DEFAULTS: dict = {
    "enabled": True,
    "panel": None,
    "panel_size": 4,
    "diversity": "provider",
    "judge_model": None,
    "synthesizer_model": None,
    "allow_paid": True,
    "report": {"metadata": True},
    "forced_capability": "restrict",
}

# Minimum number of distinct candidate models needed for fusion to be meaningful
# (and to be advertised in GET /v1/models). One model is not a deliberation.
MIN_PANEL = 2

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge in a multi-model deliberation. Several AI models "
    "were each given the same user request and answered independently. Their "
    "answers are provided below, labeled by source. Compare the answers; do not "
    "merge them and do not write a new answer to the user's request. Identify "
    "where the models agree (higher-confidence consensus), where they contradict "
    "one another, what only some models covered, any uniquely valuable insights, "
    "and any blind spots none of them addressed. Respond with a single JSON object "
    "and nothing else, using exactly these keys: \"consensus\" (array of strings), "
    "\"contradictions\" (array of strings), \"coverage_gaps\" (array of strings), "
    "\"unique_insights\" (array of strings), \"blind_spots\" (array of strings)."
)

SYNTH_SYSTEM_PROMPT = (
    "You are a synthesizer in a multi-model deliberation. Several AI models each "
    "answered the user's request independently, and a judge compared them. Using "
    "the panel answers and the judge's analysis, write the single best final "
    "answer to the user's original request. Prefer points of consensus, resolve "
    "contradictions on the merits rather than by majority vote, incorporate "
    "uniquely valuable insights, and address blind spots where you can. Write the "
    "answer directly to the user; do not mention the panel, the judge, or that "
    "multiple models were involved."
)


def get_fusion_config(config: dict) -> dict:
    """Return the fusion config with defaults applied and values sanitized.

    Defensive against hand-edited configs: a missing or non-dict ``fusion`` block
    yields the defaults, and individual malformed values fall back to their
    default rather than raising.
    """
    raw = config.get("fusion")
    if not isinstance(raw, dict):
        return dict(_FUSION_DEFAULTS)

    merged = dict(_FUSION_DEFAULTS)
    merged.update({k: v for k, v in raw.items() if k in _FUSION_DEFAULTS})

    # Sanitize panel_size to a positive int.
    try:
        merged["panel_size"] = max(MIN_PANEL, int(merged["panel_size"]))
    except (TypeError, ValueError):
        merged["panel_size"] = _FUSION_DEFAULTS["panel_size"]

    if merged["diversity"] not in ("provider", "none"):
        merged["diversity"] = "provider"
    if merged["forced_capability"] not in ("restrict", "bypass"):
        merged["forced_capability"] = "restrict"
    if not isinstance(merged.get("report"), dict):
        merged["report"] = dict(_FUSION_DEFAULTS["report"])
    if not (isinstance(merged.get("panel"), list) or merged.get("panel") is None):
        merged["panel"] = None
    return merged


def select_panel(
    ordered_candidates: list[tuple[str, dict, str]],
    panel_size: int,
    prefer_diversity: bool,
) -> list[tuple[str, dict, str]]:
    """Pick up to *panel_size* panel members from pre-ordered *ordered_candidates*.

    *ordered_candidates* is assumed to already carry the desired base ordering
    (capacity headroom for the free pool, random rotation otherwise), so this
    function only layers the diversity preference on top without disturbing that
    order otherwise.

    With *prefer_diversity*, a first pass takes the first candidate seen from each
    distinct provider (preserving input order), which spreads the panel across
    vendors so the deliberation benefits from genuinely different training and
    decoding rather than several near-identical siblings. If that pass yields
    fewer than *panel_size* members, a second pass fills the remaining slots from
    the leftovers in their original order. Without *prefer_diversity*, the first
    *panel_size* candidates are returned as-is.
    """
    if panel_size <= 0 or not ordered_candidates:
        return []
    if not prefer_diversity:
        return ordered_candidates[:panel_size]

    chosen: list[tuple[str, dict, str]] = []
    leftovers: list[tuple[str, dict, str]] = []
    seen_providers: set[str] = set()
    for cand in ordered_candidates:
        provider_name = cand[0]
        if provider_name not in seen_providers and len(chosen) < panel_size:
            chosen.append(cand)
            seen_providers.add(provider_name)
        else:
            leftovers.append(cand)
    for cand in leftovers:
        if len(chosen) >= panel_size:
            break
        chosen.append(cand)
    return chosen[:panel_size]


def extract_message_text(body_bytes: bytes) -> str:
    """Pull the assistant message text from an OpenAI-canonical chat completion.

    Returns an empty string when the body is unparseable or carries no content,
    so a malformed panel response degrades to "no contribution" rather than
    breaking the whole deliberation.
    """
    try:
        data = json.loads(body_bytes)
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # Some providers return content as a list of parts.
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return ""
    except Exception as e:  # noqa: BLE001
        print(f"[fusion:extract_message_text] {e}")
        traceback.print_exc()
        return ""


def _panel_block(panel: list[dict]) -> str:
    """Render the labeled panel answers as a text block for the judge/synth."""
    parts = []
    for entry in panel:
        label = entry.get("label", "Model")
        content = entry.get("content", "").strip()
        parts.append(f"=== {label} ===\n{content}")
    return "\n\n".join(parts)


def build_judge_messages(original_messages: list[dict], panel: list[dict]) -> list[dict]:
    """Construct the judge call's messages.

    The judge receives the original conversation (so it understands the request
    in full context) followed by a single user turn carrying the labeled panel
    answers and the comparison instruction. Equation-to-code style note: this is
    step 3 of the pipeline docstring above, and its output JSON is parsed by
    :func:`parse_analysis` and fed into :func:`build_synthesizer_messages`.
    """
    block = _panel_block(panel)
    instruction = (
        "Here are the panel's answers to the user's request above. Compare them "
        "and return only the JSON analysis described in your instructions.\n\n"
        f"{block}"
    )
    return (
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
        + list(original_messages)
        + [{"role": "user", "content": instruction}]
    )


def build_synthesizer_messages(
    original_messages: list[dict],
    panel: list[dict],
    analysis: dict | None,
) -> list[dict]:
    """Construct the synthesizer call's messages.

    The synthesizer receives the original conversation, the labeled panel
    answers, and the judge's analysis when available. When *analysis* is None
    (the judge failed or returned unparseable output), the synthesizer is asked
    to synthesize directly from the panel answers, which is the graceful
    degradation path for a failed judge.
    """
    block = _panel_block(panel)
    if analysis is not None:
        analysis_text = json.dumps(analysis, indent=2, ensure_ascii=False)
        instruction = (
            "The panel's answers to the user's request above are below, followed "
            "by the judge's structured analysis. Write the single best final "
            "answer to the user's original request, grounded in the analysis.\n\n"
            f"--- Panel answers ---\n{block}\n\n"
            f"--- Judge analysis (JSON) ---\n{analysis_text}"
        )
    else:
        instruction = (
            "The panel's answers to the user's request above are below. A judge "
            "comparison was unavailable, so weigh the answers yourself and write "
            "the single best final answer to the user's original request.\n\n"
            f"--- Panel answers ---\n{block}"
        )
    return (
        [{"role": "system", "content": SYNTH_SYSTEM_PROMPT}]
        + list(original_messages)
        + [{"role": "user", "content": instruction}]
    )


def parse_analysis(text: str) -> dict | None:
    """Tolerantly parse the judge's structured-analysis JSON.

    Strips Markdown code fences and extracts the first balanced ``{...}`` object
    before parsing, so a judge that wraps its JSON in prose or fences still
    yields usable analysis. Returns None when no JSON object can be recovered,
    which the caller treats as "judge failed" and degrades gracefully.
    """
    if not text:
        return None
    cleaned = text.strip()
    # Drop ```json ... ``` or ``` ... ``` fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)
        cleaned = cleaned[1] if len(cleaned) > 1 else ""
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    # Extract the first balanced top-level object.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        obj = json.loads(cleaned[start:end])
        return obj if isinstance(obj, dict) else None
    except Exception as e:  # noqa: BLE001
        print(f"[fusion:parse_analysis] {e}")
        return None


def build_report(
    *,
    panel_used: list[str],
    judge_model: str | None,
    synthesizer_model: str | None,
    failed_models: list[dict],
    analysis: dict | None,
    fell_back: bool,
    free: bool,
) -> dict:
    """Build the additive ``llmproxy_fusion`` report block.

    This is surfaced both as a top-level key on non-streaming responses and as
    the ``X-LLMProxy-Fusion`` response header (a robust, dialect-agnostic channel
    that also works for streamed responses). It reports which models formed the
    panel, which judged and synthesized, which panel members failed, whether the
    pipeline fell back (judge/synth failure), and the judge analysis when present.
    """
    report = {
        "object": "fusion.report",
        "free": free,
        "panel": panel_used,
        "judge_model": judge_model,
        "synthesizer_model": synthesizer_model,
        "failed_models": failed_models,
        "fell_back": fell_back,
    }
    if analysis is not None:
        report["analysis"] = analysis
    return report


def inject_report(body_bytes: bytes, report: dict) -> bytes:
    """Return *body_bytes* (an OpenAI chat completion) with the report attached.

    The ``llmproxy_fusion`` object is added as a top-level key. Strict OpenAI
    clients ignore unknown top-level keys, while clients that look for it get the
    full provenance. On any parse failure the original bytes are returned
    unchanged so a serialization edge case never corrupts the response.
    """
    try:
        data = json.loads(body_bytes)
        if isinstance(data, dict):
            data["llmproxy_fusion"] = report
            return json.dumps(data).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[fusion:inject_report] {e}")
        traceback.print_exc()
    return body_bytes
