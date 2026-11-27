"""Normalize streamed OpenAI ``chat.completion.chunk`` frames before relay.

Some OpenAI-compatible upstreams stream chunks that are close to the OpenAI
shape but not quite. Cloudflare Workers AI is the known case: its choices omit
``index``, its tool-call deltas can omit ``index`` or ``id`` and can send
``function.arguments`` as a JSON object instead of a string, and it can send a
usage-only chunk with no ``choices`` at all. Lenient clients shrug that off.
Strict ones do not: the Vercel AI SDK's openai-compatible provider (used by
opencode) validates every chunk with a zod schema and aborts the whole stream
with "Invalid ... stream event" on the first one that does not match.

The identity relay used to forward upstream bytes untouched, so one quirky
candidate in a failover pool broke the client. This module sits in that relay:

* ``normalize_chat_chunk`` repairs one parsed chunk dict in place and reports
  whether it changed anything. Text strings and typed content parts are
  preserved. Numeric and boolean content becomes JSON text, bare objects are
  omitted, and untyped entries in content arrays are dropped.
* ``ChunkNormalizer`` carries the little per-stream state that repair needs
  (a stable ``id``, tool-call ``id`` to ``index`` mapping) across chunks.
* ``normalize_sse_stream`` reframes a raw SSE byte stream on event boundaries
  and runs each ``data:`` payload through the normalizer. An event that needed
  no change is relayed as its original bytes, so a stream that is already valid
  comes out byte-identical.

Standalone error payloads (``{"error": ...}``) pass through unchanged so a
client still sees the upstream's failure. OpenAI wire repairs still supply
choice and tool indices, delta objects and tool argument strings, although
the SDK schema allows some of those fields to be absent or null. SSE comments and keep-alives stay comments, and
``[DONE]`` is kept. Data that is not a JSON object carrying ``choices``,
``usage`` or ``error`` is dropped rather than relayed as a data event a strict
client would reject.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger("llmproxy.stream_normalize")

CHUNK_OBJECT = "chat.completion.chunk"

# SSE line endings only (str.splitlines also splits on U+2028 and friends,
# which are legal inside JSON strings).
_LINE_SPLIT = re.compile(r"\r\n|\n|\r")
# An event ends at a blank line; tolerate mixed CRLF/LF terminators.
_EVENT_SEP = re.compile(rb"(?:\r\n|\n|\r(?!\n)){2}")


class ChunkNormalizer:
    """Per-stream state for ``normalize_chat_chunk``.

    ``model`` and ``created`` fill a chunk that lacks them. The first ``id`` the
    upstream sends is reused for later chunks that lack one, so every chunk of a
    stream carries the same id even when the upstream only sent it once.
    """

    def __init__(self, model: str = "", created: int | None = None):
        self.model = model or ""
        self.created = int(created if created is not None else time.time())
        self.id: str | None = None
        # choice index -> {tool call id -> tool call index}
        self._tool_ids: dict[int, dict[str, int]] = {}
        # choice index -> index of the tool call the last delta belonged to
        self._last_tool: dict[int, int] = {}
        # choice index -> next unused tool call index
        self._next_tool: dict[int, int] = {}
        # choice index -> tool call indices that already carried a function name
        self._named: dict[int, set[int]] = {}

    def normalize(self, chunk) -> tuple[dict | None, bool]:
        return normalize_chat_chunk(chunk, self)


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v.isascii() and re.fullmatch(r"-?[0-9]+", v):
            try:
                return int(v)
            except ValueError:  # Python's integer string length limit
                return None
    return None


def _as_index(value) -> int | None:
    """A usable array index: a non-negative integer, else ``None``."""
    i = _as_int(value)
    return i if i is not None and i >= 0 else None


def _set(d: dict, key: str, value) -> bool:
    """Set ``d[key]`` and report whether that changed anything."""
    if key in d and d[key] == value and type(d[key]) is type(value):
        return False
    d[key] = value
    return True


_USAGE_DETAILS = {
    "prompt_tokens_details": ("cached_tokens",),
    "completion_tokens_details": (
        "reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens",
    ),
}
_NUMERIC_STRING = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _is_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _finite_json(value):
    """Replace exponent overflow with null when a frame must be serialized."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def _reject_json_constant(value: str):
    raise ValueError(f"not a JSON number: {value}")


def _drop_unless(obj: dict, key: str, valid: Callable[[object], bool]) -> bool:
    value = obj.get(key)
    if value is None or valid(value):
        return False
    del obj[key]
    return True


def _normalize_usage_counts(obj: dict, keys: tuple[str, ...]) -> bool:
    changed = False
    for key in keys:
        value = obj.get(key)
        if value is None or _is_number(value):
            continue
        if isinstance(value, str) and _NUMERIC_STRING.fullmatch(value.strip()):
            number = _as_int(value)
            if number is None:
                number = float(value)
            if _is_number(number):
                obj[key] = number
                changed = True
                continue
        del obj[key]
        changed = True
    return changed


def _normalize_content(delta: dict) -> bool:
    content = delta.get("content")
    if isinstance(content, list):
        typed = [part for part in content if isinstance(part, dict) and isinstance(part.get("type"), str)]
        return len(typed) != len(content) and _set(delta, "content", typed)
    if isinstance(content, bool) or _is_number(content):
        return _set(delta, "content", json.dumps(content))
    return _drop_unless(delta, "content", lambda value: isinstance(value, str))


def _normalize_tool_calls(calls: list, choice_index: int, state: ChunkNormalizer) -> bool:
    changed = False
    ids = state._tool_ids.setdefault(choice_index, {})
    named = state._named.setdefault(choice_index, set())
    for pos, call in enumerate(calls):
        changed |= _drop_unless(call, "id", lambda value: isinstance(value, str))
        fn = call["function"]
        changed |= _drop_unless(fn, "name", lambda value: isinstance(value, str))
        changed |= _drop_unless(call, "extra_content", lambda value: isinstance(value, dict))
        extra = call.get("extra_content")
        if isinstance(extra, dict):
            changed |= _drop_unless(extra, "google", lambda value: isinstance(value, dict))
            google = extra.get("google")
            if isinstance(google, dict):
                changed |= _drop_unless(google, "thought_signature", lambda value: isinstance(value, str))
        call_id = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else None
        name = fn.get("name")
        has_name = isinstance(name, str) and bool(name)
        nxt = state._next_tool.get(choice_index, 0)
        idx = _as_index(call.get("index"))
        if idx is None:
            last = state._last_tool.get(choice_index)
            if call_id is not None and call_id in ids:
                idx = ids[call_id]
            elif call_id is not None:
                # A new call: the next free slot, so parallel calls that each
                # arrive without an index still get distinct ones.
                idx = max(nxt, pos)
            elif last is not None and pos == 0 and len(calls) == 1 and not (has_name and last in named):
                # An id-less continuation fragment belongs to the call in progress
                # (unless it names a function and that call already has a name).
                idx = last
            elif has_name and last is not None:
                idx = max(nxt, pos)
            else:
                idx = pos
        changed |= _set(call, "index", idx)
        if call_id is not None:
            ids.setdefault(call_id, idx)
        if has_name:
            named.add(idx)
        state._last_tool[choice_index] = idx
        state._next_tool[choice_index] = max(nxt, idx + 1)

        if has_name and not (isinstance(call.get("type"), str) and call.get("type")):
            changed |= _set(call, "type", "function")
        args = fn.get("arguments")
        if args is None:
            changed |= _set(fn, "arguments", "")
        elif not isinstance(args, str):
            changed |= _set(fn, "arguments", json.dumps(_finite_json(args), ensure_ascii=False, allow_nan=False))
    return changed


def normalize_chat_chunk(chunk, state: ChunkNormalizer | None = None) -> tuple[dict | None, bool]:
    """Repair one streamed chat chunk.

    Returns ``(chunk, changed)``. ``chunk`` is ``None`` when the payload should
    be dropped (not a dict, or a dict with none of ``choices``, ``usage`` or
    ``error``). An ``error`` payload is returned untouched. Otherwise the dict is
    fixed in place: each choice has an integer ``index`` (its position when
    missing) and a ``delta`` object; each tool call has an integer ``index``, ``type: "function"`` when it
    names a function, and a string ``function.arguments``; ``finish_reason`` is
    ``None`` or a string. When any of that needed repair, missing ``object``,
    ``id``, ``created`` and ``model`` are filled too. Missing metadata alone
    is accepted, while wrong-typed metadata triggers repair. Token counts
    preserve numeric values, coerce numeric strings and drop other types.
    """
    if state is None:
        state = ChunkNormalizer()
    if not isinstance(chunk, dict):
        return None, False
    has_choices = isinstance(chunk.get("choices"), list)
    has_usage = isinstance(chunk.get("usage"), dict)
    if not has_choices and not has_usage:
        if "error" in chunk:
            return chunk, False
        return None, False

    # Structural repairs first: these are what strict clients reject.
    changed = False
    if not has_choices:
        # Usage-only chunk: strict schemas want the (empty) array OpenAI sends.
        changed |= _set(chunk, "choices", [])

    choices = [(pos, choice) for pos, choice in enumerate(chunk["choices"]) if isinstance(choice, dict)]
    if len(choices) != len(chunk["choices"]):
        changed |= _set(chunk, "choices", [choice for _, choice in choices])
    for pos, choice in choices:
        idx = _as_index(choice.get("index"))
        changed |= _set(choice, "index", pos if idx is None else idx)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
            changed |= _set(choice, "delta", delta)
        fr = choice.get("finish_reason")
        if fr is not None and not isinstance(fr, str):
            changed |= _set(choice, "finish_reason", None)
        changed |= _normalize_content(delta)
        changed |= _drop_unless(delta, "role", lambda value: value in ("assistant", ""))
        for key in ("reasoning", "reasoning_content"):
            changed |= _drop_unless(delta, key, lambda value: isinstance(value, str))
        changed |= _drop_unless(delta, "tool_calls", lambda value: isinstance(value, list))
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            valid_calls = [
                call for call in calls
                if isinstance(call, dict) and isinstance(call.get("function"), dict)
            ]
            changed |= _set(delta, "tool_calls", valid_calls)
            changed |= _normalize_tool_calls(valid_calls, choice["index"], state)

    changed |= _drop_unless(chunk, "usage", lambda value: isinstance(value, dict))
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        changed |= _normalize_usage_counts(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
        for key, counts in _USAGE_DETAILS.items():
            changed |= _drop_unless(usage, key, lambda value: isinstance(value, dict))
            details = usage.get(key)
            if isinstance(details, dict):
                changed |= _normalize_usage_counts(details, counts)

    cid = chunk.get("id")
    if isinstance(cid, str) and cid and state.id is None:
        state.id = cid
    if cid is not None and not isinstance(cid, str):
        changed = True
    model = chunk.get("model")
    if model is not None and not isinstance(model, str):
        changed = True
    created = chunk.get("created")
    if created is not None and not _is_number(created):
        changed = True
    if not changed:
        # Missing top-level metadata alone is accepted by every client we know
        # of (the AI SDK schema has id/created/model as nullish), so a chunk that
        # needs nothing else is relayed exactly as the upstream sent it.
        return chunk, False

    # The chunk is re-serialized anyway: complete its top-level metadata too.
    if not isinstance(chunk.get("object"), str) or not chunk.get("object"):
        _set(chunk, "object", CHUNK_OBJECT)
    if not (isinstance(cid, str) and cid):
        if state.id is None:
            state.id = f"chatcmpl-llmproxy-{state.created}"
        _set(chunk, "id", state.id)
    if not _is_number(created):
        timestamp = _as_int(created)
        _set(chunk, "created", timestamp if _is_number(timestamp) else state.created)
    if not isinstance(chunk.get("model"), str):
        _set(chunk, "model", state.model)
    return chunk, changed


def _split_events(buf: bytearray) -> list[bytes]:
    """Pop every complete SSE event (terminated by a blank line) off ``buf``.

    One linear scan per call, then a single trim of the consumed prefix.
    """
    events: list[bytes] = []
    pos = 0
    for m in _EVENT_SEP.finditer(buf):
        events.append(bytes(buf[pos:m.end()]))
        pos = m.end()
    if pos:
        del buf[:pos]
    return events


def _normalize_event(raw: bytes, state: ChunkNormalizer) -> bytes:
    """Return the bytes to relay for one complete SSE event (``b""`` drops it)."""
    text = raw.decode("utf-8", "replace")
    lines = _LINE_SPLIT.split(text)
    data_lines = [ln for ln in lines if ln.startswith("data:")]
    if not data_lines:
        return raw  # comment / keep-alive / bare event: relay as-is
    data = "\n".join(ln[len("data:"):].lstrip(" ") for ln in data_lines)
    if data.strip() == "[DONE]":
        return raw
    try:
        parsed = json.loads(data, parse_constant=_reject_json_constant)
    except (ValueError, TypeError):
        logger.debug("[stream_normalize] dropping non-JSON data event: %r", data[:120])
        return b""
    try:
        fixed, changed = state.normalize(parsed)
    except Exception:  # never let a repair bug kill a stream: relay as received
        logger.warning("[stream_normalize] normalizer error; relaying event unchanged", exc_info=True)
        return raw
    if fixed is None:
        logger.debug("[stream_normalize] dropping non-chunk data event: %r", data[:120])
        return b""
    if not changed:
        return raw
    others = [ln for ln in lines if ln and not ln.startswith("data:")]
    out = "".join(f"{ln}\n" for ln in others)
    try:
        out += "data: " + json.dumps(
            _finite_json(fixed), ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ) + "\n\n"
        return out.encode("utf-8", errors="backslashreplace")
    except (ValueError, TypeError):
        logger.warning("[stream_normalize] could not re-serialize event; relaying unchanged", exc_info=True)
        return raw


def normalize_sse_stream(byte_iter: Iterable[bytes], model: str = "") -> Iterator[bytes]:
    """Reframe an OpenAI chat SSE byte stream and normalize each chunk.

    Yields one bytes object per relayed event, in order. Events that need no
    repair are yielded as their original bytes, so a valid stream is relayed
    byte-for-byte. A trailing event with no terminating blank line is processed
    at the end of the stream.
    """
    state = ChunkNormalizer(model=model)
    buf = bytearray()
    for chunk in byte_iter:
        if not chunk:
            continue
        buf += chunk
        for event in _split_events(buf):
            out = _normalize_event(event, state)
            if out:
                yield out
    if buf.strip():
        out = _normalize_event(bytes(buf), state)
        if out:
            yield out
