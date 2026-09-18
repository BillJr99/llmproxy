"""OpenAI **Responses API** inbound dialect (``POST /v1/responses``).

The Responses API is the shape newer OpenAI-ecosystem agents speak, and until
now llmproxy had no route for it at all: such a request fell through to the
generic ``/v1/<subpath>`` passthrough, which resolves a provider directly and
has no cycling engine behind it, so a virtual model like ``llmproxy/free``
simply failed there. This module gives the Responses shape the same treatment
Anthropic Messages and Gemini already get — translate to the canonical OpenAI
chat form on the way in, route it through the full virtual-model machinery
(capacity ordering, capability ordering, context fit, failover), and render the
result back — so an agent that speaks Responses gets free-tier failover like
everything else.

Three shapes have to be reconciled:

* **Input.** ``input`` is a string or a list of *items*, where an item is a
  message, a ``function_call``, or a ``function_call_output``. Chat's flat
  ``messages`` list has a slot for each of those, so the mapping is total.
* **Output.** A chat completion has one ``message`` carrying optional
  ``tool_calls``; a Response has an ``output`` *array* whose entries are
  ``message`` and ``function_call`` items. One choice fans out to several items.
* **Streaming.** Chat streams homogeneous ``chat.completion.chunk`` deltas;
  Responses streams a typed, ordered event vocabulary with a running
  ``sequence_number``. ``ResponsesStreamRenderer`` below is the state machine
  that turns one into the other.

Statefulness (``store`` / ``previous_response_id``) is supported on a
best-effort basis against a bounded in-process store. See ``_ResponseStore``
for exactly what that does and does not promise.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator

from .base import InboundAdapter, register_inbound, sse_event

# Default ceiling when a Responses request does not set ``max_output_tokens``.
# Chat treats a missing ``max_tokens`` as "the model's maximum", and so does
# this: the key is simply omitted rather than invented.
_MAX_STORED_RESPONSES = 256


def _new_id(prefix: str) -> str:
    """A Responses-style opaque id (``resp_``/``msg_``/``fc_`` + hex)."""
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Best-effort conversation store
# ---------------------------------------------------------------------------

class _ResponseStore:
    """Bounded, in-process store backing ``previous_response_id``.

    The real Responses API keeps conversation state server-side, so an agent may
    send only its newest turn and reference the previous response by id. llmproxy
    is otherwise entirely stateless, so this is the one place that holds
    conversation data, and it is deliberately modest about it:

    * **In-process and in-memory.** Nothing is written to disk, so a restart
      drops every stored conversation, and a second gunicorn worker has its own
      store. A client that gets ``previous_response_id`` wrong is told so with a
      clear 400 rather than being silently answered without its own history,
      which would produce a confidently wrong reply.
    * **Bounded.** At most ``_MAX_STORED_RESPONSES`` conversations, evicted
      oldest-first, so a long-running proxy cannot grow without limit.
    * **Opt-out honored.** ``store: false`` on the request skips saving.

    This is enough for an agent loop that stays inside one process for the life
    of a task, which is the case that matters here. It is not a durable
    conversation service and is documented as such in the README.
    """

    def __init__(self, limit: int = _MAX_STORED_RESPONSES) -> None:
        self._data: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()
        self._limit = limit

    def save(self, response_id: str, messages: list[dict]) -> None:
        if not response_id:
            return
        with self._lock:
            self._data[response_id] = messages
            self._data.move_to_end(response_id)
            while len(self._data) > self._limit:
                self._data.popitem(last=False)

    def get(self, response_id: str) -> list[dict] | None:
        with self._lock:
            found = self._data.get(response_id)
            if found is not None:
                self._data.move_to_end(response_id)
            return list(found) if found is not None else None

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._data.pop(response_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


STORE = _ResponseStore()


class UnknownPreviousResponse(ValueError):
    """Raised when ``previous_response_id`` names a conversation we do not hold.

    Surfaced to the client as a 400. Answering without the referenced history
    would silently drop most of the conversation and produce a confident,
    wrong answer, which is worse than an explicit error.
    """


# The request's resolved transcript and store preference, handed from
# ``to_canonical_request`` to whichever render method runs next. Flask serves a
# request on one thread (gthread), so thread-local state is the right scope; the
# dialect layer deliberately does not import Flask, so ``g`` is not an option.
_pending = threading.local()


def _set_pending(messages: list[dict], store: bool) -> None:
    _pending.messages = messages
    _pending.store = store


def _take_pending() -> tuple[list[dict], bool]:
    messages = getattr(_pending, "messages", None)
    store = getattr(_pending, "store", False)
    _pending.messages = None
    _pending.store = False
    return (messages or []), bool(store)


# ---------------------------------------------------------------------------
# Responses request  ->  canonical OpenAI chat
# ---------------------------------------------------------------------------

def _content_to_text(content) -> str:
    """Flatten Responses content (string or typed parts) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    out: list[str] = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and part.get("type") in (
            "input_text", "output_text", "text", "summary_text", "refusal",
        ):
            out.append(part.get("text") or part.get("refusal") or "")
    return "".join(out)


def _content_to_canonical(content):
    """Responses content -> canonical chat content, keeping images as parts.

    Text-only content collapses to a plain string, which is what the rest of the
    proxy expects and what keeps a simple request simple. Content carrying an
    image stays a parts list so ``_request_has_image`` can still see it and route
    to a vision-capable model.
    """
    if isinstance(content, str) or not isinstance(content, list):
        return _content_to_text(content)
    parts: list[dict] = []
    has_image = False
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("input_image", "image_url"):
            has_image = True
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            image: dict = {"type": "image_url", "image_url": {"url": url or ""}}
            if part.get("detail"):
                image["image_url"]["detail"] = part["detail"]
            parts.append(image)
        elif ptype in ("input_text", "output_text", "text", "summary_text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "refusal":
            parts.append({"type": "text", "text": part.get("refusal", "")})
    if not has_image:
        return "".join(p.get("text", "") for p in parts)
    return parts


def _input_to_messages(value) -> list[dict]:
    """``input`` (string or item list) -> canonical chat messages.

    Handles all three item kinds the Responses API defines: messages (including
    the bare ``{"role", "content"}`` form), ``function_call`` items emitted by a
    previous assistant turn, and ``function_call_output`` items carrying a tool
    result. Consecutive ``function_call`` items collapse onto one assistant
    message, which is how chat represents parallel tool calls.
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value else []
    if not isinstance(value, list):
        return []

    messages: list[dict] = []
    pending_calls: list[dict] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": list(pending_calls),
            })
            pending_calls.clear()

    for item in value:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            pending_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "{}",
                },
            })
            continue
        flush_calls()
        if itype == "function_call_output":
            output = item.get("output")
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "",
                "content": output if isinstance(output, str) else json.dumps(output),
            })
            continue
        if itype in (None, "message") and item.get("role"):
            messages.append({
                "role": item["role"],
                "content": _content_to_canonical(item.get("content")),
            })
            continue
        if itype in ("reasoning", "item_reference"):
            # Reasoning items are the model's own scratch work and carry no
            # instruction for the next turn; item references point into the
            # server-side store this proxy does not reproduce. Both are dropped
            # rather than guessed at.
            continue
    flush_calls()
    return messages


def _tools_to_canonical(tools) -> list[dict]:
    """Responses tools (flat) -> chat tools (nested under ``function``).

    Non-function built-ins (web search, file search, code interpreter) are
    dropped: they are executed by OpenAI's own infrastructure, and there is
    nothing behind llmproxy that could run them. Passing them through would
    advertise a capability no upstream in the pool actually has.
    """
    out: list[dict] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") not in (None, "function"):
            continue
        # Both the flat Responses shape and the nested chat shape are accepted,
        # since SDKs in the wild send each.
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        entry: dict = {
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        if fn.get("strict") is not None:
            entry["function"]["strict"] = fn["strict"]
        out.append(entry)
    return out


def _tool_choice_to_canonical(choice):
    """Responses ``tool_choice`` -> the chat spelling."""
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "function" and choice.get("name"):
            return {"type": "function", "function": {"name": choice["name"]}}
        if isinstance(choice.get("function"), dict):
            return choice
    return None


def _responses_to_canonical(body: dict) -> dict:
    """Translate a Responses request body into a canonical chat payload."""
    messages: list[dict] = []

    previous = body.get("previous_response_id")
    if previous:
        prior = STORE.get(previous)
        if prior is None:
            raise UnknownPreviousResponse(previous)
        messages.extend(prior)

    instructions = body.get("instructions")
    if instructions:
        # ``instructions`` replaces the system prompt for this turn rather than
        # appending to a stored one, matching the real API's documented behavior.
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": _content_to_text(instructions)})

    messages.extend(_input_to_messages(body.get("input")))

    payload: dict = {"model": body.get("model", ""), "messages": messages}

    if body.get("max_output_tokens") is not None:
        payload["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "top_logprobs", "seed", "user"):
        if body.get(key) is not None:
            payload[key] = body[key]
    if body.get("stream"):
        payload["stream"] = True
    if body.get("parallel_tool_calls") is not None:
        payload["parallel_tool_calls"] = body["parallel_tool_calls"]
    if body.get("metadata") is not None:
        payload["metadata"] = body["metadata"]
    if body.get("prompt_cache_key") is not None:
        payload["prompt_cache_key"] = body["prompt_cache_key"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        payload["reasoning_effort"] = reasoning["effort"]
    elif reasoning:
        payload["reasoning"] = reasoning

    text = body.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        fmt = text["format"]
        if fmt.get("type") == "json_schema":
            # Responses inlines the schema; chat nests it under ``json_schema``.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    k: v for k, v in fmt.items() if k != "type"
                },
            }
        elif fmt.get("type"):
            payload["response_format"] = {"type": fmt["type"]}

    tools = _tools_to_canonical(body.get("tools"))
    if tools:
        payload["tools"] = tools
        choice = _tool_choice_to_canonical(body.get("tool_choice"))
        if choice is not None:
            payload["tool_choice"] = choice

    # ``store`` defaults to true in the real API; honor that so a follow-up
    # request carrying previous_response_id finds its history.
    store = body.get("store")
    _set_pending(messages, True if store is None else bool(store))
    return payload


# ---------------------------------------------------------------------------
# canonical chat response  ->  Responses object
# ---------------------------------------------------------------------------

_FINISH_TO_STATUS = {
    "stop": "completed",
    "tool_calls": "completed",
    "function_call": "completed",
    "length": "incomplete",
    "content_filter": "incomplete",
}


def _output_items_from_message(message: dict) -> tuple[list[dict], str]:
    """Split one chat message into Responses ``output`` items.

    Returns ``(items, output_text)``. A message carrying both prose and tool
    calls produces a ``message`` item followed by one ``function_call`` item per
    call, which is the ordering the Responses SDKs expect.
    """
    items: list[dict] = []
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(
            p.get("text", "") for p in text if isinstance(p, dict)
        )
    text = text or ""
    refusal = message.get("refusal")
    if text or (not message.get("tool_calls") and not refusal):
        items.append({
            "type": "message",
            "id": _new_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    if refusal:
        items.append({
            "type": "message",
            "id": _new_id("msg"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": refusal}],
        })
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        items.append({
            "type": "function_call",
            "id": _new_id("fc"),
            "call_id": call.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments") or "{}",
            "status": "completed",
        })
    return items, text


def _usage_to_responses(usage) -> dict | None:
    """Chat usage -> the Responses counter names."""
    if not isinstance(usage, dict):
        return None
    out = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        out["input_tokens_details"] = {"cached_tokens": details["cached_tokens"]}
    return out


def _build_response_object(
    data: dict,
    *,
    response_id: str | None = None,
    status: str | None = None,
) -> tuple[dict, str]:
    """Canonical chat completion dict -> a Responses object. Returns (obj, text)."""
    choices = data.get("choices") or []
    message = {}
    finish = None
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        finish = choices[0].get("finish_reason")
    items, text = _output_items_from_message(message if isinstance(message, dict) else {})
    rid = response_id or _new_id("resp")
    obj: dict = {
        "id": rid,
        "object": "response",
        "created_at": data.get("created") or int(time.time()),
        "status": status or _FINISH_TO_STATUS.get(finish or "stop", "completed"),
        "model": data.get("model", ""),
        "output": items,
        # The SDKs expose ``output_text`` as a convenience accessor; providing it
        # directly saves every caller from re-walking ``output``.
        "output_text": text,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if finish == "length" else None
        ),
        "metadata": None,
    }
    usage = _usage_to_responses(data.get("usage"))
    if usage:
        obj["usage"] = usage
    return obj, text


def _remember(response_id: str, assistant_message: dict) -> None:
    """Persist this turn's transcript so a follow-up can reference it."""
    messages, store = _take_pending()
    if not store:
        return
    full = list(messages)
    if assistant_message:
        full.append(assistant_message)
    STORE.save(response_id, full)


# ---------------------------------------------------------------------------
# Streaming: canonical chunks -> the typed Responses event vocabulary
# ---------------------------------------------------------------------------

class ResponsesStreamRenderer:
    """State machine turning canonical chat chunks into Responses SSE events.

    Chat streams one flat sequence of deltas. Responses streams a *structured*
    lifecycle: the response opens, output items are added and completed in order,
    text arrives as deltas inside a content part, and tool-call arguments arrive
    as their own delta type. Every event carries a monotonic
    ``sequence_number``.

    The renderer therefore has to do three things the chat stream does not:
    track which output item is currently open, emit the matching ``.added`` and
    ``.done`` bookends around it, and accumulate the full text and argument
    strings so the ``.done`` events and the final response object can carry
    them.
    """

    def __init__(self) -> None:
        self.seq = 0
        self.response_id = _new_id("resp")
        self.model = ""
        self.created = int(time.time())
        self.text_parts: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.output_index = 0
        self.message_item_id: str | None = None
        self.message_open = False
        self.open_tool_index: int | None = None
        self.finish_reason: str | None = None
        self.usage: dict | None = None

    # — helpers —

    def _event(self, name: str, payload: dict) -> bytes:
        payload = {"type": name, "sequence_number": self.seq, **payload}
        self.seq += 1
        return sse_event(name, payload)

    def _skeleton(self, status: str) -> dict:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "metadata": None,
        }

    # — lifecycle —

    def start(self) -> Iterator[bytes]:
        yield self._event("response.created", {"response": self._skeleton("in_progress")})
        yield self._event("response.in_progress", {"response": self._skeleton("in_progress")})

    def _open_message(self) -> Iterator[bytes]:
        if self.message_open:
            return
        self.message_open = True
        self.message_item_id = _new_id("msg")
        yield self._event("response.output_item.added", {
            "output_index": self.output_index,
            "item": {
                "type": "message", "id": self.message_item_id,
                "status": "in_progress", "role": "assistant", "content": [],
            },
        })
        yield self._event("response.content_part.added", {
            "item_id": self.message_item_id,
            "output_index": self.output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

    def _close_message(self) -> Iterator[bytes]:
        if not self.message_open:
            return
        text = "".join(self.text_parts)
        yield self._event("response.output_text.done", {
            "item_id": self.message_item_id,
            "output_index": self.output_index,
            "content_index": 0,
            "text": text,
        })
        yield self._event("response.content_part.done", {
            "item_id": self.message_item_id,
            "output_index": self.output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })
        yield self._event("response.output_item.done", {
            "output_index": self.output_index,
            "item": {
                "type": "message", "id": self.message_item_id, "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        })
        self.message_open = False
        self.output_index += 1

    def _close_tool(self, index: int) -> Iterator[bytes]:
        call = self.tool_calls.get(index)
        if not call or not call.get("_open"):
            return
        yield self._event("response.function_call_arguments.done", {
            "item_id": call["item_id"],
            "output_index": call["output_index"],
            "arguments": call["arguments"],
        })
        yield self._event("response.output_item.done", {
            "output_index": call["output_index"],
            "item": {
                "type": "function_call", "id": call["item_id"],
                "call_id": call["call_id"], "name": call["name"],
                "arguments": call["arguments"], "status": "completed",
            },
        })
        call["_open"] = False

    def handle(self, chunk: dict) -> Iterator[bytes]:
        """Emit the events implied by one canonical chat chunk."""
        if chunk.get("model"):
            self.model = chunk["model"]
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if content:
                # Text after a tool call means the model went back to prose;
                # close the open call so items stay properly nested.
                if self.open_tool_index is not None:
                    yield from self._close_tool(self.open_tool_index)
                    self.open_tool_index = None
                yield from self._open_message()
                self.text_parts.append(content)
                yield self._event("response.output_text.delta", {
                    "item_id": self.message_item_id,
                    "output_index": self.output_index,
                    "content_index": 0,
                    "delta": content,
                })
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                yield from self._handle_tool_delta(tc)

    def _handle_tool_delta(self, tc: dict) -> Iterator[bytes]:
        index = tc.get("index", 0)
        fn = tc.get("function") or {}
        if index not in self.tool_calls:
            # A new call starts: finish whatever item was open before it.
            yield from self._close_message()
            if self.open_tool_index is not None and self.open_tool_index != index:
                yield from self._close_tool(self.open_tool_index)
            item_id = _new_id("fc")
            self.tool_calls[index] = {
                "item_id": item_id,
                "call_id": tc.get("id") or item_id,
                "name": fn.get("name", ""),
                "arguments": "",
                "output_index": self.output_index,
                "_open": True,
            }
            self.output_index += 1
            self.open_tool_index = index
            yield self._event("response.output_item.added", {
                "output_index": self.tool_calls[index]["output_index"],
                "item": {
                    "type": "function_call", "id": item_id,
                    "call_id": self.tool_calls[index]["call_id"],
                    "name": self.tool_calls[index]["name"],
                    "arguments": "", "status": "in_progress",
                },
            })
        call = self.tool_calls[index]
        if tc.get("id") and not call["call_id"].startswith("call"):
            call["call_id"] = tc["id"]
        if fn.get("name"):
            call["name"] = fn["name"]
        fragment = fn.get("arguments")
        if fragment:
            call["arguments"] += fragment
            yield self._event("response.function_call_arguments.delta", {
                "item_id": call["item_id"],
                "output_index": call["output_index"],
                "delta": fragment,
            })

    def finish(self) -> Iterator[bytes]:
        yield from self._close_message()
        for index in list(self.tool_calls):
            yield from self._close_tool(index)
        items: list[dict] = []
        if self.text_parts:
            items.append({
                "type": "message", "id": self.message_item_id or _new_id("msg"),
                "status": "completed", "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "".join(self.text_parts),
                    "annotations": [],
                }],
            })
        for _index, call in sorted(self.tool_calls.items()):
            items.append({
                "type": "function_call", "id": call["item_id"],
                "call_id": call["call_id"], "name": call["name"],
                "arguments": call["arguments"], "status": "completed",
            })
        status = _FINISH_TO_STATUS.get(self.finish_reason or "stop", "completed")
        final = self._skeleton(status)
        final["output"] = items
        final["output_text"] = "".join(self.text_parts)
        if self.finish_reason == "length":
            final["incomplete_details"] = {"reason": "max_output_tokens"}
        usage = _usage_to_responses(self.usage)
        if usage:
            final["usage"] = usage
        # Record the turn before the terminal event, so a client that
        # immediately follows up with previous_response_id cannot lose the race.
        _remember(self.response_id, self._assistant_message())
        event = "response.incomplete" if status == "incomplete" else "response.completed"
        yield self._event(event, {"response": final})

    def _assistant_message(self) -> dict:
        message: dict = {"role": "assistant", "content": "".join(self.text_parts) or None}
        calls = [
            {
                "id": c["call_id"], "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for _i, c in sorted(self.tool_calls.items())
        ]
        if calls:
            message["tool_calls"] = calls
        return message


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ResponsesInbound(InboundAdapter):
    """``POST /v1/responses`` in, canonical OpenAI chat out (and back)."""

    name = "responses"
    is_identity = False

    def to_canonical_request(self, body: dict) -> dict:
        return _responses_to_canonical(body)

    def render_response(self, canonical: bytes) -> bytes:
        try:
            data = json.loads(canonical)
        except (ValueError, TypeError):
            return canonical
        if not isinstance(data, dict):
            return canonical
        if data.get("error") and not data.get("choices"):
            # Upstream errors keep their own shape; the route sets the status.
            return canonical
        obj, _text = _build_response_object(data)
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        _remember(obj["id"], message if isinstance(message, dict) else {})
        return json.dumps(obj).encode("utf-8")

    def render_stream(self, chunks: Iterator[dict | None]) -> Iterator[bytes]:
        renderer = ResponsesStreamRenderer()
        yield from renderer.start()
        for chunk in chunks:
            if chunk is None:  # the chat [DONE] sentinel has no Responses analogue
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                yield renderer._event("error", {
                    "code": None,
                    "message": (chunk["error"] or {}).get("message", "Upstream error."),
                    "param": None,
                })
                continue
            yield from renderer.handle(chunk)
        yield from renderer.finish()


register_inbound(ResponsesInbound())
