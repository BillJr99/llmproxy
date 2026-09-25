"""Outbound adapter that answers structured chat requests with TypeSafe's Jev.

Jev is a decision model: it evaluates a ``state`` against typed ``questions``
(``noul`` = yes/no probability, ``choice`` = pick one, ``score`` = rubric level)
and never generates text. A chat request can still be answered by it when the
caller asks for JSON through a ``response_format`` JSON schema whose fields are
all decisions. Each field becomes one question, the conversation becomes the
state, and the answers come back as the JSON object the schema describes.

Field mapping (property schema -> question -> value in the reply):

    boolean                              noul    answer >= 0.5
    number, minimum 0, maximum 1         noul    the probability itself
    enum of 2..255 values                choice  the chosen value
    oneOf/anyOf of {const, description}  choice  the chosen const (the
                                                 descriptions are the rubric)
    integer oneOf/anyOf of consts        score   the level's const
    integer, minimum..maximum, 2..10     score   the nearest level
    anything else                        rejected with a 400

The property's ``description`` (or ``title``) is the question's instructions.

The adapter is a process-wide singleton, and ``translate_response`` sees only
the response body, so the decoding rule for each answer travels in the question
id itself: ``json.dumps([kind, field, extra])``. TypeSafe documents question
ids as keys it "is not sent to the underlying model and is not used in
inference", so the tag costs nothing in accuracy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator

from ..config import provider_api_key
from .base import OutboundAdapter, register_outbound

# Question kinds carried in the question id.
_BOOL, _PROB, _CHOICE, _CHOICE_JSON, _SCORE = "b", "p", "c", "j", "s"

_MAX_CHOICES = 255   # TypeSafe: at most 255 options in one choice question
_MAX_LEVELS = 10     # TypeSafe: a score accepts at most 10 levels


def _field_type(spec: dict):
    """The single non-null JSON type of *spec* (``["boolean", "null"]`` -> boolean)."""
    t = spec.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return non_null[0] if len(non_null) == 1 else None
    return t


def _instructions(name: str, spec: dict) -> str:
    return spec.get("description") or spec.get("title") or f"What is the value of `{name}`?"


def _const_options(spec: dict) -> list[dict] | None:
    """The ``{const, description?}`` branches of a oneOf/anyOf, if that is all it is."""
    branches = spec.get("oneOf") or spec.get("anyOf")
    if not isinstance(branches, list) or not branches:
        return None
    if not all(isinstance(b, dict) and "const" in b for b in branches):
        return None
    return branches


def _question_for(name: str, spec: dict) -> tuple[str, dict]:
    """Build ``(question_id, question)`` for one schema property, or raise ValueError."""
    if not isinstance(spec, dict):
        raise ValueError(f"field '{name}' has no schema")
    ftype = _field_type(spec)
    instructions = _instructions(name, spec)

    if ftype == "boolean":
        return json.dumps([_BOOL, name, None]), {"type": "noul", "instructions": instructions}

    if ftype == "number" and spec.get("minimum") == 0 and spec.get("maximum") == 1:
        return json.dumps([_PROB, name, None]), {"type": "noul", "instructions": instructions}

    branches = _const_options(spec)
    if branches is not None:
        consts = [b["const"] for b in branches]
        if ftype == "integer" and all(isinstance(c, int) and not isinstance(c, bool) for c in consts):
            ordered = sorted(branches, key=lambda b: b["const"])
            if not 2 <= len(ordered) <= _MAX_LEVELS:
                raise ValueError(
                    f"field '{name}': a score takes 2 to {_MAX_LEVELS} levels, got {len(ordered)}")
            levels = [b["const"] for b in ordered]
            criteria = [b.get("description") or b.get("title") or str(b["const"]) for b in ordered]
            return (json.dumps([_SCORE, name, levels]),
                    {"type": "score", "instructions": instructions, "criteria": criteria})
        return _choice(name, instructions, consts,
                       [b.get("description") or b.get("title") for b in branches])

    if isinstance(spec.get("enum"), list):
        return _choice(name, instructions, spec["enum"], [None] * len(spec["enum"]))

    if ftype == "integer" and isinstance(spec.get("minimum"), int) and isinstance(spec.get("maximum"), int):
        levels = list(range(spec["minimum"], spec["maximum"] + 1))
        if not 2 <= len(levels) <= _MAX_LEVELS:
            raise ValueError(
                f"field '{name}': an integer range becomes a score of 2 to {_MAX_LEVELS} "
                f"levels, but minimum..maximum spans {len(levels)}")
        return (json.dumps([_SCORE, name, levels]),
                {"type": "score", "instructions": instructions, "criteria": [str(v) for v in levels]})

    raise ValueError(
        f"field '{name}' (type {spec.get('type')!r}) is not a decision. Jev answers "
        "booleans, probabilities (number with minimum 0 and maximum 1), enums, "
        "oneOf/anyOf consts, and bounded integers; it cannot produce free text")


def _choice(name: str, instructions: str, values: list, descriptions: list) -> tuple[str, dict]:
    if not 2 <= len(values) <= _MAX_CHOICES:
        raise ValueError(f"field '{name}': a choice takes 2 to {_MAX_CHOICES} options, got {len(values)}")
    as_strings = all(isinstance(v, str) for v in values)
    options = values if as_strings else [json.dumps(v) for v in values]
    if len(set(options)) != len(options):
        raise ValueError(f"field '{name}' repeats an option")
    kind = _CHOICE if as_strings else _CHOICE_JSON
    return (json.dumps([kind, name, None]),
            {"type": "choice", "instructions": instructions,
             "criteria": dict(zip(options, descriptions, strict=True))})


def _message_text(message: dict) -> str:
    content = message.get("content")
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                out.append(part.get("text", ""))
            else:
                raise ValueError("Jev accepts text only; the conversation carries a "
                                 f"'{part.get('type') if isinstance(part, dict) else type(part).__name__}' part")
        return "".join(out)
    return str(content)


def to_systemone_request(payload: dict) -> dict:
    """Build a TypeSafe ``/v1/systemone`` body from a canonical chat payload.

    Raises ValueError, with a message fit for the client, when the request
    cannot be expressed as decisions.
    """
    if payload.get("tools"):
        raise ValueError("Jev cannot call tools")
    fmt = payload.get("response_format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        raise ValueError(
            "Jev answers only structured requests: send response_format "
            "{type: 'json_schema', json_schema: {schema: {...}}} whose fields are decisions")
    wrapper = fmt.get("json_schema") or {}
    schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict) \
            or not schema["properties"]:
        raise ValueError("response_format.json_schema.schema must be an object schema with properties")

    questions = dict(_question_for(name, spec) for name, spec in schema["properties"].items())

    messages = payload.get("messages") or []
    state = [{"role": m.get("role", "user"), "content": _message_text(m)}
             for m in messages if isinstance(m, dict)]
    if not any(s["content"] for s in state):
        raise ValueError("the conversation is empty; Jev needs a state to evaluate")
    return {"model": payload.get("model", ""), "state": state, "questions": questions}


def _decode(question_id: str, answer: dict):
    """Return ``(field, value)`` for one TypeSafe answer."""
    try:
        kind, field, extra = json.loads(question_id)
    except (ValueError, TypeError):
        # Not one of ours (should not happen); surface the raw answer.
        return question_id, answer
    if kind == _BOOL:
        return field, float(answer.get("noul", 0.0)) >= 0.5
    if kind == _PROB:
        return field, answer.get("noul")
    if kind == _CHOICE:
        return field, answer.get("choice")
    if kind == _CHOICE_JSON:
        return field, json.loads(answer.get("choice", "null"))
    if kind == _SCORE:
        levels = extra or []
        idx = min(max(int(round(float(answer.get("score", 0.0)))), 0), len(levels) - 1)
        return field, levels[idx]
    return field, answer


def _to_chat_completion(data: dict) -> dict:
    values: dict = {}
    raw: dict = {}
    for qid, answer in (data.get("answers") or {}).items():
        field, value = _decode(qid, answer if isinstance(answer, dict) else {})
        values[field] = value
        raw[field] = answer
    usage_in = data.get("usage") or {}
    prompt = int(usage_in.get("input_tokens") or 0)
    completion = int(usage_in.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-typesafe-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(values)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
        # The calibrated answers behind each value (probabilities, confidence),
        # keyed by field. Strict OpenAI clients ignore unknown top-level keys.
        "typesafe": {"model": data.get("model", ""), "answers": raw},
    }


class TypeSafeOutbound(OutboundAdapter):
    name = "typesafe"

    def build_request(self, endpoint, base_url, provider_cfg, payload, *, stream, forwarded_headers):
        headers = {"Content-Type": "application/json"}
        api_key = provider_api_key(provider_cfg)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        user_agent = (forwarded_headers or {}).get("User-Agent")
        if user_agent:
            headers["User-Agent"] = user_agent
        return f"{base_url}/systemone", headers, to_systemone_request(payload)

    def translate_response(self, content: bytes) -> bytes:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content
        if not isinstance(data, dict):
            return content
        return json.dumps(_to_chat_completion(data)).encode("utf-8")

    def parse_stream(self, byte_iter: Iterable[bytes]) -> Iterator[dict | None]:
        # TypeSafe does not stream: the body is one JSON document, relayed to a
        # streaming client as a single chunk carrying the whole answer.
        data = json.loads(b"".join(byte_iter))
        full = _to_chat_completion(data)
        yield {
            "id": full["id"],
            "object": "chat.completion.chunk",
            "created": full["created"],
            "model": full["model"],
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": full["choices"][0]["message"]["content"]},
                "finish_reason": "stop",
            }],
            "usage": full["usage"],
            "typesafe": full["typesafe"],
        }
        yield None


register_outbound(TypeSafeOutbound())
