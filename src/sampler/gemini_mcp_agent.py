"""
Gemini-driven Flashpoint MCP agent.

Gemini chooses which Flashpoint MCP tools to call and with what arguments (function calling).
Each chosen call is executed live against https://api.flashpoint.ai/mcp by
GeminiFlashpointMCPClient, and the real result is returned to Gemini. Nothing is simulated.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Set

import google.generativeai as genai

from src.sampler.gemini_mcp_client import GeminiFlashpointMCPClient
from src.utils.dotenv import load_dotenv_keys

load_dotenv_keys(keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"])

GEMINI_MODEL = "gemini-3.5-flash-lite"

# Tools Gemini may be given. Excluded on purpose: deprecated aliases and
# `payment_confirmation` (settles a panel payment). Callers narrow this per task.
EXPOSED_TOOLS = {"demographics", "panel", "tool_survey", "tool_qual", "presentation", "chat"}

SYSTEM_PROMPT = (
    "You are a research agent. Use the Flashpoint MCP tools to answer the user's request. "
    "Call only the tools you need, then answer from the tool results only."
)

_GEMINI_TYPES = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}


def _gemini_schema(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a JSON Schema fragment into the subset Gemini function declarations accept.
    Returns None for shapes Gemini cannot express (free-form objects), so the caller drops them."""
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"]
        if not options:
            return None
        inner = _gemini_schema({**options[0], "description": schema.get("description", options[0].get("description"))})
        return inner

    kind = schema.get("type")
    out: Dict[str, Any] = {}
    if schema.get("description"):
        out["description"] = schema["description"]

    if kind == "object":
        props = schema.get("properties")
        if not props:
            return None
        converted = {}
        for name, sub in props.items():
            c = _gemini_schema(sub)
            if c is not None:
                converted[name] = c
        out["type"] = "OBJECT"
        out["properties"] = converted
        out["required"] = [r for r in schema.get("required", []) if r in converted]
        return out
    if kind == "array":
        items = _gemini_schema(schema.get("items", {"type": "string"}))
        if items is None:
            return None
        out["type"] = "ARRAY"
        out["items"] = items
        return out
    if kind in _GEMINI_TYPES:
        out["type"] = _GEMINI_TYPES[kind]
        if schema.get("enum"):
            out["enum"] = schema["enum"]
        return out
    return None


def build_declarations(tools: List[Dict[str, Any]], exposed: Set[str] = EXPOSED_TOOLS) -> List[Dict[str, Any]]:
    declarations = []
    for tool in tools:
        if tool["name"] not in exposed:
            continue
        params = _gemini_schema(tool.get("inputSchema", {"type": "object", "properties": {}}))
        if params is None:
            raise RuntimeError(f"Cannot express Flashpoint tool '{tool['name']}' schema for Gemini")
        declarations.append({
            "name": tool["name"],
            "description": tool.get("description", "").strip(),
            "parameters": params,
        })
    missing = exposed - {d["name"] for d in declarations}
    if missing:
        raise RuntimeError(f"Flashpoint MCP did not list expected tools: {sorted(missing)}")
    return declarations


def _to_plain(value: Any) -> Any:
    """Proto-plus args come back as floats for integers; restore integral values."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list) or hasattr(value, "__iter__") and not isinstance(value, str):
        return [_to_plain(v) for v in value]
    return value


def run_flashpoint_agent(
    task: str,
    mcp: Optional[GeminiFlashpointMCPClient] = None,
    model_name: str = GEMINI_MODEL,
    max_turns: int = 8,
    tools: Optional[Set[str]] = None,
    check_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Let Gemini drive Flashpoint MCP tools to complete `task`.

    `tools` narrows which Flashpoint tools Gemini sees. `check_call(tool, args)` runs before each
    call is paid for; if it raises ValueError the call is not sent and Gemini is told why.

    Returns {"model", "answer", "tool_calls": [{"tool", "args", "ok", "result"?, "error"?}], "spent_usdc"}.
    Raises if Gemini is not configured, the tool list is wrong, or the turn budget runs out.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not set")

    mcp = mcp or GeminiFlashpointMCPClient()
    declarations = build_declarations(mcp.list_tools(), tools or EXPOSED_TOOLS)

    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=[{"function_declarations": declarations}],
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat()
    response = chat.send_message(task)

    calls: List[Dict[str, Any]] = []
    for _ in range(max_turns):
        function_calls = [p.function_call for p in response.candidates[0].content.parts if p.function_call.name]
        if not function_calls:
            break
        replies = []
        for fc in function_calls:
            args = _to_plain(fc.args)
            record: Dict[str, Any] = {"tool": fc.name, "args": args}
            try:
                if check_call is not None:
                    check_call(fc.name, args)
                result = mcp.call_mcp_tool(fc.name, args)
                record["ok"] = True
                record["result"] = result
                reply = {"result": result}
            except (RuntimeError, ValueError) as e:
                record["ok"] = False
                record["error"] = str(e)
                reply = {"error": str(e)}
            calls.append(record)
            replies.append(genai.protos.Part(
                function_response=genai.protos.FunctionResponse(name=fc.name, response=reply)
            ))
        response = chat.send_message(replies)
    else:
        raise RuntimeError(f"Gemini agent used all {max_turns} turns without finishing")

    answer = "".join(p.text for p in response.candidates[0].content.parts if p.text)
    return {"model": model_name, "answer": answer, "tool_calls": calls, "spent_usdc": mcp.spent_usdc}
