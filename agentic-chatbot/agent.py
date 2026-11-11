"""Agentic chatbot: GPT tool use in a loop + prompt chaining (plan -> act -> respond)."""
import json, math, datetime
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"


# ---- tools ----
def calculator(expression: str) -> str:
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))  # noqa: S307 (restricted namespace)
    except Exception as e:
        return f"error: {e}"

def get_time() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

NOTES: list[str] = []
def save_note(text: str) -> str:
    NOTES.append(text); return f"saved note #{len(NOTES)}"

def list_notes() -> str:
    return json.dumps(NOTES)

REGISTRY = {"calculator": calculator, "get_time": get_time, "save_note": save_note, "list_notes": list_notes}

def _fn(name, desc, props=None):
    props = props or {}
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}

TOOLS = [
    _fn("calculator", "Evaluate a math expression.", {"expression": {"type": "string"}}),
    _fn("get_time", "Get the current date and time."),
    _fn("save_note", "Save a note for later.", {"text": {"type": "string"}}),
    _fn("list_notes", "List saved notes."),
]


# ---- prompt chain ----
def plan(task: str) -> str:
    r = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "Break the user's task into 1-4 short numbered steps. Mention which tools could help."},
        {"role": "user", "content": task}])
    return r.choices[0].message.content

def act(task: str, plan_text: str, max_steps: int = 6) -> str:
    msgs = [{"role": "system", "content": "You are an autonomous assistant. Follow the plan, use tools when needed, then give the result."},
            {"role": "user", "content": f"Task: {task}\n\nPlan:\n{plan_text}"}]
    for _ in range(max_steps):
        r = client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS)
        m = r.choices[0].message
        msgs.append(m)
        if not m.tool_calls:
            return m.content
        for call in m.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            out = REGISTRY[call.function.name](**args)
            print(f"  [tool] {call.function.name}({args}) -> {out}")
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return "Stopped: step limit reached."

def review(task: str, draft: str) -> str:
    r = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system", "content": "Check the draft answers the task accurately. Return a corrected, concise final answer."},
        {"role": "user", "content": f"Task: {task}\n\nDraft:\n{draft}"}])
    return r.choices[0].message.content

def run(task: str) -> str:
    p = plan(task); print("PLAN:\n", p)
    return review(task, act(task, p))


if __name__ == "__main__":
    print("Agentic chatbot (type 'quit' to exit)")
    while (t := input("\nYou: ").strip()).lower() not in {"quit", "exit"}:
        print("\nBot:", run(t))
