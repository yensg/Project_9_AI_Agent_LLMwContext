# Build a Local Agent in Python with Context — Part 2

> Notes based on the tutorial by [Indently](https://www.youtube.com/@Indently?utm_source=chatgpt.com)
> Topic: Dynamic Context Injection for Local LLM Agents

---

# Table of Contents

1. Introduction
2. Why Local LLMs Need Dynamic Context
3. Understanding `Callable`
4. Building Dynamic Context Injection
5. Context Decorators
6. Structured Context Injection
7. Stateless vs Stateful Context
8. Request Flow Architecture
9. Dynamic Context Regeneration
10. Interactive Chat Loop
11. Why the Model Sometimes Ignores Context
12. Key Takeaways

---

# Introduction

In Part 1, the local agent could chat with a local LLM through an OpenAI-compatible API.

In Part 2, the goal is to make the agent smarter by injecting **fresh realtime context** into every request.

This solves one major limitation of local LLMs:

* They do not know the current time
* They do not know the current user
* They do not know live system state
* They do not automatically access external realtime information

So instead of relying purely on model memory, we inject context manually before every request.

This pattern is called:

* **Dynamic Context Injection**
* **Prompt-based Context Injection**

---

# Why Local LLMs Need Dynamic Context

A local model is frozen at training time.

Without external context:

```text
User: What time is it?
LLM: I do not know.
```

So we inject realtime data ourselves:

```python
Current date and time: 2026-05-09 00:46:22
Current user: yenlim
```

The agent then sends this together with the user message.

---

# Understanding `Callable`

```python
from typing import Dict, Callable, Any

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]):
        self.tools[name] = fn
```

## What is `Callable`?

`Callable` is a typing annotation used to describe functions.

Unlike normal types:

```python
int
str
list
```

`Callable` describes:

* function inputs
* function outputs

---

## Structure

```python
Callable[[arguments], return_type]
```

Example:

```python
Callable[[int, int], int]
```

Means:

```text
A function that:
- accepts two integers
- returns one integer
```

---

## Example

```python
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> str:
    return f"Hello {name}"

registry = ToolRegistry()
registry.register("add", add)
registry.register("greet", greet)
```

Internally:

```python
self.tools = {
    "add": add,
    "greet": greet
}
```

So the registry stores actual function objects.

---

# Building Dynamic Context Injection

The core logic happens inside:

```python
def chat(self, user_message) -> str:
```

---

## Full Flow

```python
def chat(self, user_message) -> str:
    self.messages.append({"role": "user", "content": user_message})

    context_content = "\n\n".join(
        f"<context>\n<{n}>{fn()}</{n}>\n</context>"
        for n, fn in self.contexts.items()
    )

    prefix: list[dict[str, Any]] = [
        {"role": "system", "content": self.system_prompt},
        {"role": "system", "content": context_content},
    ]

    url = f"{self.base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        url=url,
        headers=headers,
        json={"model": self.model, "messages": prefix + self.messages},
        timeout=300,
    )

    r.raise_for_status()

    data = r.json()
    choices = data.get("choices")

    if not choices:
        raise RuntimeError("Model response missing choices")

    message = choices[0].get("message")

    if message is None:
        raise RuntimeError("Model response missing message")

    response = message.get("content") or ""

    self.messages.append({
        "role": "assistant",
        "content": response
    })

    return response
```

---

# Context Decorators

```python
def context(self, func: Callable[[], str]) -> Callable[[], str]:
    self.contexts[func.__name__] = func
    return func
```

---

## Why Decorators?

Most modern frameworks use decorators because they:

* scale better
* reduce manual wiring
* improve organization
* automatically register functionality

---

## Without Decorator

```python
class Registry:
    def __init__(self):
        self.contexts = {}

    def register(self, func):
        self.contexts[func.__name__] = func

registry = Registry()

def get_weather():
    return "sunny"

registry.register(get_weather)
```

You must:

1. define function
2. manually register function

---

## With Decorator

```python
class Registry:
    def __init__(self):
        self.contexts = {}

    def context(self, func):
        self.contexts[func.__name__] = func
        return func

registry = Registry()

@registry.context
def get_weather():
    return "sunny"
```

Now registration happens automatically during function definition.

---

# Structured Context Injection

## `context_content`

```python
context_content = "\n\n".join(
    f"<context>\n<{n}>{fn()}</{n}>\n</context>"
    for n, fn in self.contexts.items()
)
```

This:

1. runs every registered context function
2. converts output into structured text
3. injects it into the LLM prompt

---

## Result Example

```xml
<context>
<user_context>
Current date and time: 2026-05-09 00:46:22
Current user: yenlim
</user_context>
</context>
```

---

## Why Structured Tags Matter

Structured formatting improves:

* clarity
* predictability
* parsing consistency
* prompt reliability

Instead of:

```text
The current time is 3PM and user is Bob.
```

You provide:

```xml
<context>
<time>3PM</time>
<user>Bob</user>
</context>
```

This gives the LLM cleaner semantic boundaries.

---

# Stateless vs Stateful Context

This is extremely important architecturally.

---

## `context_content`

```python
context_content = "what is true RIGHT NOW"
```

Examples:

* current time
* current user
* CPU usage
* live stock price
* weather

This is:

* dynamic
* temporary
* regenerated every request

This is **stateless injection**.

---

## `ConversationContext`

From your Flight Assistant architecture:

```python
ConversationContext = "what we know OVER TIME"
```

Examples:

* previous intent
* remembered airport
* user preferences
* extracted entities
* unresolved slots

This is:

* persistent
* accumulative
* feedback-based

This is a **stateful feedback loop**.

---

# Request Flow Architecture

The final request structure becomes:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "system", "content": context_content},
    ...
    previous messages ...
]
```

---

## `prefix + self.messages`

```python
prefix = [
    {"role": "system", "content": "You are assistant"}
]

messages = [
    {"role": "user", "content": "Hello"}
]

prefix + messages
```

Result:

```python
[
    {"role": "system", "content": "You are assistant"},
    {"role": "user", "content": "Hello"}
]
```

This merges two lists into one flat conversation list.

---

# HTTP Response Handling

## `r.raise_for_status()`

```python
r.raise_for_status()
```

This converts HTTP errors into Python exceptions.

Example:

```text
404 → raises exception
500 → raises exception
```

Without this:

* requests may silently fail
* debugging becomes harder

---

## `data = r.json()`

```python
data = r.json()
```

This parses the HTTP response body into Python data structures.

Example:

```json
{
  "choices": [...]
}
```

Becomes:

```python
{
  "choices": [...]
}
```

---

# `main()` Flow

```python
def main() -> None:
    agent = Agent(
        model="qwen3.5",
        system_prompt="End every message with a yo mama joke.",
    )

    @agent.context
    def user_context() -> str:
        return(
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current user: {getpass.getuser()}\n"
        )

    response = agent.chat("What time is it and who am I?")
```

---

## What Happens Internally

```text
@agent.context
↓
register function into agent.contexts
↓
chat()
↓
run all context functions
↓
inject outputs into prompt
↓
send to LLM
```

---

# `import getpass`

```python
import getpass

password = getpass.getpass("Enter password: ")
```

Characters become hidden:

```text
Enter password: ********
```

Useful for:

* passwords
* API keys
* secrets

---

# Dynamic Context Regeneration

## Calling `chat()` Twice

```python
@agent.context
def time_context() -> str:
    now = datetime.datetime.now()
    return f"The current hour is {now.hour}:00"
```

Every call to:

```python
agent.chat(...)
```

re-runs:

```python
time_context()
```

This means the context is freshly regenerated every request.

---

# Interactive Chat Loop

```python
while True:
    console.print("[green]You:[/green] ", end="")
    user_input = console.input()

    if user_input.strip().lower() in {"quit", "exit"}:
        return

    response = agent.chat(user_input).strip()
```

This creates a persistent CLI chat experience.

---

# Why the Model Sometimes Ignores Context

Example:

```text
You: what is the equation for energy?
```

The model answered correctly:

E = mc^2

But ignored the instruction to mention the current time.

---

## Why This Happens

LLMs do not execute rules deterministically.

They perform:

```text
probabilistic next-token prediction
```

So instructions compete against each other.

The model prioritizes:

1. answering the user's main request
2. maintaining coherence
3. following system instructions

Sometimes:

```text
energy equation
```

becomes more semantically important than:

```text
mention the current time
```

---

## Why Stronger Models Behave Better

More capable models are better at:

* instruction hierarchy
* long-context retention
* multi-objective prompting
* system prompt obedience

Smaller local models often:

* forget instructions
* lose context
* prioritize recent tokens
* ignore secondary requirements

---

# Key Takeaways

## Dynamic Context Injection

You manually inject fresh realtime data into prompts before every request.

---

## Decorators

Decorators automatically register functions during definition.

---

## Structured Context

Structured XML-like tags improve reliability and clarity.

---

## Stateless vs Stateful Context

| Type                  | Purpose            |
| --------------------- | ------------------ |
| `context_content`     | realtime truth     |
| `ConversationContext` | accumulated memory |

---

## Local LLM Limitation

Local models do not inherently know:

* current time
* current user
* live information

You must inject those manually.

---

## Architectural Evolution

Your learning path is progressing through these stages:

```text
Simple Prompting
↓
Structured Prompting
↓
Context Injection
↓
Tool Calling
↓
JSON Schema Enforcement
↓
Stateful Agent Systems
↓
Agentic Orchestration
```

---

## Related Project Context

Your current Flight Assistant architecture already moved beyond simple prompt injection into:

* tool calling
* schema validation
* orchestration loops
* persistent conversation state
* stateful memory systems

So this tutorial represents the foundational pattern underneath many modern agent architectures.

---

## Additional Resource

Microsoft's beginner-friendly agentic AI repository:

[AI Agents for Beginners (GitHub)](https://github.com/microsoft/ai-agents-for-beginners?utm_source=chatgpt.com) 
