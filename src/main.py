import requests
from typing import Callable, Any
from dataclasses import dataclass, field

from rich.console import Console
import datetime
import getpass


@dataclass
class Agent:
    system_prompt: str = "You are a helpful assistant"
    model: str = "qwen3.5"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = field(default="NO_API_KEY", repr=False)
    contexts: dict[str, Callable[[],str]] = field(default_factory=dict)
    messages: list[dict[str,Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    # Context decorator
    def context(self, func: Callable[[],str]) -> Callable[[],str]:
        self.contexts[func.__name__] = func
        return func

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
        self.messages.append({"role": "assistant", "content": response})
        return response

# # Non-dynamic Context
# def main() -> None:
#     agent = Agent(
#         model="qwen3.5",
#         system_prompt="End every message with a yo mama joke.",
#     )
#
#     @agent.context
#     def user_context() -> str:
#         return(
#             f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#             f"Current user: {getpass.getuser()}\n"
#         )
#
#     console = Console()
#     with console.status("[dim]Thinking...[/dim]", spinner="arc"):
#         response = agent.chat("What time is it and who am I?")
#
#     console.print(f"[blue]You:[/blue] What time is it and who am I?")
#     console.print(f"[blue]Assistant:[/blue] {response}")

# # Dynamic Context
# def main() -> None:
#     agent = Agent(
#         model="qwen3.5",
#         system_prompt="You are a helpful assistant that always mentions the current hour.",
#     )
#
#     @agent.context
#     def time_context() -> str:
#         now = datetime.datetime.now()
#         return f"The current hour is {now.hour}:00"
#
#     console = Console()
#     with console.status("[dim]Thinking...[/dim]", spinner="arc"):
#         response = agent.chat("What hour is it?")
#
#     console.print(f"[blue]You:[/blue] What hour is it?")
#     console.print(f"[blue]Assistant:[/blue] {response}")
#
#     console.print(f"[dim]Making second request (context regenerated)...[/dim]")
#     with console.status("[dim]Thinking...[/dim]", spinner="arc"):
#         response2 = agent.chat("Tell me the hour again.")
#
#     console.print(f"[blue]You:[/blue] Tell me the hour again.")
#     console.print(f"[blue]Assistant:[/blue] {response2}")

def main() -> None:
    agent = Agent(
        model="qwen3.5",
        system_prompt="You are a helpful assistant that highlights the current time.",
    )

    # everytime Agent is instantiated, this user_context is invoked and saved under context and then under prefix
    @agent.context
    def user_context() -> str:
        return(
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Current user: {getpass.getuser()}\n"
        )

    console = Console()

    while True:
        console.print("[green]You:[/green] ", end="")
        user_input = console.input()
        if user_input.strip().lower() in {"quit", "exit"}:
            console.print("[dim]Goodbye![/dim]")
            return
        with console.status("[dim]Thinking...[/dim]", spinner="arc"):
            response = agent.chat(user_input).strip()
        console.print(f"[blue]Assistant:[/blue] {response}")


if __name__ == '__main__':
    main()