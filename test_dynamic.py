import random
import time
from collections import deque
from collections.abc import Generator
from typing import cast, override

from rich.align import Align
from rich.console import Console, ConsoleOptions
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class MessagePanel(Panel):
    """A panel to display a live log of messages."""

    def __init__(
        self,
        max_messages: int = 10,
        title: str = "[bold green]Live Message Log[/bold green]",
        border_style: str = "green",
    ) -> None:
        """Initialize the MessagePanel.

        Args:
            max_messages: Maximum number of messages to display.
            title: Title of the panel.
            border_style: Border style for the panel.
            **kwargs: Additional keyword arguments for Panel.

        """
        self.messages: deque[str] = deque(maxlen=max_messages)
        self.max_messages: int = max_messages
        self.console: Console = Console()

        super().__init__(
            Align.center(Text("No messages yet...", style="dim italic")),
            title=title,
            border_style=border_style,
            expand=True,
            subtitle="[dim]Displaying last 0 messages[/dim]",
        )

    def add_message(self, message: str) -> None:
        """Add a message to the panel.

        Args:
            message: The message string to add.

        """
        self.messages.append(message)

    @override
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> Generator[str]:
        """Render the panel with the latest messages.

        Args:
            console: The console instance.
            options: Console rendering options.

        Yields:
            RenderableType: The rendered panel.

        """
        if not self.messages:
            inner_content = Align.center(Text("No messages yet...", style="dim italic"))
        else:
            message_lines: list[Text] = []
            for i, msg in enumerate(self.messages):
                if i == len(self.messages) - 1:
                    message_lines.append(Text(f"• {msg}", style="bold yellow"))
                elif i == len(self.messages) - 2:
                    message_lines.append(Text(f"• {msg}", style="orange3"))
                else:
                    message_lines.append(Text(f"• {msg}", style="grey50"))
            inner_content = Text("\n").join(message_lines)

        self.renderable: str = inner_content
        self.subtitle: Text | str | None = Text(
            f"Displaying last {len(self.messages)} messages",
            style="dim",
        )

        yield from cast(
            "Generator[str]",
            super().__rich_console__(console, options),
        )


def main():
    console = Console()
    log_panel = MessagePanel(max_messages=10)

    example_messages = [
        "System startup initiated.",
        "Database connection established.",
        "User 'john.doe' logged in.",
        "Processing request ID: 12345.",
        "Data validation successful.",
        "API endpoint /status accessed.",
        "Job 'daily_report' started.",
        "Queue depth: 5 items.",
        "Service 'auth' responded in 150ms.",
        "Disk usage: 75%.",
        "New user registration: 'jane.smith'.",
        "Order #54321 placed successfully.",
        "Warning: High CPU utilization (85%).",
        "Critical: Service 'payment' down!",
        "Recovery initiated for 'payment' service.",
        "Payment service restored. Monitoring...",
        "User 'alice' updated profile.",
        "Configuration reloaded.",
        "Backup process completed.",
        "Application shutting down.",
    ]
    message_index = 0

    with Live(log_panel, refresh_per_second=4, console=console) as live:
        try:
            while True:
                if message_index < len(example_messages):
                    msg = example_messages[message_index]
                    log_panel.add_message(msg)
                    message_index += 1
                else:
                    random_msg = f"Random event {random.randint(1000, 9999)} occurred at {time.strftime('%H:%M:%S')}"
                    log_panel.add_message(random_msg)

                time.sleep(1.5)
        except KeyboardInterrupt:
            console.print("\n[bold red]Exiting live panel.[/bold red]")


if __name__ == "__main__":
    main()
