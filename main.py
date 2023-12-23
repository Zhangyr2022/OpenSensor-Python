import asyncio
import logging
import os
from dotenv import load_dotenv
from backend import Backend
from rich.console import Console
from rich.logging import RichHandler
import time

load_dotenv()

DEFAULT_FILTER_NAME = "KalmanFilter"
DEFAULT_LOGGING_LEVEL = "INFO"
DEFAULT_SERVER_PORT = 39999
SERILOG_TEMPLATE = "%(message)s"
STATE_DIMENSION = 3
BUFFER_SIZE = 5
REAL_TIME_PLOT = False
CSV_PATH = f"./data/{time.strftime('%Y%m%d%H%M%S', time.localtime())}.csv"


async def main_async():
    filter_name = os.getenv("FILTER_NAME", DEFAULT_FILTER_NAME)
    logging_level_string = os.getenv("LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL)
    server_port = int(os.getenv("SERVER_PORT", DEFAULT_SERVER_PORT))

    setup_logging(logging_level_string)

    backend = Backend(
        server_port,
        filter_name,
        STATE_DIMENSION,
        BUFFER_SIZE,
        real_time_plot=REAL_TIME_PLOT,
        csv_path=CSV_PATH,
    )

    # Use asyncio.create_task to start the backend asynchronously
    asyncio.create_task(backend.start())

    # Add an exit condition or use asyncio.sleep() to let the loop yield
    await asyncio.sleep(1)


def setup_logging(logging_level_string: str):
    # Configure RichHandler for colorful console logging
    console = Console()
    rich_handler = RichHandler(console=console, show_time=True)
    rich_handler.setFormatter(logging.Formatter(fmt=SERILOG_TEMPLATE))

    # Configure Python logging
    logging.basicConfig(
        level=logging.getLevelName(logging_level_string),
        handlers=[rich_handler],
    )

    # Configure websockets logging
    logging.getLogger("websockets").addHandler(rich_handler)

    logging.info(f"Logging level set to {logging_level_string}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.create_task(main_async())
        loop.run_forever()  # Or use await main_async() here
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C interruption
    finally:
        loop.close()
