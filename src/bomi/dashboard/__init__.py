"""bomi interactive dashboard — FastAPI backend + Svelte/D3 frontend."""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import pandas as pd


def launch(df: pd.DataFrame | None = None, port: int = 8050, open_browser: bool = True) -> None:
    """Start the bomi dashboard server.

    Parameters
    ----------
    df:
        Pre-loaded board DataFrame. When provided the dashboard opens with that
        board already active. When ``None`` the user loads data via the UI.
    port:
        TCP port for the local server.
    open_browser:
        Whether to automatically open a browser tab.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "Dashboard dependencies not installed. Run: pip install 'bomi[dashboard]'"
        ) from exc

    from .server.main import create_app
    from .server import loader

    if df is not None:
        loader.set_board(df)

    app = create_app()

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    """CLI entry point: ``bomi-dashboard``."""
    parser = argparse.ArgumentParser(
        prog="bomi-dashboard",
        description="Launch the bomi interactive board dashboard.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="CSV or Trello JSON file to load on startup.",
    )
    parser.add_argument(
        "--board-id",
        metavar="ID",
        help="Trello board ID or URL slug to fetch on startup.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Local port (default: 8050).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )
    args = parser.parse_args()

    df: pd.DataFrame | None = None

    if args.file:
        path = Path(args.file)
        if path.suffix == ".json":
            from bomi.dashboard.server.loader import load_from_json_bytes
            df = load_from_json_bytes(path.read_bytes())
        else:
            from bomi.dashboard.server.loader import load_from_csv_bytes
            df = load_from_csv_bytes(path.read_bytes(), path.name)
    elif args.board_id:
        from bomi.dashboard.server.loader import load_from_board_id
        print(f"Fetching board {args.board_id} …")
        df = load_from_board_id(args.board_id)

    launch(df, port=args.port, open_browser=not args.no_browser)
