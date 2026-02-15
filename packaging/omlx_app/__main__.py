"""Entry point for oMLX menubar app.

Wraps the app launch in comprehensive error handling so that failures
(missing dependencies, PyObjC issues, etc.) are shown to the user
via a native dialog instead of being silently swallowed.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path


def _get_crash_log_path() -> Path:
    """Get crash log file path in Application Support."""
    app_support = Path.home() / "Library" / "Application Support" / "oMLX"
    app_support.mkdir(parents=True, exist_ok=True)
    return app_support / "crash.log"


def _write_crash_log(exc_text: str) -> Path:
    """Append crash info to the crash log file."""
    crash_log = _get_crash_log_path()
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(crash_log, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Crash at {timestamp}\n")
            f.write(f"{'=' * 60}\n")
            f.write(exc_text)
            f.write("\n")
    except Exception:
        pass
    return crash_log


def _show_error_dialog(title: str, message: str) -> None:
    """Show error dialog via osascript (works without PyObjC).

    Uses AppleScript 'return' character for newlines in the dialog text.
    """
    import subprocess

    # Escape backslashes and double quotes for AppleScript string literal
    escaped = message.replace("\\", "\\\\").replace('"', '\\"')
    title_escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    # AppleScript: use 'return' for newlines inside display dialog
    script = (
        f'set msg to "{escaped}"\n'
        f'display dialog msg buttons {{"OK"}} '
        f'default button 1 with icon stop with title "{title_escaped}"'
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=60)
    except Exception:
        pass


try:
    from .app import main

    main()
except Exception as e:
    exc_text = traceback.format_exc()
    crash_log = _write_crash_log(exc_text)

    # Build a concise message for the dialog (full traceback is in crash log)
    error_line = str(e).replace("\n", " ")[:200]
    _show_error_dialog(
        "oMLX Launch Error",
        f"{type(e).__name__}: {error_line}\n\nCrash log: {crash_log}",
    )
    sys.exit(1)
