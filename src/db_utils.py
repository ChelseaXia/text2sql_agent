"""SQLite execution helpers for BIRD local evaluation."""

import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from config import DEFAULT_SQL_TIMEOUT
except ModuleNotFoundError:
    from src.config import DEFAULT_SQL_TIMEOUT


def _normalize_value(value):
    """Normalize scalar values so result comparison is less brittle."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows: Iterable[Sequence]) -> Counter:
    """Normalize SQL rows into a multiset for order-insensitive comparison."""
    normalized = []
    for row in rows:
        normalized.append(tuple(_normalize_value(value) for value in row))
    return Counter(normalized)


def same_result(pred_rows: List[Sequence], gold_rows: List[Sequence]) -> bool:
    """Return whether two SQL execution results are equivalent."""
    return normalize_rows(pred_rows) == normalize_rows(gold_rows)


def run_sql(sql: str, db_path: str, timeout: int = DEFAULT_SQL_TIMEOUT) -> dict:
    """Execute a SQLite query and return rows or a clear error payload."""
    db_file = Path(db_path)
    if not db_file.exists():
        return {
            "success": False,
            "rows": [],
            "error": f"Database file not found: {db_file}",
        }

    connection = None
    started_at = time.monotonic()

    try:
        # Use a normal absolute path because SQLite URI mode can be brittle on
        # some local macOS setups for large files.
        connection = sqlite3.connect(str(db_file.resolve()), timeout=timeout)
        connection.row_factory = None
        connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")

        def _progress_handler() -> int:
            if time.monotonic() - started_at > timeout:
                return 1
            return 0

        connection.set_progress_handler(_progress_handler, 10_000)
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()

        return {
            "success": True,
            "rows": rows,
            "error": None,
        }
    except sqlite3.OperationalError as exc:
        error_message = str(exc)
        if "interrupted" in error_message.lower():
            error_message = f"SQL execution timed out after {timeout}s"
        return {
            "success": False,
            "rows": [],
            "error": error_message,
        }
    except Exception as exc:
        return {
            "success": False,
            "rows": [],
            "error": str(exc),
        }
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    try:
        from load_bird import load_bird_dev
    except ModuleNotFoundError:
        from src.load_bird import load_bird_dev

    samples = load_bird_dev(limit=1)
    sample = samples[0]
    result = run_sql(sample["gold_sql"], sample["db_path"])

    print("Question:", sample["question"])
    print("DB Path:", sample["db_path"])
    print("SQL:", sample["gold_sql"])
    print("Success:", result["success"])
    print("Rows preview:", result["rows"][:5])
    print("Error:", result["error"])
