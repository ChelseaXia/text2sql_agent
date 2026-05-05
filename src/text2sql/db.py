"""SQLite execution helpers for local evaluation."""

import sqlite3
import time
from collections import Counter
from pathlib import Path

from text2sql.config import DEFAULT_SQL_TIMEOUT


def _normalize_value(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows):
    normalized = []
    for row in rows:
        normalized.append(tuple(_normalize_value(value) for value in row))
    return Counter(normalized)


def same_result(pred_rows, gold_rows):
    return normalize_rows(pred_rows) == normalize_rows(gold_rows)


def run_sql(sql, db_path, timeout=DEFAULT_SQL_TIMEOUT):
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
        connection = sqlite3.connect(str(db_file.resolve()), timeout=timeout)
        connection.row_factory = None
        connection.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")

        def _progress_handler():
            if time.monotonic() - started_at > timeout:
                return 1
            return 0

        connection.set_progress_handler(_progress_handler, 10_000)
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        return {"success": True, "rows": rows, "error": None}
    except sqlite3.OperationalError as exc:
        error_message = str(exc)
        if "interrupted" in error_message.lower():
            error_message = f"SQL execution timed out after {timeout}s"
        return {"success": False, "rows": [], "error": error_message}
    except Exception as exc:
        return {"success": False, "rows": [], "error": str(exc)}
    finally:
        if connection is not None:
            connection.close()
