"""
Test harness — executes all 37 SQL business questions against the loaded
database and reports pass/fail per query.
"""
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "database" / "sql_business_questions.sql"


def main():
    sql = SQL_FILE.read_text()
    parts = re.split(r"^--\s*(\d{2})\.\s.*$", sql, flags=re.M)
    conn = psycopg2.connect(host="localhost", dbname="netflix",
                            user="postgres", password="postgres")
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=RealDictCursor)

    passed, failed = [], []
    for i in range(1, len(parts), 2):
        num = parts[i].strip()
        body = re.sub(r"--.*$", "", parts[i + 1], flags=re.M).strip()
        try:
            cur.execute(body)
            rows = cur.fetchall()
            passed.append((num, len(rows)))
        except Exception as e:
            failed.append((num, str(e).split("\n")[0]))

    print(f"PASSED: {len(passed)}/37")
    for n, r in passed:
        print(f"  Q{n}: OK ({r} rows)")
    if failed:
        print(f"\nFAILED: {len(failed)}")
        for n, e in failed:
            print(f"  Q{n}: {e}")
        sys.exit(1)
    print("\nAll 37 SQL business questions pass.")
    conn.close()


if __name__ == "__main__":
    main()
