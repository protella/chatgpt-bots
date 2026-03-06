"""
Migration script: Bump all users to GPT-5.4

Changes:
  1. model → 'gpt-5.4' for all users
  2. reasoning_effort 'minimal' → 'low' (minimal is not valid on GPT-5.4)

Usage:
  python3 migrate_to_gpt54.py              # dry run (default)
  python3 migrate_to_gpt54.py --apply      # apply changes
  python3 migrate_to_gpt54.py --db path    # specify db path (default: data/slack.db)
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def migrate(db_path: str, dry_run: bool = True):
    if not Path(db_path).exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_preferences'"
    )
    if not cursor.fetchone():
        print("ERROR: user_preferences table does not exist")
        conn.close()
        sys.exit(1)

    # Show current state
    rows = conn.execute(
        "SELECT slack_user_id, model, reasoning_effort FROM user_preferences ORDER BY slack_user_id"
    ).fetchall()

    print(f"Database: {db_path}")
    print(f"Total users: {len(rows)}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLYING CHANGES'}\n")

    # --- Migration 1: model → gpt-5.4 ---
    model_update = conn.execute(
        "SELECT COUNT(*) FROM user_preferences WHERE model != 'gpt-5.4'"
    ).fetchone()[0]
    print(f"Users needing model update to gpt-5.4: {model_update}")
    if model_update > 0:
        breakdown = conn.execute(
            "SELECT model, COUNT(*) as cnt FROM user_preferences WHERE model != 'gpt-5.4' GROUP BY model"
        ).fetchall()
        for row in breakdown:
            print(f"  {row['model']}: {row['cnt']} users")

    # --- Migration 2: reasoning_effort minimal → low ---
    reasoning_update = conn.execute(
        "SELECT COUNT(*) FROM user_preferences WHERE reasoning_effort = 'minimal'"
    ).fetchone()[0]
    print(f"\nUsers with reasoning_effort='minimal' (invalid for 5.4, will become 'low'): {reasoning_update}")

    # No other fields need migration - verbosity (low/medium/high) and temp/top_p are valid across models

    if dry_run:
        print("\n--- DRY RUN - no changes made. Run with --apply to execute. ---")
        conn.close()
        return

    # Apply changes
    print("\nApplying migrations...")

    c1 = conn.execute("UPDATE user_preferences SET model = 'gpt-5.4' WHERE model != 'gpt-5.4'")
    print(f"  Model updated: {c1.rowcount} rows")

    c2 = conn.execute("UPDATE user_preferences SET reasoning_effort = 'low' WHERE reasoning_effort = 'minimal'")
    print(f"  Reasoning effort fixed: {c2.rowcount} rows")

    conn.commit()
    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate user preferences to GPT-5.4")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--db", default="data/slack.db", help="Path to database (default: data/slack.db)")
    args = parser.parse_args()

    migrate(args.db, dry_run=not args.apply)
