#!/usr/bin/env python3
"""
Migration script to update users from GPT-5 to GPT-5.1 with appropriate defaults.

This script:
1. Finds all users with model="gpt-5" (the old default)
2. Updates them to model="gpt-5.1" with reasoning_effort="none"
3. Leaves users with other models (gpt-4.1, gpt-5-mini, etc.) unchanged

Run this script after updating to GPT-5.1 support to migrate existing users.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database import DatabaseManager
from config import config


def migrate_users_to_gpt51():
    """Migrate users from GPT-5 to GPT-5.1 with appropriate defaults"""

    print("=" * 70)
    print("GPT-5 to GPT-5.1 User Migration Script")
    print("=" * 70)
    print()

    # Connect to database
    db_path = Path(config.database_dir) / "slack.db"
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("   No migration needed - database will be created with new defaults.")
        return

    print(f"📂 Connecting to database: {db_path}")
    db = DatabaseManager("slack")

    try:
        # Get all users with their current preferences
        print("\n🔍 Scanning for users with model='gpt-5'...")

        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT slack_user_id, model, reasoning_effort
            FROM user_preferences
            WHERE model = 'gpt-5'
        """)

        users_to_migrate = cursor.fetchall()

        if not users_to_migrate:
            print("✅ No users found with model='gpt-5' - all up to date!")
            return

        print(f"\n📋 Found {len(users_to_migrate)} user(s) to migrate:")
        print()

        for slack_user_id, current_model, current_reasoning in users_to_migrate:
            print(f"  User: {slack_user_id}")
            print(f"    Current: model={current_model}, reasoning_effort={current_reasoning}")
            print(f"    New:     model=gpt-5.1, reasoning_effort=none")
            print()

        # Ask for confirmation
        response = input("Proceed with migration? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n❌ Migration cancelled by user.")
            return

        print("\n🔄 Migrating users...")

        # Update each user
        migrated_count = 0
        for slack_user_id, _, _ in users_to_migrate:
            try:
                cursor.execute("""
                    UPDATE user_preferences
                    SET model = 'gpt-5.1',
                        reasoning_effort = 'none'
                    WHERE slack_user_id = ?
                """, (slack_user_id,))
                migrated_count += 1
                print(f"  ✓ Migrated user: {slack_user_id}")
            except Exception as e:
                print(f"  ✗ Failed to migrate user {slack_user_id}: {e}")

        # Commit changes
        db.conn.commit()

        print()
        print("=" * 70)
        print(f"✅ Migration complete!")
        print(f"   Successfully migrated {migrated_count} of {len(users_to_migrate)} users")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.conn.close()


if __name__ == "__main__":
    try:
        migrate_users_to_gpt51()
    except KeyboardInterrupt:
        print("\n\n❌ Migration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
