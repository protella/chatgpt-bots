#!/usr/bin/env python3
"""
Script to update all GPT-5.1 users to reasoning_effort="low" as the default.

This script:
1. Finds all users with model="gpt-5.1"
2. Updates their reasoning_effort to "low" (regardless of current value)
3. Users can manually change their reasoning level later via settings if desired

Run this script to standardize reasoning_effort for GPT-5.1 users.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database import DatabaseManager
from config import config


def update_gpt51_reasoning():
    """Update all GPT-5.1 users to reasoning_effort='low'"""

    print("=" * 70)
    print("GPT-5.1 Reasoning Effort Update Script")
    print("=" * 70)
    print()

    # Connect to database
    db_path = Path(config.database_dir) / "slack.db"
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("   No update needed - database will be created with new defaults.")
        return

    print(f"📂 Connecting to database: {db_path}")
    db = DatabaseManager("slack")

    try:
        # Get all users with GPT-5.1
        print("\n🔍 Scanning for users with model='gpt-5.1'...")

        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT slack_user_id, model, reasoning_effort
            FROM user_preferences
            WHERE model = 'gpt-5.1'
        """)

        users_to_update = cursor.fetchall()

        if not users_to_update:
            print("✅ No users found with model='gpt-5.1' - nothing to update!")
            return

        print(f"\n📋 Found {len(users_to_update)} user(s) to update:")
        print()

        for slack_user_id, current_model, current_reasoning in users_to_update:
            print(f"  User: {slack_user_id}")
            print(f"    Current: model={current_model}, reasoning_effort={current_reasoning}")
            print(f"    New:     model={current_model}, reasoning_effort=low")
            print()

        # Ask for confirmation
        response = input("Proceed with update? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n❌ Update cancelled by user.")
            return

        print("\n🔄 Updating users...")

        # Update each user
        updated_count = 0
        for slack_user_id, _, _ in users_to_update:
            try:
                cursor.execute("""
                    UPDATE user_preferences
                    SET reasoning_effort = 'low'
                    WHERE slack_user_id = ? AND model = 'gpt-5.1'
                """, (slack_user_id,))
                updated_count += 1
                print(f"  ✓ Updated user: {slack_user_id}")
            except Exception as e:
                print(f"  ✗ Failed to update user {slack_user_id}: {e}")

        # Commit changes
        db.conn.commit()

        print()
        print("=" * 70)
        print(f"✅ Update complete!")
        print(f"   Successfully updated {updated_count} of {len(users_to_update)} users")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.conn.close()


if __name__ == "__main__":
    try:
        update_gpt51_reasoning()
    except KeyboardInterrupt:
        print("\n\n❌ Update interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
