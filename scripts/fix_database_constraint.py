#!/usr/bin/env python3
"""
Fix script cho SQLite constraint error
Xóa database cũ và tạo lại với schema mới
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def backup_and_recreate_database():
    """Backup database cũ và tạo lại"""
    
    db_path = Path("data/chatbot_database.db")
    backup_dir = Path("data/backup_database_fix")
    
    print("🔧 Fixing SQLite constraint error...")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    
    # Backup existing database if exists
    if db_path.exists():
        backup_path = backup_dir / f"chatbot_database_backup_{int(os.path.getmtime(db_path))}.db"
        shutil.copy2(db_path, backup_path)
        print(f"📁 Backed up existing database to: {backup_path}")
        
        # Remove old database
        os.remove(db_path)
        print("🗑️  Removed old database with constraint issue")
    
    # Now create new database with correct schema
    try:
        from database.database_manager import get_database_manager
        
        print("🔧 Creating new database with fixed schema...")
        db_manager = get_database_manager()
        
        # Test basic operations
        session_id = db_manager.create_session()
        print(f"✅ Created test session: {session_id}")
        
        # Test system message (this was causing the error)
        msg_id = db_manager.add_message(session_id, "system", "Test system message")
        print(f"✅ Added system message: {msg_id}")
        
        # Test user and assistant messages
        user_msg_id = db_manager.add_message(session_id, "user", "Hello")
        assistant_msg_id = db_manager.add_message(session_id, "assistant", "Hi there!")
        
        print(f"✅ Added user message: {user_msg_id}")
        print(f"✅ Added assistant message: {assistant_msg_id}")
        
        # Verify messages
        messages = db_manager.get_chat_history(session_id)
        print(f"✅ Retrieved {len(messages)} messages from database")
        
        print("🎉 Database recreated successfully with fixed schema!")
        print("   - System messages are now supported")
        print("   - All role types (user, assistant, system) work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to recreate database: {e}")
        return False

def main():
    """Main function"""
    
    print("SQLite Constraint Fix Tool")
    print("=" * 40)
    
    # Check if database exists
    db_path = Path("data/chatbot_database.db")
    
    if db_path.exists():
        print(f"📁 Found existing database: {db_path}")
        confirm = input("Do you want to backup and recreate the database? (y/N): ")
        
        if confirm.lower().startswith('y'):
            success = backup_and_recreate_database()
            
            if success:
                print("\n✅ Fix completed successfully!")
                print("You can now run the chatbot without constraint errors.")
            else:
                print("\n❌ Fix failed. Please check the error messages above.")
        else:
            print("Operation cancelled.")
    else:
        print("📁 No existing database found. Creating new one...")
        success = backup_and_recreate_database()
        
        if success:
            print("\n✅ New database created successfully!")
        else:
            print("\n❌ Failed to create database.")

if __name__ == "__main__":
    main()
