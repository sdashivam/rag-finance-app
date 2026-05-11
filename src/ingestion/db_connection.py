"""
SQLite database connection testing and configuration utilities.

Provides connection validation for the financial table storage layer.
Used primarily for debugging and environment setup verification.
"""

import sqlite3
import yaml
import os


def test_connection():
    """Validates database connectivity using config.yaml settings.

    Reads db_path from configuration, attempts connection,
    and reports success or failure status.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_path = config.get('db_path')
    if db_path and not os.path.isabs(db_path):
        db_path = os.path.join(base_dir, db_path)
    
    try:
        # Establishing connection to SQLite
        conn = sqlite3.connect(db_path)
        print("Connected successfully to SQLite!")
        conn.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    test_connection()