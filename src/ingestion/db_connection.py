import sqlite3
import yaml
import os

def test_connection():
    # Load config to get the db path
    """
    this module is responsible for establishing a connection 
    to the SQLite database using the path specified in the config.yaml file. 
    It reads the configuration, retrieves the database path, and attempts to connect to the database,
    printing a success message if the connection is established or an error message if it fails.
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