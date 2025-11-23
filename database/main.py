
# main.py
"""
This is the main file for the database, this does the following:
1) Download the data file from the internet.
2) Set up the connection to the database.
3) Parse the data file and create the database.
4) Close the connection to the database.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

# Database connection configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345") # Default password, change as needed
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "motor_shield")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_database_if_not_exists():
    """Creates the database if it doesn't exist."""
    # Connect to default 'postgres' database to create the new one
    default_db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    
    try:
        engine = create_engine(default_db_url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"))
            if not result.fetchone():
                print(f"Database '{DB_NAME}' does not exist. Creating...")
                # Postgres does not support CREATE DATABASE IF NOT EXISTS directly in one statement easily
                # and we are in python so we can just check first.
                conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
                print(f"Database '{DB_NAME}' created successfully.")
            else:
                print(f"Database '{DB_NAME}' already exists.")
        engine.dispose()
    except Exception as e:
        print(f"Warning: Could not check/create database '{DB_NAME}'. Error: {e}")
        print("Please ensure the database exists manually if connection fails.")

def init_db(engine):
    """Reads the schema.sql file and creates the table."""
    print("Initializing database schema...")
    try:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        
        with engine.connect() as connection:
            # Drop tables to ensure clean schema update
            connection.execute(text("DROP TABLE IF EXISTS experiment_features, model_training_data, model_predictions, experiments, recordings, sensor_measurements, fault_classes CASCADE"))
            connection.commit()
            
            # Split by ; to handle multiple statements
            statements = schema_sql.split(';')
            for statement in statements:
                if statement.strip():
                    connection.execute(text(statement))
            connection.commit()
        print("Schema initialized successfully.")
    except Exception as e:
        print(f"Error initializing schema: {e}")
        raise

def load_data(engine):
    """Loads all CSV files from the data directory into the database."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        print(f"Data folder does not exist at {data_dir}")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in data folder")
        return

    print(f"Found {len(csv_files)} CSV files to load.")

    # Column mapping from CSV to DB
    column_mapping = {
        'x': 'vibration_x',
        'y': 'vibration_y',
        'Z': 'vibration_z',
        'I1': 'current_phase_1',
        'I2': 'current_phase_2',
        'I3': 'current_phase_3',
        'V1': 'voltage_phase_1',
        'V2': 'voltage_phase_2',
        'V3': 'voltage_phase_3'
    }

    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"Processing {file_name}...")

        # 1. Create Recording Entry (Metadata)
        # We use a raw SQL execution to insert and get the ID back
        try:
            with engine.connect() as conn:
                # Check if recording exists
                result = conn.execute(text("SELECT id FROM recordings WHERE file_name = :fn"), {"fn": file_name})
                row = result.fetchone()
                
                if row:
                    recording_id = row[0]
                    print(f"  Recording already exists for {file_name} (ID: {recording_id}). Skipping metadata insert.")
                else:
                    # Insert new recording
                    result = conn.execute(
                        text("INSERT INTO recordings (file_name) VALUES (:fn) RETURNING id"),
                        {"fn": file_name}
                    )
                    recording_id = result.fetchone()[0]
                    conn.commit()
                    print(f"  Created new recording for {file_name} (ID: {recording_id})")
        except Exception as e:
            print(f"  Error creating recording for {file_name}: {e}")
            continue
        
        # 2. Load Experiments (Sensor Data Rows)
        chunksize = 100000
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Rename columns
            chunk.rename(columns=column_mapping, inplace=True)
            
            # Add recording_id column (links row to the file metadata)
            chunk['recording_id'] = recording_id
            
            # Load to DB (Table is now 'experiments')
            chunk.to_sql('experiments', engine, if_exists='append', index=False)
            print(f"  Loaded chunk of {len(chunk)} experiments (rows) for Recording {recording_id}")

    print("All data loaded successfully.")

def main():
    create_database_if_not_exists()
    print(f"Connecting to database: {DATABASE_URL}")
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            pass
    except Exception as e:
        print(f"Failed to connect to database. Please check your credentials and ensure the database '{DB_NAME}' exists.")
        print(f"Error: {e}")
        return

    init_db(engine)
    load_data(engine)
    
    engine.dispose()

if __name__ == "__main__":
    main()