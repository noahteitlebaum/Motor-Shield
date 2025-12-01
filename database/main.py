"""
This is the main file for the database, this does the following:
1) Create the database if it doesn't exist.
2) Set up the connection to the database.
3) Parse the data files and create the database schema.
4) Load all CSV data into the database.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

# Database connection configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345")  # Default password, change as needed
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "motor_shield")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def create_database_if_not_exists():
    """Creates the database if it doesn't exist."""
    default_db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"

    try:
        engine = create_engine(default_db_url, isolation_level="AUTOCOMMIT")
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": DB_NAME},
            )
            if not result.fetchone():
                print(f"Database '{DB_NAME}' does not exist. Creating...")
                conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
                print(f"Database '{DB_NAME}' created successfully.")
            else:
                print(f"Database '{DB_NAME}' already exists.")
        engine.dispose()
    except Exception as e:
        print(f"Warning: Could not check/create database '{DB_NAME}'. Error: {e}")
        print("Please ensure the database exists manually if connection fails.")


def init_db(engine):
    """Reads the schema.sql file and creates the tables."""
    print("Initializing database schema...")
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    try:
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        # Split schema into individual statements; filter out empty ones
        statements = [
            stmt.strip()
            for stmt in schema_sql.split(";")
            if stmt.strip()
        ]

        # Use a single transaction for all DDL
        with engine.begin() as conn:
            # Drop old tables to ensure a clean schema
            conn.execute(
                text(
                    """
                    DROP TABLE IF EXISTS
                        experiment_features,
                        model_training_data,
                        model_predictions,
                        experiments,
                        recordings,
                        sensor_measurements,
                        fault_classes
                    CASCADE
                    """
                )
            )

            for stmt in statements:
                conn.execute(text(stmt))

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
        "x": "vibration_x",
        "y": "vibration_y",
        "Z": "vibration_z",
        "I1": "current_phase_1",
        "I2": "current_phase_2",
        "I3": "current_phase_3",
        "V1": "voltage_phase_1",
        "V2": "voltage_phase_2",
        "V3": "voltage_phase_3",
    }

    # Use a single transaction for the entire data load to reduce overhead
    with engine.begin() as conn:
        # 1) Ensure fault classes exist
        fault_class_ids = {}
        try:
            classes_to_define = [
                ("Healthy", "Normal operating condition"),
                ("Faulty", "Motor with faults"),
            ]

            for fname, fdesc in classes_to_define:
                res = conn.execute(
                    text("SELECT id FROM fault_classes WHERE fault_name = :name"),
                    {"name": fname},
                )
                row = res.fetchone()
                if row:
                    fault_class_ids[fname] = row[0]
                else:
                    res = conn.execute(
                        text(
                            """
                            INSERT INTO fault_classes (fault_name, description)
                            VALUES (:name, :desc)
                            RETURNING id
                            """
                        ),
                        {"name": fname, "desc": fdesc},
                    )
                    fault_class_ids[fname] = res.fetchone()[0]

            print(f"Fault classes ensured: {fault_class_ids}")
        except Exception as e:
            print(f"Error setting up fault classes: {e}")
            return

        # 2) Process each CSV file
        chunksize = 100_000

        for file_name in sorted(csv_files):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing {file_name}...")

            # Determine metadata based on filename
            current_fault_class_id = None
            current_load_condition = None

            # Case-insensitive check for filename patterns
            fname_upper = file_name.upper()
            if "FILE 1" in fname_upper:
                current_load_condition = "normal load"
                current_fault_class_id = fault_class_ids.get("Healthy")
            elif "FILE 6" in fname_upper:
                current_load_condition = "normal load"
                current_fault_class_id = fault_class_ids.get("Faulty")

            # 2a) Create or fetch Recording entry (metadata)
            try:
                result = conn.execute(
                    text("SELECT id FROM recordings WHERE file_name = :fn"),
                    {"fn": file_name},
                )
                row = result.fetchone()

                if row:
                    recording_id = row[0]
                    print(
                        f"  Recording already exists for {file_name} "
                        f"(ID: {recording_id}). Skipping metadata insert."
                    )
                else:
                    result = conn.execute(
                        text(
                            """
                            INSERT INTO recordings (file_name, fault_class_id, load_condition)
                            VALUES (:fn, :fc_id, :load)
                            RETURNING id
                            """
                        ),
                        {
                            "fn": file_name,
                            "fc_id": current_fault_class_id,
                            "load": current_load_condition,
                        },
                    )
                    recording_id = result.fetchone()[0]
                    print(
                        f"  Created new recording for {file_name} "
                        f"(ID: {recording_id}, FaultID: {current_fault_class_id}, "
                        f"Load: {current_load_condition})"
                    )
            except Exception as e:
                print(f"  Error creating recording for {file_name}: {e}")
                continue

            # 2b) Load experiments (sensor data rows) in chunks
            try:
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    # Rename columns to match DB schema
                    chunk = chunk.rename(columns=column_mapping)

                    # Add recording_id column
                    chunk["recording_id"] = recording_id

                    # Write to DB using batched inserts for speed
                    chunk.to_sql(
                        "experiments",
                        conn,
                        if_exists="append",
                        index=False,
                        method="multi",       # batched INSERTs
                    )
                    print(
                        f"  Loaded chunk of {len(chunk)} experiments (rows) "
                        f"for Recording {recording_id}"
                    )
            except Exception as e:
                print(f"  Error loading data from {file_name}: {e}")
                continue

    print("All data loaded successfully.")


def main():
    create_database_if_not_exists()
    print(f"Connecting to database: {DATABASE_URL}")
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        print(
            f"Failed to connect to database. Please check your credentials "
            f"and ensure the database '{DB_NAME}' exists."
        )
        print(f"Error: {e}")
        return

    init_db(engine)
    load_data(engine)

    engine.dispose()


if __name__ == "__main__":
    main()
