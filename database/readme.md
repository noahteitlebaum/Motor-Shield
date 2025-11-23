# Motor Shield Database

This directory contains the database schema and data loading scripts for the Motor Shield project. The database is designed to store synchronized multi-sensor data from induction motors for fault diagnosis and predictive maintenance.

## Prerequisites

1.  **PostgreSQL**: Ensure you have PostgreSQL installed and running.
    *   [Download PostgreSQL](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)
    *   Default configuration assumes user `postgres` and password `password` (or `12345` as per some local setups). You can configure these via environment variables.
2.  **Python 3.8+**: Ensure Python is installed.

## Setup

1.  **Navigate to the project root**:
    ```bash
    cd /path/to/Motor-Shield
    ```

2.  **Install Python Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r database/requirements.txt
    ```

## Database Initialization & Data Loading

The `database/main.py` script handles everything:
1.  Creates the `motor_shield` database if it doesn't exist.
2.  Applies the schema from `database/schema.sql` (resetting tables if they exist).
3.  Loads all CSV files from `database/data/` into the database.

**Run the script:**
```bash
python database/main.py
```

*Note: This process may take a few minutes as it loads approximately 10 million rows of sensor data.*

## Schema Overview

The database uses a normalized schema optimized for machine learning:

*   **`recordings`**: Metadata for each data file (e.g., motor type, sampling rate).
*   **`experiments`**: The actual high-frequency sensor data rows (Vibration X/Y/Z, Current, Voltage).
*   **`fault_classes`**: Labels for different fault types (Normal, Misalignment, etc.).
*   **`experiment_features`**: A flexible JSONB table for storing derived features for ML.
*   **`model_predictions`** & **`model_training_data`**: Tables to track AI model performance and training sets.

See [ER_DIAGRAM.md](ER_DIAGRAM.md) for a visual representation of the relationships.

## Configuration

You can override default database credentials using environment variables:

```bash
export DB_USER=your_user
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=motor_shield
```

## Troubleshooting

*   **"Database does not exist"**: The script tries to create it automatically. If this fails (e.g., due to permissions), log in to `psql` and run `CREATE DATABASE motor_shield;` manually.
*   **"Password authentication failed"**: Check your `DB_PASSWORD` in `database/main.py` or set the environment variable.
*   **"Data folder does not exist"**: Ensure you have downloaded the dataset into `database/data/`.


# Dataset Description
```txt
Induction motors play a crucial role in industrial applications, but their operation is often compromised by various mechanical and electrical faults. This paper presents a new dataset for comprehensive fault diagnosis of three-phase induction motors, using synchronized multi-sensor data collection. The dataset includes real-time measurements of vibration, voltage, and current collected from a 0.2 kW squirrel cage induction motor. Fault scenarios such as phase removal and mechanical misalignments were simulated to capture a wide range of motor behaviors. Data were collected using high-resolution sensors, with the vibration ,voltage and current sampled at 50kHz. The dataset is organized into tem distinct CSV files, covering different operational scenarios, providing a comprehensive resource for researchers aiming to develop or test fault detection algorithms. The dataset was used to train a RandomForest classifier for fault detection, achieving an accuracy of 99.82%. This demonstrates the effectiveness of the dataset for developing machine learning models aimed at real-time fault diagnosis and predictive maintenance. Unlike existing datasets, this collection provides synchronized data across multiple sensor types, enabling cross-analysis of electrical and mechanical faults. The dataset is publicly available, offering a valuable tool for advancing research in motor fault diagnosis and predictive maintenance.
```
