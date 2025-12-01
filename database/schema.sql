-- Improved PostgreSQL Schema for Motor Shield Data
-- Updated terminology: Each row of sensor data is an "Experiment" (Sample).
-- Files are "Recordings" that group these experiments.

-- 1. Fault Classes (Labels)
CREATE TABLE IF NOT EXISTS fault_classes (
    id SERIAL PRIMARY KEY,
    fault_name VARCHAR(50) NOT NULL UNIQUE, -- e.g., 'Normal', 'Phase Removal', 'Misalignment'
    description TEXT
);

-- 2. Recordings (Metadata for each file/session)
-- Formerly 'experiments'
CREATE TABLE IF NOT EXISTS recordings (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL UNIQUE, -- e.g., 'FILE 1.csv'
    fault_class_id INTEGER, -- The label applies to the whole recording usually
    collection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sampling_rate_hz INTEGER DEFAULT 50000,
    motor_type VARCHAR(100) DEFAULT '0.2 kW squirrel cage induction motor',
    load_condition VARCHAR(50),
    FOREIGN KEY (fault_class_id) REFERENCES fault_classes(id)
);

-- 3. Experiments (The actual sensor data rows)
-- Formerly 'sensor_measurements'. User specified "each row is its own experiment".
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    recording_id INTEGER NOT NULL, -- Links back to the source file/metadata
    
    -- Vibration Data (3-Axis)
    vibration_x DOUBLE PRECISION,
    vibration_y DOUBLE PRECISION,
    vibration_z DOUBLE PRECISION,
    
    -- Current Data (3-Phase)
    current_phase_1 DOUBLE PRECISION,
    current_phase_2 DOUBLE PRECISION,
    current_phase_3 DOUBLE PRECISION,
    
    -- Voltage Data (3-Phase)
    voltage_phase_1 DOUBLE PRECISION,
    voltage_phase_2 DOUBLE PRECISION,
    voltage_phase_3 DOUBLE PRECISION,
    
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- 4. Model Predictions (Per-experiment/row predictions)
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL, -- Links to the specific row
    model_name VARCHAR(50) NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- 5. Model Training Data (Tracking which rows were used for training)
CREATE TABLE IF NOT EXISTS model_training_data (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- 6. Feature Engineering (Flexible storage for derived features)
-- Designed to be expandable using JSONB for dynamic feature sets
CREATE TABLE IF NOT EXISTS experiment_features (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    
    -- Identification of the feature extraction run/version
    feature_version VARCHAR(50) DEFAULT 'v1',
    
    -- Flexible JSONB column to store key-value pairs of features
    -- e.g., {"rms_vibration": 0.45, "peak_current": 1.2, "fft_energy": 500.2}
    feature_data JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiments_recording_id ON experiments(recording_id);
CREATE INDEX IF NOT EXISTS idx_experiments_id ON experiments(id);
CREATE INDEX IF NOT EXISTS idx_features_experiment_id ON experiment_features(experiment_id);
-- GIN index allows efficient querying of keys/values within the JSONB data
CREATE INDEX IF NOT EXISTS idx_features_data ON experiment_features USING gin (feature_data);
