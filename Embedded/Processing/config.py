PORT = "COM3"
BAUD_RATE = 921600

# Sampling settings
SAMPLE_RATE_HZ = 1000
PHASE_COUNT = 3
BLOCK_SAMPLES = 256

# ESP32 ADC data format
NUMPY_DTYPE = "uint16"
BYTES_PER_VALUE = 2
BLOCK_BYTES = BLOCK_SAMPLES * PHASE_COUNT * BYTES_PER_VALUE

# Sensor / signal settings
ADC_ZERO_OFFSET = 2048.0  # Midpoint for 12-bit ADC (0-4095)

# Temporal fixed DC bus voltage
VDC_CONSTANT = 24.0  # Volts

# Filtering settings
LOW_PASS_CUTOFF_HZ = 20.0  # Low-pass filter cutoff frequency

# Output files
PROCESSED_DATA_FILE = "output/processed/processed_data.csv"
RAW_DATA_FILE = "output/raw/raw_data.bin"