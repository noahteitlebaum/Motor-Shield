import time
import serial
import numpy as np

# Open serial connection to the esp32
def open_serial(port: str, baud_rate: int, timeout: float = 1.0) -> serial.Serial:
    ser = serial.Serial(port, baud_rate, timeout=timeout)
    time.sleep(2)
    ser.reset_input_buffer()
    return ser

# Read a block of raw bytes from the serial port
def read_block(ser: serial.Serial, block_bytes: int) -> bytes:
    return ser.read(block_bytes)

# Convert raw bytes to a 2D numpy array of shape (block_samples, channel_count)
def parse_block(raw: bytes, dtype: str, block_samples: int, channel_count: int) -> np.ndarray:
    data = np.frombuffer(raw, dtype=dtype)

    expected_values = block_samples * channel_count
    if data.size != expected_values:
        raise ValueError(
            f"Expected {expected_values} values, got {data.size}."
        )
    # Reshape the 1D array into a 2-D matrix so that each row corresponds to a sample of three motor phases
    return data.reshape(block_samples, channel_count)