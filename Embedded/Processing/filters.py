import numpy as np

# Simple filters for processing the raw ADC data from the esp32 before estimating the rotor position.
def adc_to_centered(block: np.ndarray, offset: float) -> np.ndarray:
    return block.astype(np.float64) - offset

# Remove the DC component from the signal by subtracting the mean
def remove_dc(signal: np.ndarray) -> np.ndarray:
    return signal - np.mean(signal)

# Convert the raw ADC values to centered currents by applying the offset and removing the DC component
def convert_block_to_raw_currents(block: np.ndarray, offset: float) -> np.ndarray:
    centered = adc_to_centered(block, offset)

    ia = remove_dc(centered[:, 0])
    ib = remove_dc(centered[:, 1])
    ic = remove_dc(centered[:, 2])

    return np.column_stack((ia, ib, ic))

# Apply a low-pass filter to the signal using FFT. This is a simple way to remove high-frequency noise from the raw current signals.
def fft_lowpass(signal: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
    n = len(signal)

    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    fft_vals[freqs > cutoff] = 0

    return np.fft.irfft(fft_vals, n=n)

# Apply the low-pass filter to each of the three phases in the block of raw ADC data, returning a new block of filtered currents.
def filter_three_phase_block(block: np.ndarray, sample_rate: int, cutoff: float, offset: float) -> np.ndarray:
    raw_currents = convert_block_to_raw_currents(block, offset)

    ia_f = fft_lowpass(raw_currents[:, 0], sample_rate, cutoff)
    ib_f = fft_lowpass(raw_currents[:, 1], sample_rate, cutoff)
    ic_f = fft_lowpass(raw_currents[:, 2], sample_rate, cutoff)

    return np.column_stack((ia_f, ib_f, ic_f))