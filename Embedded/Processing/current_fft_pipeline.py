import time
import numpy as np
import serial

# Optional plots (comment out if you don't want plots)
import matplotlib.pyplot as plt

# ===================== SETTINGS =====================
PORT = "COM3"           # <-- change this
BAUD = 921600           # must match ESP32
FS_PER_CH = 5000.0      # sampling rate per phase (Hz)
BLOCK_N = 4096          # samples per phase per processing block
CUTOFF_HZ = 500.0       # low-pass cutoff (Hz)
PLOT = False            # set True if you want quick plots
# ====================================================


def parse_line(line: str):
    """
    ESP32 CSV format expected:
      t_us,ch,raw,volts,amps
    Returns (ch:int, amps:float) or None if invalid.
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("t_us"):
        return None

    parts = line.split(",")
    if len(parts) < 5:
        return None

    try:
        ch = int(parts[1])
        amps = float(parts[4])
        if ch not in (0, 1, 2):
            return None
        return ch, amps
    except ValueError:
        return None


def fft_lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """
    Frequency-domain low-pass:
      FFT -> zero bins above cutoff -> IFFT
    """
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=n)


def filter_block(iu: np.ndarray, iv: np.ndarray, iw: np.ndarray):
    """
    Remove DC, low-pass, add DC back.
    """
    mu, mv, mw = iu.mean(), iv.mean(), iw.mean()
    iu_f = fft_lowpass(iu - mu, FS_PER_CH, CUTOFF_HZ) + mu
    iv_f = fft_lowpass(iv - mv, FS_PER_CH, CUTOFF_HZ) + mv
    iw_f = fft_lowpass(iw - mw, FS_PER_CH, CUTOFF_HZ) + mw
    return iu_f, iv_f, iw_f


def main():
    if CUTOFF_HZ >= FS_PER_CH / 2:
        raise ValueError(f"CUTOFF_HZ must be < Nyquist (FS/2={FS_PER_CH/2}).")

    buf = {0: [], 1: [], 2: []}

    print(f"Opening {PORT} @ {BAUD} ...")
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()

        print("Reading ESP32 stream... (Ctrl+C to stop)")
        while True:
            line = ser.readline().decode(errors="ignore")
            parsed = parse_line(line)
            if parsed is None:
                continue

            ch, amps = parsed
            buf[ch].append(amps)

            # When we have a full block for each phase, filter it
            if len(buf[0]) >= BLOCK_N and len(buf[1]) >= BLOCK_N and len(buf[2]) >= BLOCK_N:
                iu = np.array(buf[0][:BLOCK_N], dtype=np.float64)
                iv = np.array(buf[1][:BLOCK_N], dtype=np.float64)
                iw = np.array(buf[2][:BLOCK_N], dtype=np.float64)

                # Drop used samples
                buf[0] = buf[0][BLOCK_N:]
                buf[1] = buf[1][BLOCK_N:]
                buf[2] = buf[2][BLOCK_N:]

                iu_f, iv_f, iw_f = filter_block(iu, iv, iw)

                # Example: print first few filtered values (you can save to file instead)
                print(f"Filtered block ready: iu_f[0]={iu_f[0]:.4f}, iv_f[0]={iv_f[0]:.4f}, iw_f[0]={iw_f[0]:.4f}")

                if PLOT:
                    t = np.arange(BLOCK_N) / FS_PER_CH
                    plt.figure()
                    plt.plot(t, iu, label="Iu raw")
                    plt.plot(t, iu_f, label="Iu filtered")
                    plt.legend()
                    plt.grid(True)
                    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
