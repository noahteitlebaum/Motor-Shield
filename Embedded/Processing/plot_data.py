import csv
import matplotlib.pyplot as plt
from config import RAW_DATA_FILE, PROCESSED_DATA_FILE, SAMPLE_RATE_HZ

PLOT_SECONDS = 2
DOWNSAMPLE = 4   # smaller = smoother plot


def load_csv(file_path):
    data = {}

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(value))

    return data


def main():

    raw = load_csv(RAW_DATA_FILE)
    processed = load_csv(PROCESSED_DATA_FILE)

    samples = int(PLOT_SECONDS * SAMPLE_RATE_HZ)

    # slice and lightly downsample
    t = raw["time"][:samples:DOWNSAMPLE]

    Ia_raw = raw["Ia"][:samples:DOWNSAMPLE]
    Ib_raw = raw["Ib"][:samples:DOWNSAMPLE]
    Ic_raw = raw["Ic"][:samples:DOWNSAMPLE]

    Ia_proc = processed["Ia"][:samples:DOWNSAMPLE]
    Ib_proc = processed["Ib"][:samples:DOWNSAMPLE]
    Ic_proc = processed["Ic"][:samples:DOWNSAMPLE]

    # -------- RAW SIGNAL --------
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes1[0].plot(t, Ia_raw)
    axes1[0].set_title("Raw Phase A Current")
    axes1[0].grid(True)

    axes1[1].plot(t, Ib_raw)
    axes1[1].set_title("Raw Phase B Current")
    axes1[1].grid(True)

    axes1[2].plot(t, Ic_raw)
    axes1[2].set_title("Raw Phase C Current")
    axes1[2].set_xlabel("Time (s)")
    axes1[2].grid(True)

    plt.tight_layout()

    # -------- FILTERED SIGNAL --------
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes2[0].plot(t, Ia_proc)
    axes2[0].set_title("Filtered Phase A Current")
    axes2[0].grid(True)

    axes2[1].plot(t, Ib_proc)
    axes2[1].set_title("Filtered Phase B Current")
    axes2[1].grid(True)

    axes2[2].plot(t, Ic_proc)
    axes2[2].set_title("Filtered Phase C Current")
    axes2[2].set_xlabel("Time (s)")
    axes2[2].grid(True)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()