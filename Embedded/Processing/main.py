from config import (
    PORT,
    BAUD_RATE,
    SAMPLE_RATE_HZ,
    PHASE_COUNT,
    BLOCK_SAMPLES,
    NUMPY_DTYPE,
    BLOCK_BYTES,
    ADC_ZERO_OFFSET,
    VDC_CONSTANT,
    LOW_PASS_CUTOFF_HZ,
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
)

from capture import open_serial, read_block, parse_block
from filters import convert_block_to_raw_currents, filter_three_phase_block
from io_utils import write_csv_header, append_rows


def main() -> None:
    header = ["time", "Vdc", "Ia", "Ib", "Ic"]

    write_csv_header(RAW_DATA_FILE, header)
    write_csv_header(PROCESSED_DATA_FILE, header)

    ser = open_serial(PORT, BAUD_RATE)
    sample_index = 0

    print("Starting capture. Press Ctrl+C to stop.")

    try:
        while True:
            raw = read_block(ser, BLOCK_BYTES)

            if len(raw) != BLOCK_BYTES:
                continue

            try:
                block = parse_block(raw, NUMPY_DTYPE, BLOCK_SAMPLES, PHASE_COUNT)
            except ValueError as e:
                print(e)
                continue

            raw_current_block = convert_block_to_raw_currents(
                block,
                ADC_ZERO_OFFSET
            )

            filtered_block = filter_three_phase_block(
                block,
                SAMPLE_RATE_HZ,
                LOW_PASS_CUTOFF_HZ,
                ADC_ZERO_OFFSET
            )

            raw_rows = []
            processed_rows = []

            for i in range(BLOCK_SAMPLES):
                time_s = sample_index / SAMPLE_RATE_HZ

                ia_raw, ib_raw, ic_raw = raw_current_block[i]
                ia_filt, ib_filt, ic_filt = filtered_block[i]

                raw_rows.append([
                    time_s,
                    VDC_CONSTANT,
                    float(ia_raw),
                    float(ib_raw),
                    float(ic_raw),
                ])

                processed_rows.append([
                    time_s,
                    VDC_CONSTANT,
                    float(ia_filt),
                    float(ib_filt),
                    float(ic_filt),
                ])

                sample_index += 1

            append_rows(RAW_DATA_FILE, raw_rows)
            append_rows(PROCESSED_DATA_FILE, processed_rows)

            print(f"Saved block ending at sample {sample_index}")

    except KeyboardInterrupt:
        print("Stopping capture.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()