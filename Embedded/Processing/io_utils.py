import os
import csv

# Utility functions for file handling and CSV writing
def ensure_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

# Write the header row to a CSV file, creating the file if it doesn't exist
def write_csv_header(file_path: str, header: list[str]) -> None:
    ensure_dir(file_path)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Append rows of data to an existing CSV file
def append_rows(file_path: str, rows: list[list]) -> None:
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)