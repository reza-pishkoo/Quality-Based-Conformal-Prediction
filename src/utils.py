# src/utils.py
import os, csv, datetime as dt

def append_rows_csv(csv_path, rows, header):
    """
    Append list of dict rows to csv_path.
    Creates file with header if missing.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")
