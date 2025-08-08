# src/utils.py
import csv, os, datetime

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def append_rows_csv(path, rows, header):
    """
    Append rows to CSV in a parallel-safe way (Linux clusters) using fcntl file lock.
    Creates parent dirs and header if file doesn't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        import fcntl
        with open(path, "a+", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        # fallback (not fully safe), but Discovery is Linux so fcntl exists
        with open(path, "a+", newline="") as f:
            write_header = f.tell() == 0
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
