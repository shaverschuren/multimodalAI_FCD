import argparse
from pathlib import Path
import re
import sys

"""
Create an empty LABEL.txt file in each immediate subdirectory named RESP#### (digits).
Usage:
    python label_fs_ds.py /path/to/base_dir
If no path is given, uses the current working directory.
"""


PATTERN = re.compile(r"^RESP\d+$")
FILENAME = "FASTSURFER.txt"

def main():
    p = argparse.ArgumentParser(description="Add an empty FASTSURFER.txt to RESPxxxx subdirs")
    p.add_argument("base", nargs="?", default=".", help="Base directory (default: current dir)")
    args = p.parse_args()

    base = Path(args.base)
    if not base.exists() or not base.is_dir():
        print(f"Base path does not exist or is not a directory: {base}", file=sys.stderr)
        sys.exit(2)

    created = []
    skipped = []
    for entry in base.iterdir():
        if entry.is_dir() and PATTERN.fullmatch(entry.name):
            label_path = entry / FILENAME
            if label_path.exists():
                skipped.append(str(label_path))
            else:
                try:
                    label_path.write_text("")  # creates an empty file
                    created.append(str(label_path))
                except Exception as e:
                    print(f"Failed to create {label_path}: {e}", file=sys.stderr)

    if created:
        print("Created:")
        for p in created:
            print("  " + p)
    if skipped:
        print("Already exists (skipped):")
        for p in skipped:
            print("  " + p)


if __name__ == "__main__":
    main()