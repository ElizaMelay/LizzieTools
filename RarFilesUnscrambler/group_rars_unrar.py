import argparse
import os
import re
import subprocess
from collections import defaultdict

UNRAR_CMD = "unrar"  # assumes unrar.exe is in PATH


def call_unrar_lt_bulk(pattern):
    """Run a single `unrar lt` over a glob pattern and return stdout."""
    try:
        result = subprocess.run(
            [UNRAR_CMD, "lt", "-c-", "-p-", pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return result.stdout
    except FileNotFoundError:
        raise SystemExit("unrar.exe not found on PATH. Install WinRAR/UnRAR or add unrar to PATH.")


def parse_unrar_lt_bulk(output):
    """Parse bulk `unrar lt` output into per-archive records.

    Returns list of dicts with keys: path, inner_name, total_size, volume.
    """
    records = []

    arch_re = re.compile(r"^Archive:\s+(.+)$", re.IGNORECASE)
    name_re = re.compile(r"^\s*Name:\s+(.+)$", re.IGNORECASE)
    details_re = re.compile(r"^Details:\s+RAR\s+\d+,\s+volume\s+(\d+)", re.IGNORECASE)
    size_re = re.compile(r"^\s*Size:\s+(\d+)", re.IGNORECASE)

    cur = None

    for line in output.splitlines():
        line = line.rstrip("\r\n")

        m = arch_re.match(line)
        if m:
            # flush previous
            if cur is not None:
                records.append(cur)
            cur = {
                "path": m.group(1).strip(),
                "inner_name": None,
                "total_size": None,
                "volume": None,
            }
            continue

        if cur is None:
            continue

        m = name_re.match(line)
        if m and cur["inner_name"] is None:
            cur["inner_name"] = m.group(1).strip()
            continue

        m = details_re.match(line)
        if m and cur["volume"] is None:
            cur["volume"] = int(m.group(1))
            continue

        m = size_re.match(line)
        if m and cur["total_size"] is None:
            try:
                cur["total_size"] = int(m.group(1))
            except ValueError:
                pass

    if cur is not None:
        records.append(cur)

    return records


def is_rar(path):
    try:
        with open(path, "rb") as f:
            return f.read(7).startswith(b"Rar!\x1a\x07")
    except OSError:
        return False


def scan_directory(root):
    files = []
    for entry in os.scandir(root):
        if entry.is_file():
            path = entry.path
            if is_rar(path):
                files.append(path)
    return files


def group_rars_with_unrar(paths):
    """Group files by inner file name + uncompressed size using bulk `unrar lt`.

    This runs a single `unrar lt` over the directory's `*.rar` and
    parses its output for much faster performance.
    """
    if not paths:
        return []

    # Assume all paths share the same parent directory; build a glob there.
    parent = os.path.dirname(paths[0]) or "."
    pattern = os.path.join(parent, "*.rar")
    print(f"Running bulk unrar lt on pattern: {pattern}")
    bulk_out = call_unrar_lt_bulk(pattern)
    records = parse_unrar_lt_bulk(bulk_out)

    # Map from absolute path to record
    rec_by_path = {os.path.abspath(r["path"]): r for r in records}

    groups = defaultdict(list)
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        abs_path = os.path.abspath(path)
        print(f"[{idx}/{total}] Processing {os.path.basename(path)}")
        rec = rec_by_path.get(abs_path)
        size = os.path.getsize(path)

        if not rec:
            key = ("unknown", None)
            vol = 0
        else:
            key = (rec["inner_name"], rec["total_size"])
            vol = rec["volume"] if rec["volume"] is not None else 0

        groups[key].append((path, size, vol))

    grouped = []
    for key, files in groups.items():
        files_sorted = sorted(files, key=lambda x: (x[2], x[0]))
        grouped.append((key, files_sorted))

    grouped.sort(key=lambda g: -len(g[1]))
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Group scrambled multi-part RAR files into sets using unrar metadata."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing scrambled files (default: current)",
    )
    parser.add_argument(
        "--write-commands",
        action="store_true",
        help="Write a .txt file per group with suggested extraction command.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    print(f"Scanning directory: {root}")

    rar_files = scan_directory(root)
    if not rar_files:
        print("No RAR files detected.")
        return

    print(f"Detected {len(rar_files)} potential RAR parts.")
    groups = group_rars_with_unrar(rar_files)

    for idx, (key, files) in enumerate(groups, start=1):
        inner_name, total_size = key
        print("=" * 60)
        print(
            f"Set #{idx}: inner_name={inner_name!r}, total_unpacked_size={total_size}, found_parts={len(files)}"
        )
        for path, size, vol in files:
            print(f"  vol={vol:3d}  {size:>12} bytes  {os.path.basename(path)}")

        if args.write_commands:
            primary = files[0][0]
            cmd_path = os.path.join(
                root, f"rar_set_{idx:03d}_extract_command.txt"
            )
            content = []
            content.append("# Suggested extraction commands for this set")
            content.append("# Adjust paths / tool (7z or unrar) as needed.")
            content.append("")
            content.append("# Using 7-Zip (CLI):")
            content.append(f'7z x "{os.path.basename(primary)}"')
            content.append("")
            content.append("# Using WinRAR/UnRAR (CLI):")
            content.append(f'unrar x "{os.path.basename(primary)}"')
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            print(f"  -> Wrote helper: {os.path.basename(cmd_path)}")

    print("=" * 60)
    print(f"Identified {len(groups)} group(s).")


if __name__ == "__main__":
    main()