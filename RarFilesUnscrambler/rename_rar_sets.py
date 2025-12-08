import argparse
import os
from typing import List, Tuple

from group_rars_unrar import scan_directory, group_rars_with_unrar


def plan_renames_for_group(
    set_index: int,
    inner_name: str,
    files: List[Tuple[str, int, int]],
) -> List[Tuple[str, str]]:
    """Return list of (old_path, new_path) for one group.

    Files are tuples of (path, size, vol). We rename them to
    something like:
      set001_TargetOutputFile1.bin_part001.rar
      set001_TargetOutputFile1.bin_part002.rar
    in the same directory.
    """
    renames = []
    # Derive a safe base from inner_name
    safe_inner = inner_name or "unknown"
    # Replace path separators just in case
    safe_inner = safe_inner.replace("\\", "_").replace("/", "_")

    for idx, (path, _size, vol) in enumerate(files, start=1):
        dirname = os.path.dirname(path)
        # Prefer volume number if present, else sequence index
        part_no = vol if vol > 0 else idx
        new_filename = f"set{set_index:03d}_{safe_inner}_part{part_no:03d}.rar"
        new_path = os.path.join(dirname, new_filename)
        renames.append((path, new_path))
    return renames


def apply_renames(renames: List[Tuple[str, str]], dry_run: bool = False) -> None:
    # Avoid collisions: check all targets first
    targets = {}
    for old, new in renames:
        if old == new:
            continue
        if new in targets.values() or (os.path.exists(new) and new not in targets):
            raise RuntimeError(f"Target already exists, aborting rename: {new}")
        targets[old] = new

    for old, new in renames:
        if old == new:
            continue
        print(f"REN: {os.path.basename(old)} -> {os.path.basename(new)}")
        if not dry_run:
            os.rename(old, new)


def write_extract_helper(root: str, set_index: int, first_new_path: str) -> None:
    cmd_path = os.path.join(root, f"rar_set_{set_index:03d}_extract_command.txt")
    base = os.path.basename(first_new_path)
    content = []
    content.append("# Suggested extraction commands for this set")
    content.append("# Adjust paths / tool (7z or unrar) as needed.")
    content.append("")
    content.append("# Using 7-Zip (CLI):")
    content.append(f"7z x \"{base}\"")
    content.append("")
    content.append("# Using WinRAR/UnRAR (CLI):")
    content.append(f"unrar x \"{base}\"")
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"  -> Wrote helper: {os.path.basename(cmd_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Group scrambled multi-part RAR files into sets using unrar metadata "
            "and rename each set to sequential part names."
        )
    )
    parser.add_argument(
        "directory",
        help="Directory containing the scrambled .rar files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing any files.",
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

    all_renames: List[Tuple[str, str]] = []

    for set_index, (key, files) in enumerate(groups, start=1):
        inner_name, total_size = key
        print("=" * 60)
        print(
            f"Set #{set_index}: inner_name={inner_name!r}, total_unpacked_size={total_size}, parts={len(files)}"
        )
        for path, size, vol in files:
            print(f"  vol={vol:3d}  {size:>12} bytes  {os.path.basename(path)}")

        renames = plan_renames_for_group(set_index, inner_name, files)
        for old, new in renames:
            print(f"  PLAN: {os.path.basename(old)} -> {os.path.basename(new)}")
        all_renames.extend(renames)

        if renames:
            # Write helper based on the first new filename (after rename)
            write_extract_helper(root, set_index, renames[0][1])

    if not all_renames:
        print("Nothing to rename.")
        return

    if args.dry_run:
        print("Dry run complete. No files were renamed.")
        return

    print("Applying renames...")
    apply_renames(all_renames, dry_run=False)
    print("Done.")


if __name__ == "__main__":
    main()
