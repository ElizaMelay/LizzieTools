# Automatically upscale textures dumped from pcsx2 using realesrgan

# General algorithm for upscaling images using Real-ESRGAN
# Run Real-ESRGAN on the dumped textures folder into an intermediate folder
# Patch up texture mips by copying the highest mip level to all lower mip levels
# Copy the upscaled textures to a final output folder

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}
MIP_REGEX = re.compile(r"^(?P<prefix>.*)-mip(?P<mip>\d+)(?P<suffix>.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upscale PCSX2 dumped textures using Real-ESRGAN, patch mip levels, and copy to output."
        )
    )
    parser.add_argument(
        "-r",
        "--realesrgan",
        required=True,
        help=(
            "Path to Real-ESRGAN executable or script (e.g., realesrgan-ncnn-vulkan.exe or inference_realesrgan.py).\n"
            "Can also be a directory containing the executable."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=False, help="Path to dumped textures input folder"
    )
    parser.add_argument(
        "-m", "--intermediate", required=False, help="Path to intermediate folder"
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Path to final output folder"
    )
    parser.add_argument(
        "-g", "--game", required=False, help="Path to PCSX2 game texture folder (sets input/intermediate/output automatically)"
    )
    parser.add_argument(
        "--realesrgan-args",
        default="",
        help=(
            "Extra arguments passed through to the Real-ESRGAN command. Provide as a single string."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (can be specified multiple times, e.g. -vvv)",
    )
    return parser.parse_args()
def vprint(msg: str, level: int, verbosity: int):
    if verbosity >= level:
        print(msg)


def discover_realesrgan_cmd(path: Path) -> List[str]:
    """
    Return a command list to execute Real-ESRGAN with -i and -o.
    Supports:
      - realesrgan-ncnn-vulkan(.exe)
      - Python script (inference_realesrgan.py)
      - Generic executable assumed to accept -i/-o
    If a directory is provided, attempts to locate common executables inside it.
    """
    candidates: List[Path] = []

    if path.is_dir():
        # Common names
        names = [
            "realesrgan-ncnn-vulkan.exe",
            "realesrgan-ncnn-vulkan",
            "inference_realesrgan.py",
        ]
        for n in names:
            cand = path / n
            if cand.exists():
                candidates.append(cand)
    elif path.exists():
        candidates.append(path)

    if not candidates:
        # Use as-is (maybe it's on PATH)
        return [str(path)]

    tool = candidates[0]
    if tool.suffix.lower() == ".py":
        return [sys.executable, str(tool)]
    return [str(tool)]


def run_realesrgan_on_dir(cmd: List[str], input_dir: Path, output_dir: Path, extra_args: str, dry_run: bool = False, verbosity: int = 0) -> int:
    """
    Attempt to run Real-ESRGAN once on the whole directory using -i/-o.
    Returns process return code (0 means success). If dry_run, returns 0.
    """
    # Split extra args respecting quotes
    import shlex

    args = cmd + ["-i", str(input_dir), "-o", str(output_dir)]
    if extra_args:
        args += shlex.split(extra_args)

    vprint(f"[Real-ESRGAN] Command: {' '.join(args)}", 3, verbosity)
    if dry_run:
        return 0
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    vprint(proc.stdout, 3, verbosity)

    error_lines = []
    error_keywords = ["error", "failed", "invalid", "exception", "unable", "not found", "denied"]

    if proc.stderr:
        for line in proc.stderr.splitlines():
            line_stripped = line.strip()
            # Check for error keywords only
            if any(kw in line_stripped.lower() for kw in error_keywords):
                error_lines.append(line)

    if error_lines:
        print("[Error] Real-ESRGAN reported the following error output:")
        for err in error_lines:
            print(err)
        return 1

    return proc.returncode


def patch_mips_in_place(root: Path, dry_run: bool = False, verbosity: int = 0) -> None:
    """
    For files matching *-mipN*.{ext}, copy the highest mip (the file WITHOUT the -mip suffix)
    to all other mip levels in the same group.
    Grouping is based on file stem sans the -mipN part within each directory and extension.
    Only overwrites existing files; does not create new ones.
    """
    vprint(f"[Mips] Patching mip levels under: {root}", 2, verbosity)

    # Map[(dirpath, key, ext)] -> {mip_num: Path}
    groups: Dict[Tuple[Path, str, str], Dict[int, Path]] = {}
    # Map to the base (no -mip) image for each group, if present
    base_map: Dict[Tuple[Path, str, str], Path] = {}

    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        names_set = set(filenames)
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = Path(name).stem  # filename without extension
            m = MIP_REGEX.match(stem)
            if not m:
                continue
            prefix = m.group("prefix")
            suffix = m.group("suffix")
            try:
                mip = int(m.group("mip"))
            except ValueError:
                continue
            key = f"{prefix}__{suffix}"
            groups.setdefault((d, key, ext), {})[mip] = d / name

            # Detect the base (no -mip) file in the same directory
            base_name = f"{prefix}{suffix}{ext}"
            if base_name in names_set:
                base_map[(d, key, ext)] = d / base_name

    total_overwrites = 0
    for (d, key, ext), mip_map in groups.items():
        if not mip_map:
            continue
        src_path = base_map.get((d, key, ext))
        if src_path is None:
            # Fallback to the smallest mip index (legacy behavior) if no base is present
            highest_mip = min(mip_map.keys())  # usually 0
            src_path = mip_map[highest_mip]
        for mip, dst_path in mip_map.items():
            if src_path == dst_path:
                continue
            vprint(f"[Mips] Overwrite {dst_path.name} with {src_path.name}", 2, verbosity)
            total_overwrites += 1
            if not dry_run:
                # Overwrite bytes
                shutil.copyfile(src_path, dst_path)
    if dry_run:
        vprint(f"  Would have patched {total_overwrites} files (dry run, no changes made).", 1, verbosity)
    else:
        vprint(f"  Patched {total_overwrites} files.", 1, verbosity)


def copy_tree(src: Path, dst: Path, dry_run: bool = False, verbosity: int = 0) -> None:
    vprint(f"[Copy] Mirroring {src} -> {dst}", 3, verbosity)
    copied_count = 0
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        out_dir = dst / rel if rel != "." else dst
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            src_file = Path(dirpath) / fn
            dst_file = out_dir / fn
            vprint(f"[Copy] {src_file} -> {dst_file}", 3, verbosity)
            if not dry_run:
                shutil.copy2(src_file, dst_file)
            copied_count += 1
    if dry_run:
        vprint(f"  Would have copied {copied_count} files (dry run, no changes made).", 1, verbosity)
    else:
        vprint(f"  Copied {copied_count} files.", 1, verbosity)


def main() -> None:
    args = parse_args()
    verbosity = args.verbose

    # Single banner line
    if args.game:
        print(f"Running AutoUpscalePS2 on {args.game}...")
    else:
        print(f"Running AutoUpscalePS2 on {args.input}...")

    # Check for mutually exclusive arguments
    if args.game:
        if args.input or args.intermediate or args.output:
            print("[Error] Cannot use -g/--game with -i/--input, -m/--intermediate, or -o/--output.")
            sys.exit(1)
        base = Path(args.game).resolve()
        input_dir = base / "dumps"
        interm_dir = base / "intermediates"
        output_dir = base / "replacements"
        # Sanity check: game folder must contain 'dumps' subfolder
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"[Error] The specified game texture folder does not contain a 'dumps' subfolder: {input_dir}")
            sys.exit(1)
    else:
        if not (args.input and args.intermediate and args.output):
            print("[Error] Must specify either -g/--game or all of -i/--input, -m/--intermediate, -o/--output.")
            sys.exit(1)
        input_dir = Path(args.input).resolve()
        interm_dir = Path(args.intermediate).resolve()
        output_dir = Path(args.output).resolve()

    # Initial echo of user request (under -v)
    vprint("[Config] Requested upscaling operation:", 1, verbosity)
    if args.game:
        vprint(f"  Game texture folder: {args.game}", 1, verbosity)
        vprint(f"  Input: {input_dir}\n  Intermediate: {interm_dir}\n  Output: {output_dir}", 1, verbosity)
    else:
        vprint(f"  Input: {input_dir}\n  Intermediate: {interm_dir}\n  Output: {output_dir}", 1, verbosity)
    vprint(f"  Real-ESRGAN: {args.realesrgan}", 1, verbosity)
    if args.realesrgan_args:
        vprint(f"  Real-ESRGAN extra args: {args.realesrgan_args}", 1, verbosity)
    if args.dry_run:
        print("  Dry run: enabled (no changes will be made)")

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[Error] Input folder does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    if not args.dry_run:
        interm_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Count input image files
    input_files = []
    for dirpath, _, filenames in os.walk(input_dir):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in IMAGE_EXTS:
                input_files.append(os.path.join(dirpath, fn))
    vprint(f"[Step 1] Running Real-ESRGAN on input folder -> intermediate folder", 1, verbosity)
    vprint(f"  Found {len(input_files)} input image files.", 1, verbosity)
    realesrgan_path = Path(args.realesrgan)
    cmd = discover_realesrgan_cmd(realesrgan_path)
    rc = run_realesrgan_on_dir(cmd, input_dir, interm_dir, args.realesrgan_args, dry_run=args.dry_run, verbosity=verbosity)
    if rc != 0:
        print("[Warning] Real-ESRGAN returned a non-zero exit code or error output. Check the command and paths.")
        sys.exit(rc)

    vprint("[Step 2] Patching mip levels in intermediate folder", 1, verbosity)
    patch_mips_in_place(interm_dir, dry_run=args.dry_run, verbosity=verbosity)

    vprint("[Step 3] Copying upscaled textures to final output folder", 1, verbosity)
    copy_tree(interm_dir, output_dir, dry_run=args.dry_run, verbosity=verbosity)

    vprint("[Done] Upscaling pipeline completed.", 0, verbosity)


if __name__ == "__main__":
    main()

