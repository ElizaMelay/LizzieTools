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
from typing import Dict, List, Tuple, Optional


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
    # ID-based small->large replacement feature
    parser.add_argument(
        "--id-replace",
        action="store_true",
        help=(
            "Enable replacement of small textures with larger ones sharing the same second ID segment (between first and second dashes)."
        ),
    )
    parser.add_argument(
        "--id-small-threshold",
        type=int,
        default=256,
        help="Max dimension below which a texture is considered SMALL for ID-based replacement (default 256)",
    )
    parser.add_argument(
        "--id-large-threshold",
        type=int,
        default=256,
        help="Min dimension at or above which a texture is considered LARGE for ID-based replacement (default 256)",
    )
    parser.add_argument(
        "--id-hash-threshold",
        type=int,
        default=6,
        help="Max aHash Hamming distance to allow replacement for ID-based similarity (default 6)",
    )
    parser.add_argument(
        "--id-hash-size",
        type=int,
        default=8,
        help="aHash size (NxN) for ID-based similarity (4-16, default 8)",
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
    """Run Real-ESRGAN once on the directory using -i/-o.
    Streams stderr to capture only lines with error keywords while ignoring progress noise.
    Returns process return code (0 means success) unless keyword errors were detected.
    """
    import shlex
    import threading

    args = cmd + ["-i", str(input_dir), "-o", str(output_dir)]
    if extra_args:
        args += shlex.split(extra_args)

    vprint(f"[Step 1] Real-ESRGAN command: {' '.join(args)}", 3, verbosity)
    if dry_run:
        return 0

    error_keywords = ["error", "failed", "invalid", "exception", "unable", "not found", "denied"]
    keyword_hits: List[str] = []

    # Use Popen for incremental stderr read (avoid buffering huge progress output)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    def read_stdout():
        if not proc.stdout:
            return
        for line in proc.stdout:
            if verbosity >= 3:
                print(line.rstrip())

    def read_stderr():
        if not proc.stderr:
            return
        for line in proc.stderr:
            ls = line.strip()
            # We purposely do NOT treat percentage or other non-keyword lines as errors
            if any(kw in ls.lower() for kw in error_keywords):
                keyword_hits.append(line.rstrip())
            elif verbosity >= 4:  # ultra-verbose raw stderr
                print(f"[Real-ESRGAN stderr] {ls}")

    t_out = threading.Thread(target=read_stdout)
    t_err = threading.Thread(target=read_stderr)
    t_out.start(); t_err.start()
    # Wait for process to exit; threads keep draining pipes until EOF.
    proc.wait()
    # Fully join reader threads (no timeout) to ensure all remaining lines processed.
    t_out.join(); t_err.join()

    if keyword_hits:
        print("[Error] Real-ESRGAN emitted error indicators:")
        for l in keyword_hits:
            print(l)
        # Prefer non-zero code if present, else synthesize failure code 1
        return proc.returncode if proc.returncode != 0 else 1

    return proc.returncode


def patch_mips_in_place(root: Path, dry_run: bool = False, verbosity: int = 0, errors: Optional[List[str]] = None) -> int:
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
    anomalous_groups = 0
    for (d, key, ext), mip_map in groups.items():
        if not mip_map:
            continue
        src_path = base_map.get((d, key, ext))
        if src_path is None:
            # Fallback to the smallest mip index (legacy behavior) if no base is present
            highest_mip = min(mip_map.keys())  # usually 0
            src_path = mip_map[highest_mip]
        # Check for gaps in mip indices (e.g., have 0,2 but missing 1)
        mip_indices = sorted(mip_map.keys())
        expected = list(range(mip_indices[0], mip_indices[-1] + 1))
        if mip_indices != expected:
            anomalous_groups += 1
            vprint(f"[Mips][Warn] Non-contiguous mip chain in {d}: {mip_indices}", 3, verbosity)
        for mip, dst_path in mip_map.items():
            if src_path == dst_path:
                continue
            vprint(f"[Mips] Overwrite {dst_path.name} with {src_path.name}", 2, verbosity)
            total_overwrites += 1
            if not dry_run:
                try:
                    shutil.copyfile(src_path, dst_path)
                except PermissionError as e:
                    msg = f"[Error] Permission denied while overwriting mip file: {dst_path} (is it read-only?)"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                except OSError as e:
                    msg = f"[Error] Failed to overwrite mip file: {dst_path} ({e.strerror or e})"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
    if dry_run:
        vprint(f"  Would have patched {total_overwrites} files (dry run).", 1, verbosity)
    else:
        vprint(f"  Patched {total_overwrites} files (anomalous groups: {anomalous_groups}).", 1, verbosity)
    return total_overwrites


def copy_tree(src: Path, dst: Path, dry_run: bool = False, verbosity: int = 0, errors: Optional[List[str]] = None) -> int:
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
                try:
                    shutil.copy2(src_file, dst_file)
                except PermissionError:
                    msg = f"[Error] Permission denied copying to: {dst_file} (directory or file may be read-only)"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
                except OSError as e:
                    msg = f"[Error] Failed to copy {src_file} -> {dst_file}: {e.strerror or e}"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
            copied_count += 1
    if dry_run:
        vprint(f"  Would have copied {copied_count} files (dry run).", 1, verbosity)
    else:
        vprint(f"  Copied {copied_count} files.", 1, verbosity)
    return copied_count


def replace_small_id_variants(
    root: Path,
    small_thresh: int,
    large_thresh: int,
    hash_size: int,
    hash_threshold: int,
    dry_run: bool = False,
    verbosity: int = 0,
    errors: Optional[List[str]] = None,
) -> int:
    """Replace small textures (< small_thresh) with a large texture (>= large_thresh) when they share
    the same second ID segment (filename stem split by '-'). Only considers non-mip files.

    Strategy:
      1. Scan all image files in root recursively.
      2. For each file whose stem contains at least two dashes, take segment[1] as group key.
      3. Within each group decide a canonical large image (largest area; tie -> first encountered).
      4. For each small image in group (and not itself the chosen large) compute similarity to the
         canonical large using perceptual average-hash (aHash). Only replace if Hamming distance
         <= SIM_HASH_THRESHOLD (hard‑coded) after scaling logic (hash is size invariant).
    Returns number of replacements performed (post similarity filter).
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print("[IDReplace][Error] Pillow not installed; cannot inspect image sizes. Skipping.")
        return 0

    # Hash parameters are now configurable
    SIM_HASH_SIZE = hash_size
    SIM_HASH_THRESHOLD = hash_threshold

    # Validate early (in case function is reused independently of CLI validation)
    if SIM_HASH_SIZE < 4 or SIM_HASH_SIZE > 16:
        print(f"[IDReplace][Error] Invalid hash size: {SIM_HASH_SIZE} (must be 4..16)")
        return 0
    if SIM_HASH_THRESHOLD < 0 or SIM_HASH_THRESHOLD > SIM_HASH_SIZE * SIM_HASH_SIZE:
        print(f"[IDReplace][Error] Invalid hash threshold: {SIM_HASH_THRESHOLD}")
        return 0

    def compute_ahash(path: Path) -> Optional[int]:
        """Compute an average hash (aHash) for similarity gating. Returns 64-bit int or None on failure."""
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                im_small = im.resize((SIM_HASH_SIZE, SIM_HASH_SIZE), Image.Resampling.LANCZOS)
                pixels = list(im_small.getdata())
                mean_val = sum(pixels) / len(pixels)
                bits = 0
                for i, px in enumerate(pixels):
                    if px >= mean_val:
                        bits |= 1 << i
                return bits
        except OSError as e:
            msg = f"[IDReplace][Warn] Cannot hash {path}: {e}"
            if verbosity >= 3:
                print(msg)
            if errors is not None:
                errors.append(msg)
            return None

    def hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    vprint(f"[IDReplace] Scanning for candidate textures under {root}", 2, verbosity)
    groups: Dict[str, List[Tuple[Path, int, int]]] = {}
    # Collect files
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = Path(fn).stem
            # Skip mip variants
            if MIP_REGEX.match(stem):
                continue
            parts = stem.split('-')
            if len(parts) < 3:
                continue  # not in expected pattern
            key = parts[1]
            full_path = Path(dirpath) / fn
            try:
                with Image.open(full_path) as im:
                    w, h = im.size
            except OSError as e:
                msg = f"[IDReplace][Warn] Cannot open {full_path}: {e}"
                if verbosity >= 3:
                    print(msg)
                if errors is not None:
                    errors.append(msg)
                continue
            groups.setdefault(key, []).append((full_path, w, h))

    replacements = 0
    similarity_skips = 0
    hash_cache: Dict[Path, int] = {}
    def get_hash(p: Path) -> Optional[int]:
        if p in hash_cache:
            return hash_cache[p]
        h = compute_ahash(p)
        if h is not None:
            hash_cache[p] = h
        return h
    for key, items in groups.items():
        # Partition into large and small based on thresholds
        large_items = [it for it in items if max(it[1], it[2]) >= large_thresh]
        small_items = [it for it in items if max(it[1], it[2]) < small_thresh]
        if not large_items or not small_items:
            continue
        # Pre-compute hashes for large candidates; skip group if none hashable
        large_info: List[Tuple[Path, int, int, Optional[int]]] = []  # path, w, h, hash
        for lp, lw, lh in large_items:
            hval = get_hash(lp)
            if hval is not None:
                large_info.append((lp, lw, lh, hval))
        if not large_info:
            if verbosity >= 2:
                print(f"[IDReplace] Skipping group key={key} (no hashable large images)")
            continue
        for small_path, sw, sh in small_items:
            small_hash = get_hash(small_path)
            if small_hash is None:
                continue
            # Find best large candidate (min hamming). Tie-breaker: larger area, then name
            best_candidate: Optional[Tuple[int, Path, int, int]] = None  # dist, path, w, h
            for lp, lw, lh, lhash in large_info:
                dist = hamming(lhash, small_hash)
                if best_candidate is None or dist < best_candidate[0] or (
                    dist == best_candidate[0] and (lw*lh) > (best_candidate[2]*best_candidate[3])
                ):
                    best_candidate = (dist, lp, lw, lh)
                if dist == 0:  # perfect match early exit
                    break
            assert best_candidate is not None
            dist, best_path, bw, bh = best_candidate
            if dist > SIM_HASH_THRESHOLD:
                similarity_skips += 1
                if verbosity >= 3:
                    print(f"[IDReplace] Skip (not similar) {small_path.name} vs {best_path.name} hamming={dist} > {SIM_HASH_THRESHOLD}")
                continue
            vprint(
                f"[IDReplace] {small_path.name} ({sw}x{sh}) <- {best_path.name} ({bw}x{bh}) [key={key}] hamming={dist}",
                2,
                verbosity,
            )
            if not dry_run:
                try:
                    shutil.copy2(best_path, small_path)
                except PermissionError:
                    msg = f"[IDReplace][Error] Permission denied overwriting {small_path}"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
                except OSError as e:
                    msg = f"[IDReplace][Error] Failed to overwrite {small_path}: {e.strerror or e}"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
            replacements += 1

    if dry_run:
        vprint(f"[IDReplace] Would have replaced {replacements} small textures (dry run). Similarity skips: {similarity_skips}", 1, verbosity)
    else:
        vprint(f"[IDReplace] Replaced {replacements} small textures. Similarity skips: {similarity_skips}", 1, verbosity)
    return replacements


def main() -> None:
    args = parse_args()
    verbosity = args.verbose
    from time import perf_counter

    t_start = perf_counter()

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
    if args.id_replace:
        vprint(
            f"  ID Replace: small<={args.id_small_threshold} large>={args.id_large_threshold} hash_size={args.id_hash_size} hash_threshold={args.id_hash_threshold}",
            1,
            verbosity,
        )
    if args.dry_run:
        print("  Dry run: enabled (no changes will be made)")

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[Error] Input folder does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    # Safety: disallow overlapping directories
    if input_dir == interm_dir or input_dir == output_dir or interm_dir == output_dir:
        print("[Error] Input, intermediate, and output directories must be distinct.")
        sys.exit(1)

    if not args.dry_run:
        interm_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Writable directory pre-checks (skip if dry-run)
    def check_writable(d: Path) -> bool:
        if args.dry_run:
            return True
        import tempfile
        try:
            with tempfile.TemporaryFile(dir=d):
                pass
            return True
        except PermissionError:
            print(f"[Error] Directory not writable (permission denied): {d}")
            return False
        except OSError as e:
            # Some environments might block creation differently
            print(f"[Error] Directory not writable: {d} ({e.strerror or e})")
            return False

    if not check_writable(interm_dir) or not check_writable(output_dir):
        sys.exit(2)

    # Optionally enumerate input files only if verbosity or dry_run (performance win for default quiet mode)
    input_files: List[str] = []
    if verbosity >= 1 or args.dry_run:
        for dirpath, _, filenames in os.walk(input_dir):
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in IMAGE_EXTS:
                    input_files.append(os.path.join(dirpath, fn))
        vprint(f"[Step 1] Running Real-ESRGAN on input folder -> intermediate folder", 1, verbosity)
        vprint(f"  Found {len(input_files)} input image files.", 1, verbosity)
        if len(input_files) == 0:
            print("[Info] No input images found. Nothing to do.")
            return
    else:
        vprint("[Step 1] Running Real-ESRGAN on input folder -> intermediate folder", 1, verbosity)

    t_step1 = perf_counter()
    realesrgan_path = Path(args.realesrgan)
    # Resolve path for clearer logging
    realesrgan_path = realesrgan_path.resolve()
    cmd = discover_realesrgan_cmd(realesrgan_path)
    if verbosity >= 2:
        vprint(f"[Discover] Using Real-ESRGAN command: {' '.join(cmd)}", 2, verbosity)
    rc = run_realesrgan_on_dir(cmd, input_dir, interm_dir, args.realesrgan_args, dry_run=args.dry_run, verbosity=verbosity)
    if rc != 0:
        print("[Warning] Real-ESRGAN returned a non-zero exit code or error output. Check the command and paths.")
        sys.exit(rc)
    t_step1_end = perf_counter()

    vprint("[Step 2] ID-based small texture replacement & mip patching", 1, verbosity)
    t_step2 = perf_counter()
    errors: List[str] = []
    id_replacements = 0
    if args.id_replace:
        vprint("[Step 2] ID-based small texture replacement (pre-mip patch)", 2, verbosity)
        id_replacements = replace_small_id_variants(
            interm_dir,
            args.id_small_threshold,
            args.id_large_threshold,
            args.id_hash_size,
            args.id_hash_threshold,
            dry_run=args.dry_run,
            verbosity=verbosity,
            errors=errors,
        )
    vprint("[Step 2] Patching mip levels in intermediate folder", 2, verbosity)
    mips_patched = patch_mips_in_place(interm_dir, dry_run=args.dry_run, verbosity=verbosity, errors=errors)
    t_step2_end = perf_counter()

    vprint("[Step 3] Copying upscaled textures to final output folder", 1, verbosity)
    t_step3 = perf_counter()
    files_copied = copy_tree(interm_dir, output_dir, dry_run=args.dry_run, verbosity=verbosity, errors=errors)
    t_step3_end = perf_counter()

    total_elapsed = perf_counter() - t_start
    if verbosity >= 0:
        if errors:
            print("[Summary] Pipeline completed with errors")
        else:
            print("[Summary] Pipeline complete")
        if verbosity >= 1:
            step1_time = t_step1_end - t_step1
            step2_time = t_step2_end - t_step2
            step3_time = t_step3_end - t_step3
            if input_files:
                print(f"  Input images: {len(input_files)} - upscaled in {step1_time:.2f}s")
            if args.id_replace:
                print(f"  Mips patched: {mips_patched} (ID replacements: {id_replacements}) - processed in {step2_time:.2f}s")
            else:
                print(f"  Mips patched: {mips_patched} - processed in {step2_time:.2f}s")
            print(f"  Files copied: {files_copied} - completed in {step3_time:.2f}s")
            print(f"  Total elapsed: {total_elapsed:.2f}s")
            if errors:
                print(f"  Errors: {len(errors)} (see above messages)")

    if errors:
        # Distinguish from Real-ESRGAN failure exit code (already handled earlier)
        sys.exit(3)

    vprint("[Done] Upscaling pipeline completed.", 0, verbosity)


if __name__ == "__main__":
    main()

