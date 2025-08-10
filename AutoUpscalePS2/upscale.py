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


def is_eligible_filename(stem: str) -> bool:
    """Return True if the texture filename stem has at least two dashes (>=3 segments).

    Requirement: We now ignore files that have fewer than two dashes in *all* pipeline stages.
    This keeps only names consistent with the expected ID pattern (prefix-ID-suffix...).
    """
    return stem.count('-') >= 2


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
    # Target-based iterative upscaling options
    parser.add_argument(
        "--target-max-dim",
        type=int,
        default=None,
        help=(
            "If set, iteratively upscale each eligible texture until its max(width,height) reaches this value without exceeding it. "
            "Image is scaled using the largest allowed scale factor that does not overshoot the target."
        ),
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="4,2",
        help=(
            "Comma-separated list of integer scale factors to attempt (in order). Example: 4,2 . "
            "Used only with --target-max-dim."
        ),
    )
    parser.add_argument(
        "--min-preserve-dim",
        type=int,
        default=8,
        help=(
            "Do not upscale textures whose max(width,height) <= this value (they are copied verbatim to intermediates)."
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
    parser.add_argument(
        "-c", "--clean", action="store_true",
        help="Wipe the intermediate cache folder before processing (forces full re-upscale)."
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
            vprint(line.rstrip(), 3, verbosity)

    def read_stderr():
        if not proc.stderr:
            return
        for line in proc.stderr:
            ls = line.strip()
            if any(kw in ls.lower() for kw in error_keywords):
                keyword_hits.append(line.rstrip())
            else:
                vprint(f"[Real-ESRGAN stderr] {ls}", 4, verbosity)

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


def run_realesrgan_scaled(
    base_cmd: List[str],
    scale: int,
    input_dir: Path,
    output_dir: Path,
    extra_args: str,
    dry_run: bool,
    verbosity: int,
) -> int:
    """Run Real-ESRGAN with an explicit scale factor (-s) when supported (ncnn binary). Fallback to normal run if not.

    We detect support heuristically: if the executable name contains 'ncnn' we append '-s <scale>' *unless* user already
    provided an explicit '-s' in extra_args. For Python scripts/models where scale selection is model-driven, the caller
    should provide appropriate --realesrgan-args for each pass (currently we only vary -s automatically for ncnn)."""
    # If user already specified -s we do not override.
    use_scale_flag = False
    if not re.search(r"(?<![A-Za-z0-9-])-s(\s|=)", extra_args):  # crude detection of -s presence
        exe_name = Path(base_cmd[-1]).name.lower()
        if "ncnn" in exe_name and scale not in (0, 1):
            use_scale_flag = True
    cmd = list(base_cmd)
    if use_scale_flag:
        cmd += ["-s", str(scale)]
    return run_realesrgan_on_dir(cmd, input_dir, output_dir, extra_args, dry_run=dry_run, verbosity=verbosity)


def iterative_target_upscale(
    cmd: List[str],
    interm_dir: Path,
    staged_rel_paths: List[Path],
    source_times: Dict[Path, Tuple[int, int]],
    target_max_dim: int,
    preserve_dim: int,
    scales: List[int],
    extra_args: str,
    dry_run: bool,
    verbosity: int,
) -> Dict[str, int]:
    """Iteratively upscale images so that max(w,h) approaches target_max_dim without exceeding it.

    Parameters:
        cmd: discovered Real-ESRGAN base command (without -i/-o/-s/-scale args)
        interm_dir: intermediate directory containing (or to contain) outputs
        staged_rel_paths: list of relative paths (relative to interm_dir) for images to process
        source_times: mapping of rel path -> (atime_ns, mtime_ns) so we can preserve original timestamps
        target_max_dim: target maximum dimension
        preserve_dim: images with max_dim <= preserve_dim are copied verbatim (skip upscaling)
        scales: ordered list of integer scale factors, typically descending (e.g. [4,2])
        extra_args: passthrough extra args string
    Returns stats dict.
    """
    stats = {
        "preserved_small": 0,
        "upscaled_files": 0,
        "passes": 0,
    }
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        print("[Target][Warn] Pillow not installed; falling back to single-pass behavior.")
        # Single pass fallback: treat as one run on all images larger than preserve_dim using largest scale that fits
        # (We cannot adjust scale per image without Pillow; just run once). We let caller perform default earlier path.
        return stats

    # Ensure scales are positive ints, remove duplicates while preserving order
    dedup_scales: List[int] = []
    for s in scales:
        if s > 1 and s not in dedup_scales:
            dedup_scales.append(s)
    if not dedup_scales:
        dedup_scales = [2]
    # Sort descending so we attempt largest first if user didn't already order
    dedup_scales.sort(reverse=True)

    # Initial copy of small preserved images (those that will never be upscaled) from temp input already present in interm_dir
    # At this point caller will have placed the first-pass (original) files inside a staging dir; we copy them directly here.
    # For simplicity we assume caller has already copied originals into interm_dir for small preserved ones.

    # Build set for quick membership
    rel_set = set(staged_rel_paths)

    # Helper to get current dimensions of an image path
    def get_dims(p: Path) -> Optional[Tuple[int, int]]:
        try:
            with Image.open(p) as im:
                return im.size
        except OSError:
            return None

    # Track which images are still candidates for further upscaling.
    # We'll maintain a mapping rel_path -> current max dimension.
    candidate_paths: Dict[Path, int] = {}
    for relp in staged_rel_paths:
        abs_path = interm_dir / relp
        dims = get_dims(abs_path)
        if not dims:
            continue
        mx = max(dims)
        if mx <= preserve_dim:
            stats["preserved_small"] += 1
            continue
        if mx >= target_max_dim:
            continue
        candidate_paths[relp] = mx

    if not candidate_paths:
        return stats

    pass_index = 0
    while candidate_paths:
        any_scaled_this_outer_loop = False
        # For each scale, gather files that can be scaled with that factor without exceeding target
        for scale in dedup_scales:
            batch: List[Path] = []
            for relp, cur_max in candidate_paths.items():
                if cur_max * scale <= target_max_dim and cur_max * scale > cur_max:
                    batch.append(relp)
            if not batch:
                continue
            pass_index += 1
            stats["passes"] = pass_index
            any_scaled_this_outer_loop = True
            # Prepare staging dirs
            staging_in = interm_dir / f"_pass{pass_index}_in"
            staging_out = interm_dir / f"_pass{pass_index}_out"
            if not dry_run:
                staging_in.mkdir(parents=True, exist_ok=True)
                staging_out.mkdir(parents=True, exist_ok=True)
            if verbosity >= 2:
                print(f"[Target][Pass {pass_index}] scale={scale} batch={len(batch)}")
            # Copy current versions into staging input
            for relp in batch:
                src = interm_dir / relp
                if not src.exists():
                    continue
                if not dry_run:
                    staging_in_sub = staging_in / relp.parent
                    staging_in_sub.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, staging_in_sub / relp.name)
            # Run realesrgan on this batch
            rc = run_realesrgan_scaled(
                cmd,
                scale,
                staging_in,
                staging_out,
                extra_args,
                dry_run,
                verbosity,
            )
            if rc != 0:
                print(f"[Target][Warn] Real-ESRGAN returned code {rc} on pass {pass_index}; aborting iterative upscale.")
                # Clean staging dirs
                if not dry_run:
                    shutil.rmtree(staging_in, ignore_errors=True)
                    shutil.rmtree(staging_out, ignore_errors=True)
                candidate_paths.clear()
                break
            # Merge outputs back into interm_dir
            for relp in batch:
                out_file = staging_out / relp
                dest_file = interm_dir / relp
                if not dry_run and out_file.exists():
                    try:
                        at_ns, mt_ns = source_times.get(relp, (None, None))
                        shutil.copy2(out_file, dest_file)
                        if at_ns is not None and mt_ns is not None:
                            try:
                                os.utime(dest_file, ns=(at_ns, mt_ns))
                            except OSError:
                                pass
                    except OSError as e:
                        print(f"[Target][Error] Failed to integrate output {out_file}: {e}")
                        continue
                    stats["upscaled_files"] += 1
            # Recompute candidate sizes (only for those still below target)
            to_remove: List[Path] = []
            for relp in batch:
                abs_path = interm_dir / relp
                dims = get_dims(abs_path)
                if not dims:
                    to_remove.append(relp)
                    continue
                mx = max(dims)
                if mx >= target_max_dim:
                    to_remove.append(relp)
                else:
                    candidate_paths[relp] = mx
            for relp in to_remove:
                candidate_paths.pop(relp, None)
            # Cleanup staging directories
            if not dry_run:
                shutil.rmtree(staging_in, ignore_errors=True)
                shutil.rmtree(staging_out, ignore_errors=True)
        if not any_scaled_this_outer_loop:
            # No scale factor could be applied without overshoot; break to avoid infinite loop
            break
    return stats


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
                # Preserve existing destination timestamps to avoid triggering downstream sync each run
                try:
                    try:
                        dst_stat = dst_path.stat()
                        preserve_times = (getattr(dst_stat, 'st_atime_ns', int(dst_stat.st_atime*1e9)), getattr(dst_stat, 'st_mtime_ns', int(dst_stat.st_mtime*1e9)))
                    except FileNotFoundError:
                        preserve_times = None
                    shutil.copyfile(src_path, dst_path)
                    if preserve_times is not None:
                        try:
                            os.utime(dst_path, ns=preserve_times)
                        except OSError as e2:
                            vprint(f"[Mips][Warn] Failed to restore timestamp on {dst_path}: {e2}", 4, verbosity)
                except PermissionError:
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


def copy_changed(src: Path, dst: Path, dry_run: bool = False, verbosity: int = 0, errors: Optional[List[str]] = None) -> int:
    """Copy only files whose modification timestamp differs or that do not exist in destination.

    Uses nanosecond resolution where available. Skips copy if destination exists and mtime_ns matches.
    Returns number of files actually copied.
    """
    vprint(f"[CopyΔ] Syncing {src} -> {dst} (timestamp-based)", 3, verbosity)
    copied = 0
    for dirpath, _, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        out_dir = dst / rel if rel != '.' else dst
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            sfile = Path(dirpath) / fn
            dfile = out_dir / fn
            try:
                s_stat = sfile.stat()
            except OSError as e:
                msg = f"[Error] Cannot stat source file {sfile}: {e}"
                if errors is not None:
                    errors.append(msg)
                vprint(msg, 2, verbosity)
                continue
            needs_copy = True
            if dfile.exists():
                try:
                    d_stat = dfile.stat()
                    if getattr(d_stat, 'st_mtime_ns', int(d_stat.st_mtime*1e9)) == getattr(s_stat, 'st_mtime_ns', int(s_stat.st_mtime*1e9)):
                        needs_copy = False
                except OSError:
                    needs_copy = True
            if not needs_copy:
                vprint(f"[CopyΔ][Skip] {dfile} (timestamps equal)", 4, verbosity)
                continue
            vprint(f"[CopyΔ] {sfile} -> {dfile}", 3, verbosity)
            if not dry_run:
                try:
                    shutil.copy2(sfile, dfile)
                except PermissionError:
                    msg = f"[Error] Permission denied copying to: {dfile}"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
                except OSError as e:
                    msg = f"[Error] Failed to copy {sfile} -> {dfile}: {e.strerror or e}"
                    print(msg)
                    if errors is not None:
                        errors.append(msg)
                    continue
            copied += 1
    if dry_run:
        vprint(f"  Would have copied/updated {copied} files (dry run).", 1, verbosity)
    else:
        vprint(f"  Copied/updated {copied} files (timestamp delta).", 1, verbosity)
    return copied


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
            2. Only consider stems containing at least TWO dashes (i.e. three or more segments when split by '-'). Take segment[1] as group key.
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
            vprint(msg, 3, verbosity)
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
            # Require at least two dashes in the stem (three segments) so we have a stable middle ID segment.
            if stem.count('-') < 2:
                vprint(f"[IDReplace][SkipDash] {stem} (needs >=2 dashes)", 4, verbosity)
                continue
            parts = stem.split('-')  # now guaranteed len(parts) >= 3
            key = parts[1]
            full_path = Path(dirpath) / fn
            try:
                with Image.open(full_path) as im:
                    w, h = im.size
            except OSError as e:
                msg = f"[IDReplace][Warn] Cannot open {full_path}: {e}"
                vprint(msg, 3, verbosity)
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
                vprint(f"[IDReplace] Skip (not similar) {small_path.name} vs {best_path.name} hamming={dist} > {SIM_HASH_THRESHOLD}", 3, verbosity)
                continue
            vprint(
                f"[IDReplace] {small_path.name} ({sw}x{sh}) <- {best_path.name} ({bw}x{bh}) [key={key}] hamming={dist}",
                2,
                verbosity,
            )
            if not dry_run:
                # Preserve original destination (small_path) timestamps after content replacement
                try:
                    try:
                        dst_stat = small_path.stat()
                        preserve_times = (getattr(dst_stat, 'st_atime_ns', int(dst_stat.st_atime*1e9)), getattr(dst_stat, 'st_mtime_ns', int(dst_stat.st_mtime*1e9)))
                    except FileNotFoundError:
                        preserve_times = None
                    shutil.copy2(best_path, small_path)
                    if preserve_times is not None:
                        try:
                            os.utime(small_path, ns=preserve_times)
                        except OSError as e2:
                            vprint(f"[IDReplace][Warn] Failed to restore timestamp on {small_path}: {e2}", 4, verbosity)
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

    # Optional clean of intermediates
    if args.clean:
        if args.dry_run:
            vprint(f"[Clean] Would remove all contents of intermediate folder: {interm_dir}", 1, verbosity)
        else:
            vprint(f"[Clean] Purging intermediate folder: {interm_dir}", 1, verbosity)
            try:
                if interm_dir.exists():
                    shutil.rmtree(interm_dir)
                interm_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"[Clean][Error] Failed to clean intermediates: {e}")
                sys.exit(2)

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

    # Step 1: Determine new eligible inputs and stage them for upscaling
    vprint("[Step 1] Scanning input for new eligible textures (>=2 dashes)", 1, verbosity)
    staged_files: List[Path] = []  # Newly seen (no prior intermediate)
    reupscale_files: List[Path] = []  # Source newer than existing intermediate -> re-upscale
    skipped_existing = 0  # Existing and up-to-date
    skipped_ineligible = 0
    source_times: Dict[Path, Tuple[int, int]] = {}  # rel path -> (atime_ns, mtime_ns) from original dump
    temp_in_dir = interm_dir / "_pending_upscale"
    if not args.dry_run and temp_in_dir.exists():
        shutil.rmtree(temp_in_dir)
    # Track relative paths for all files that will go through upscaling logic (to drive iterative target mode)
    staged_rel_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(input_dir):
        rel_dir = os.path.relpath(dirpath, input_dir)
        for fn in filenames:
            src_path = Path(dirpath) / fn
            ext = src_path.suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = src_path.stem
            if not is_eligible_filename(stem):
                skipped_ineligible += 1
                continue
            rel_target_dir = interm_dir / (rel_dir if rel_dir != '.' else '')
            dest_path = rel_target_dir / fn
            rel_out_path = (Path(rel_dir) / fn) if rel_dir != '.' else Path(fn)
            try:
                s_stat = src_path.stat()
                s_atime_ns = getattr(s_stat, 'st_atime_ns', int(s_stat.st_atime*1e9))
                s_mtime_ns = getattr(s_stat, 'st_mtime_ns', int(s_stat.st_mtime*1e9))
            except OSError as e:
                vprint(f"[Stage][Warn] Cannot stat source {src_path}: {e}", 2, verbosity)
                continue
            if dest_path.exists():
                try:
                    d_stat = dest_path.stat()
                    d_mtime_ns = getattr(d_stat, 'st_mtime_ns', int(d_stat.st_mtime*1e9))
                except OSError:
                    d_mtime_ns = -1
                if s_mtime_ns > d_mtime_ns:  # Source updated since last upscale
                    reupscale_files.append(src_path)
                else:
                    skipped_existing += 1
                    vprint(f"[Stage][SkipUpToDate] {dest_path}", 4, verbosity)
                    continue
            else:
                staged_files.append(src_path)
            # Stage file (new or re-upscale) into temp input
            source_times[rel_out_path] = (s_atime_ns, s_mtime_ns)
            staged_rel_paths.append(rel_out_path)
            if not args.dry_run:
                target_stage_dir = temp_in_dir / (rel_dir if rel_dir != '.' else '')
                target_stage_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_path, target_stage_dir / fn)
                except OSError as e:
                    msg = f"[Error] Failed staging {src_path}: {e.strerror or e}"
                    print(msg)
                    continue
            if verbosity >= 3:  # avoid computing kind & membership checks unless we'll log
                kind = "Updated" if src_path in reupscale_files else "New"
                vprint(f"[Stage] {kind} {src_path}", 3, verbosity)
    vprint(
        f"  New: {len(staged_files)} | Updated: {len(reupscale_files)} | Up-to-date skipped: {skipped_existing} | Ineligible skipped: {skipped_ineligible}",
        1,
        verbosity,
    )
    total_to_process = len(staged_files) + len(reupscale_files)
    if total_to_process == 0:
        vprint("[Step 1] No new or updated textures to upscale (continuing with existing intermediates)", 1, verbosity)
        t_step1 = perf_counter(); t_step1_end = t_step1
    else:
        t_step1 = perf_counter()
        realesrgan_path = Path(args.realesrgan).resolve()
        cmd = discover_realesrgan_cmd(realesrgan_path)
        if verbosity >= 2:
            vprint(f"[Discover] Using Real-ESRGAN command: {' '.join(cmd)}", 2, verbosity)
        # Two modes: target-based iterative vs legacy single-pass
        if args.target_max_dim:
            # First, copy originals directly into interm_dir so iterative function can read them.
            if not args.dry_run:
                for relp in staged_rel_paths:
                    src_file = temp_in_dir / relp
                    dest_file = interm_dir / relp
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if src_file.exists():
                        try:
                            shutil.copy2(src_file, dest_file)
                        except OSError as e:
                            print(f"[Error] Failed initial copy for {src_file}: {e}")
                # Apply original timestamps
                for rel_path, (at_ns, mt_ns) in source_times.items():
                    out_path = interm_dir / rel_path
                    if out_path.exists():
                        try:
                            os.utime(out_path, ns=(at_ns, mt_ns))
                        except OSError:
                            pass
            # Parse scale list
            try:
                scale_list = [int(s.strip()) for s in args.scales.split(',') if s.strip()]
            except ValueError:
                print("[Target][Error] Invalid --scales list; must be comma-separated integers.")
                sys.exit(2)
            target_stats = iterative_target_upscale(
                cmd,
                interm_dir,
                staged_rel_paths,
                source_times,
                args.target_max_dim,
                args.min_preserve_dim,
                scale_list,
                args.realesrgan_args,
                args.dry_run,
                verbosity,
            )
            if verbosity >= 1:
                vprint(
                    f"[Target] Iterative passes: {target_stats.get('passes',0)} | Upscaled: {target_stats.get('upscaled_files',0)} | Preserved small: {target_stats.get('preserved_small',0)}",
                    1,
                    verbosity,
                )
            if not args.dry_run and temp_in_dir.exists():
                shutil.rmtree(temp_in_dir, ignore_errors=True)
        else:
            # Legacy single-pass behavior
            rc = run_realesrgan_on_dir(
                cmd,
                temp_in_dir,
                interm_dir,
                args.realesrgan_args,
                dry_run=args.dry_run,
                verbosity=verbosity,
            )
            if rc != 0:
                print("[Warning] Real-ESRGAN returned a non-zero exit code or error output. Check the command and paths.")
                if not args.dry_run and temp_in_dir.exists():
                    shutil.rmtree(temp_in_dir, ignore_errors=True)
                sys.exit(rc)
            if not args.dry_run:
                for rel_path, (at_ns, mt_ns) in source_times.items():
                    out_path = interm_dir / rel_path
                    if out_path.exists():
                        try:
                            os.utime(out_path, ns=(at_ns, mt_ns))
                        except OSError:
                            pass
                shutil.rmtree(temp_in_dir, ignore_errors=True)
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

    if args.clean:
        vprint("[Step 3] Clean mode: full mirror intermediates -> final (bypassing timestamp delta)", 1, verbosity)
        t_step3 = perf_counter()
        files_copied = copy_tree(interm_dir, output_dir, dry_run=args.dry_run, verbosity=verbosity, errors=errors)
    else:
        vprint("[Step 3] Syncing intermediates -> final (timestamp delta)", 1, verbosity)
        t_step3 = perf_counter()
        files_copied = copy_changed(interm_dir, output_dir, dry_run=args.dry_run, verbosity=verbosity, errors=errors)
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
            print(f"  Upscaled images - new: {len(staged_files)} updated: {len(reupscale_files)} (total {total_to_process}) - processed in {step1_time:.2f}s")
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

