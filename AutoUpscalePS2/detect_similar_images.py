"""Detect similar non-mip textures in the intermediates folder.

Problem context
---------------
Some PS2 games use different texture assets for lower LOD meshes instead of
true mip chains. After an upscaling pass (e.g. with Real-ESRGAN) you may end up
with (a) a high‑resolution 1024x1024 (or larger) texture and (b) separate
lower‑resolution (e.g. 256x256 / 128x128) standalone textures that are *very*
similar (sometimes identical when scaled) but whose filenames do NOT contain
the "-mipN" pattern. These will not be patched by the regular mip patch step.

Goal
----
Identify candidate low‑resolution non‑mip textures that are visually similar to
larger non‑mip textures so they can optionally be replaced / removed / redirected.

Strategy
--------
1. Scan a target directory (normally the "intermediates" folder) recursively.
2. Collect only image files whose stem does NOT match the -mipN pattern.
3. Split into two sets:
	 high_res: max(width,height) >= --base-size (default 1024)
	 low_res:  max(width,height)  < --base-size
4. Compute a perceptual average hash (aHash) for each image (size --hash-size).
5. For each low_res image, find the high_res image with the smallest Hamming
   distance. If distance <= --hash-threshold, optionally verify using a mean
   absolute pixel difference after resizing (if --verify-diff is set) and if
   that passes (<= --diff-threshold) classify them as similar.
6. Output a report to stdout and/or an optional CSV file.

The hash pre-filter makes the comparison efficient while being robust to size
differences. The optional pixel diff adds a stricter confirmation when desired.

CLI examples
------------
  python detect_similar_images.py -g C:\\PCSX2\\textures\\<GAME_SERIAL>
  python detect_similar_images.py -i C:\\PCSX2\\textures\\<GAME_SERIAL>\\intermediates \\
	  --hash-threshold 6 --verify-diff --diff-threshold 12

Outputs: Lines like
  LOW 256x256 foo_1234.png  ->  HIGH 1024x1024 foo_abcd.png  (hamming=4, diff=7.9)

You can then decide to delete / alias / copy the high-res texture over the low
res one depending on your workflow.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
	from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - Pillow not installed yet
	print("[Error] Pillow is required. Install with: pip install Pillow", file=sys.stderr)
	raise

# Reuse similar pattern as upscale.py for identifying mip files
MIP_REGEX = re.compile(r"^(?P<prefix>.*)-mip(?P<mip>\d+)(?P<suffix>.*)$", re.IGNORECASE)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}


@dataclass
class ImgInfo:
	path: Path
	width: int
	height: int
	hash_bits: int  # packed bits as int
	hash_size: int  # dimension (e.g. 8 for 8x8)

	@property
	def size_tuple(self) -> Tuple[int, int]:
		return self.width, self.height


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Detect visually similar non-mip low-res textures to high-res base textures.")
	group = p.add_mutually_exclusive_group(required=True)
	group.add_argument("-i", "--intermediates", help="Path to intermediates folder to scan")
	group.add_argument("-g", "--game", help="Path to PCSX2 game texture root (expects intermediates subfolder)")
	p.add_argument("--base-size", type=int, default=1024, help="Size threshold (>= either dimension) to classify as high-res (default: 1024)")
	p.add_argument("--hash-size", type=int, default=8, help="aHash square dimension (default 8 => 64 bits)")
	p.add_argument("--hash-threshold", type=int, default=6, help="Maximum Hamming distance between hashes to consider candidates (default 6)")
	p.add_argument("--verify-diff", action="store_true", help="After hash match, compute mean absolute pixel diff to confirm")
	p.add_argument("--diff-threshold", type=float, default=12.0, help="Mean absolute difference (0-255) threshold if --verify-diff (default 12.0)")
	p.add_argument("--csv", help="Optional path to write matches as CSV")
	p.add_argument("--limit", type=int, help="Optional limit on number of low-res textures to process")
	p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
	return p.parse_args()


def is_non_mip_image(path: Path) -> bool:
	return path.suffix.lower() in IMAGE_EXTS and not MIP_REGEX.match(path.stem)


def iter_images(root: Path) -> Iterable[Path]:
	for dirpath, _dirnames, filenames in os.walk(root):
		for fn in filenames:
			p = Path(dirpath) / fn
			if is_non_mip_image(p):
				yield p


def compute_ahash(path: Path, hash_size: int) -> Tuple[int, int, int, int]:
	"""Return (hash_bits, width, height, hash_size). aHash method.

	Steps: convert to L (grayscale), resize to hash_size x hash_size with ANTIALIAS,
	compute mean, set bit=1 where pixel >= mean.
	"""
	with Image.open(path) as im:
		im = im.convert("L")
		width, height = im.size
		im_small = im.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
		pixels = list(im_small.getdata())
		mean_val = sum(pixels) / len(pixels)
		bits = 0
		for i, px in enumerate(pixels):
			if px >= mean_val:
				bits |= 1 << i
		return bits, width, height, hash_size


def hamming_distance(a: int, b: int) -> int:
	return (a ^ b).bit_count()


def mean_abs_diff_resized(low_path: Path, high_path: Path) -> float:
	"""Resize low image to high image size and compute mean absolute difference (0-255)."""
	with Image.open(low_path) as low_im, Image.open(high_path) as high_im:
		# Convert both to RGB for consistent channel count
		high_im = high_im.convert("RGB")
		low_im = low_im.convert("RGB").resize(high_im.size, Image.Resampling.BICUBIC)
		low_px = list(low_im.getdata())
		high_px = list(high_im.getdata())
		assert len(low_px) == len(high_px)
		total = 0
		for (r1, g1, b1), (r2, g2, b2) in zip(low_px, high_px):
			total += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
		# 3 channels
		return total / (len(low_px) * 3)


def classify(images: Sequence[ImgInfo], base_size: int) -> Tuple[List[ImgInfo], List[ImgInfo]]:
	high: List[ImgInfo] = []
	low: List[ImgInfo] = []
	for info in images:
		if max(info.width, info.height) >= base_size:
			high.append(info)
		else:
			low.append(info)
	return high, low


def load_images(root: Path, hash_size: int, verbose: int) -> List[ImgInfo]:
	out: List[ImgInfo] = []
	for p in iter_images(root):
		try:
			bits, w, h, hs = compute_ahash(p, hash_size)
			out.append(ImgInfo(p, w, h, bits, hs))
		except OSError as e:
			if verbose:
				print(f"[Warn] Failed to open {p}: {e}")
	return out


def main() -> None:
	args = parse_args()

	if args.game:
		base = Path(args.game).resolve()
		root = base / "intermediates"
		if not root.exists():
			print(f"[Error] intermediates folder does not exist: {root}")
			sys.exit(1)
	else:
		root = Path(args.intermediates).resolve()
	if not root.is_dir():
		print(f"[Error] Not a directory: {root}")
		sys.exit(1)

	if args.hash_size < 4:
		print("[Error] hash-size too small (min 4)")
		sys.exit(2)
	if args.hash_size > 16:
		print("[Error] hash-size too large (max 16 to keep performance reasonable)")
		sys.exit(2)

	print(f"[Scan] Root: {root}")
	images = load_images(root, args.hash_size, args.verbose)
	print(f"[Info] Non-mip images loaded: {len(images)}")
	if not images:
		return
	high, low = classify(images, args.base_size)
	print(f"[Info] High-res >= {args.base_size}: {len(high)}  | Low-res: {len(low)}")
	if not high or not low:
		print("[Info] Nothing to compare (need both high and low sets).")
		return

	# Sort for deterministic output
	high.sort(key=lambda i: (-(max(i.width, i.height)), i.path.name))
	low.sort(key=lambda i: (i.path.name,))
	if args.limit:
		low = low[: args.limit]

	matches: List[Tuple[ImgInfo, ImgInfo, int, Optional[float]]] = []
	for li in low:
		best: Optional[Tuple[ImgInfo, int]] = None
		for hi in high:
			dist = hamming_distance(li.hash_bits, hi.hash_bits)
			if best is None or dist < best[1]:
				best = (hi, dist)
			# Early exit if exact or below threshold 0
			if dist == 0:
				break
		assert best is not None
		hi, dist = best
		if dist <= args.hash_threshold:
			diff_val: Optional[float] = None
			if args.verify_diff:
				try:
					diff_val = mean_abs_diff_resized(li.path, hi.path)
				except OSError as e:
					if args.verbose:
						print(f"[Warn] Diff failed for {li.path} vs {hi.path}: {e}")
					continue
				if diff_val > args.diff_threshold:
					if args.verbose:
						print(f"[Skip] {li.path.name} best {hi.path.name} diff {diff_val:.1f} > threshold {args.diff_threshold}")
					continue
			matches.append((li, hi, dist, diff_val))

	if not matches:
		print("[Result] No similar low-res textures found under thresholds.")
		return

	print("[Result] Potential similar low-res -> high-res pairs:")
	for li, hi, dist, diff_val in matches:
		size_l = f"{li.width}x{li.height}"
		size_h = f"{hi.width}x{hi.height}"
		extra = f", diff={diff_val:.1f}" if diff_val is not None else ""
		print(f"  LOW {size_l:>9} {li.path.name:<40} -> HIGH {size_h:>9} {hi.path.name:<40} (hamming={dist}{extra})")

	if args.csv:
		csv_path = Path(args.csv)
		with csv_path.open("w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(["low_name", "low_w", "low_h", "high_name", "high_w", "high_h", "hamming", "diff"])  # header
			for li, hi, dist, diff_val in matches:
				w.writerow([li.path.name, li.width, li.height, hi.path.name, hi.width, hi.height, dist, f"{diff_val:.3f}" if diff_val is not None else ""])    
		print(f"[Write] CSV report: {csv_path}")

	print(f"[Summary] Matches: {len(matches)} (hash threshold {args.hash_threshold}, diff verify: {args.verify_diff})")


if __name__ == "__main__":
	main()
