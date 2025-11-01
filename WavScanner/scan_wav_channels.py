#!/usr/bin/env python3
r"""
Scan a folder of WAV files and list those that are multi‑channel audio (> stereo by default).

- Default threshold is >2 channels (i.e., 3+)
- Works with standard RIFF/WAVE and RF64 files
- Only reads headers; does not decode audio data

Usage examples:
  python scan_wav_channels.py .
    python scan_wav_channels.py C:\\Audio\\Samples -r --json
  python scan_wav_channels.py D:\\sessions --min-channels 4 --csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import io
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

RIFF = b"RIFF"
RF64 = b"RF64"
WAVE = b"WAVE"
FMT  = b"fmt "

class WavHeaderError(Exception):
    pass


def _read_exact(f: io.BufferedReader, n: int) -> bytes:
    b = f.read(n)
    if b is None or len(b) < n:
        raise WavHeaderError("Unexpected end of file while reading header")
    return b


def _parse_uint16_le(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def _parse_uint32_le(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def read_wav_channels(file_path: Path) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Return (channels, info) by reading only the RIFF/WAVE header.
    If header can't be read, channels is None and info['error'] includes reason.

    info fields (best-effort): format_tag, sample_rate, bits_per_sample, container
    """
    info: Dict[str, Any] = {
        "path": str(file_path),
        "container": None,
        "format_tag": None,
        "sample_rate": None,
        "bits_per_sample": None,
        "error": None,
    }
    try:
        with file_path.open("rb") as f:
            riff = _read_exact(f, 4)
            size = _parse_uint32_le(_read_exact(f, 4))  # total RIFF chunk size (may be 0xFFFFFFFF in RF64)
            wave = _read_exact(f, 4)
            if riff not in (RIFF, RF64) or wave != WAVE:
                raise WavHeaderError("Not a RIFF/WAVE file")
            info["container"] = "RF64" if riff == RF64 else "RIFF"

            # Iterate chunks until we find 'fmt '
            # Each chunk: 4-byte id, 4-byte size, then data[0:size], padded to even size
            while True:
                chdr = f.read(8)
                if not chdr or len(chdr) < 8:
                    break  # end of file without fmt
                cid = chdr[0:4]
                csize = _parse_uint32_le(chdr[4:8])

                if cid == FMT:
                    data = _read_exact(f, csize)
                    # fmt chunk should be at least 16 bytes
                    if csize < 4:
                        raise WavHeaderError("fmt chunk too small")
                    # wFormatTag (2), nChannels (2), nSamplesPerSec (4), nAvgBytesPerSec (4), nBlockAlign (2), wBitsPerSample (2)
                    fmt_tag = _parse_uint16_le(data[0:2]) if len(data) >= 2 else None
                    channels = _parse_uint16_le(data[2:4]) if len(data) >= 4 else None
                    samplerate = _parse_uint32_le(data[4:8]) if len(data) >= 8 else None
                    bits_per_sample = _parse_uint16_le(data[14:16]) if len(data) >= 16 else None
                    info.update({
                        "format_tag": fmt_tag,
                        "sample_rate": samplerate,
                        "bits_per_sample": bits_per_sample,
                    })
                    return channels, info
                else:
                    # skip this chunk (plus pad byte if size is odd)
                    to_skip = csize + (csize & 1)
                    # Use seek where possible to avoid reading into memory
                    f.seek(to_skip, os.SEEK_CUR)
            raise WavHeaderError("No fmt chunk found")
    except Exception as e:
        info["error"] = str(e)
        return None, info


def iter_wav_files(paths: Iterable[Path], recursive: bool = False) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".wav":
            yield p
        elif p.is_dir():
            if recursive:
                yield from (fp for fp in p.rglob("*.wav") if fp.is_file())
            else:
                yield from (fp for fp in p.glob("*.wav") if fp.is_file())


def scan(paths: List[Path], recursive: bool, min_channels: int, include_all: bool) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for wav in iter_wav_files(paths, recursive=recursive):
        channels, info = read_wav_channels(wav)
        row = {
            "path": str(wav),
            "channels": channels,
            "sample_rate": info.get("sample_rate"),
            "bits_per_sample": info.get("bits_per_sample"),
            "format_tag": info.get("format_tag"),
            "container": info.get("container"),
            "error": info.get("error"),
        }
        if include_all or (channels is not None and channels >= min_channels):
            results.append(row)
    return results


def print_plain(results: List[Dict[str, Any]], min_channels: int, include_all: bool) -> None:
    if not results:
        if include_all:
            print("No WAV files found.")
        else:
            print(f"No WAV files with >= {min_channels} channels found.")
        return

    # Determine column widths
    headers = ["Channels", "SampleRate", "Bits", "FmtTag", "Container", "Path"]
    rows = []
    for r in results:
        ch = r["channels"] if r["channels"] is not None else "?"
        sr = r["sample_rate"] if r["sample_rate"] is not None else "?"
        bp = r["bits_per_sample"] if r["bits_per_sample"] is not None else "?"
        ft = r["format_tag"] if r["format_tag"] is not None else "?"
        ct = r["container"] if r["container"] is not None else "?"
        if r.get("error"):
            rows.append(["-", "-", "-", "-", ct, f"{r['path']} (ERROR: {r['error']})"])
        else:
            rows.append([str(ch), str(sr), str(bp), str(ft), str(ct), r["path"]])

    # Calculate widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))

    # Print rows
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def write_csv(results: List[Dict[str, Any]], out: io.TextIOBase) -> None:
    fieldnames = ["path", "channels", "sample_rate", "bits_per_sample", "format_tag", "container", "error"]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for r in results:
        w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scan WAV files and list those with >= N channels (default: 3)")
    ap.add_argument("paths", nargs="*", default=["."], help="File or folder paths to scan (default: current directory)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--min-channels", type=int, default=3, help="Minimum channel count to include (default: 3)")
    outfmt = ap.add_mutually_exclusive_group()
    outfmt.add_argument("--json", dest="as_json", action="store_true", help="Output JSON")
    outfmt.add_argument("--csv", dest="as_csv", action="store_true", help="Output CSV")
    ap.add_argument("--print-all", action="store_true", help="Print all WAVs regardless of channel count (default: only >= min-channels)")

    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.paths]
    results = scan(paths, recursive=args.recursive, min_channels=args.min_channels, include_all=args.print_all)

    if args.as_json:
        print(json.dumps(results, indent=2))
    elif args.as_csv:
        write_csv(results, out=sys.stdout)
    else:
        print_plain(results, min_channels=args.min_channels, include_all=args.print_all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
