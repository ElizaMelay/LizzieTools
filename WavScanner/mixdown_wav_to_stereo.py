#!/usr/bin/env python3
"""
Downmix multi-channel WAV files to stereo WAVs.

Inputs:
- JSON produced by scan_wav_channels.py (via --from-json or --from-stdin)
- Or direct file/folder paths (optionally -r to recurse)

Downmix heuristics (common layouts, ITU-ish weights):
- 1.0 (mono): duplicate to L/R
- 2.0 (stereo): pass-through
- 3.0 (L, R, C): L += 0.707*C, R += 0.707*C
- 4.0 (L, R, Ls, Rs): L += 0.707*Ls, R += 0.707*Rs
- 5.0 (L, R, C, Ls, Rs): L += 0.707*C + 0.707*Ls, R += 0.707*C + 0.707*Rs
- 5.1 (L, R, C, LFE, Ls, Rs): L += 0.707*C + 0.5*LFE + 0.707*Ls, R += 0.707*C + 0.5*LFE + 0.707*Rs
- 7.1 (L, R, C, LFE, Ls, Rs, Lrs, Rrs): L += 0.707*C + 0.5*LFE + 0.5*(Ls+Lrs), R += 0.707*C + 0.5*LFE + 0.5*(Rs+Rrs)

If channel order is unknown, falls back to averaging surrounds where reasonable; otherwise averages all channels equally as a last resort.

By default outputs 16-bit PCM stereo, preserving sample rate. Can normalize to avoid clipping.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf

# ---------- Input discovery ----------

def _load_json_items(path: Optional[Path], from_stdin: bool) -> List[Dict[str, Any]]:
    """Load JSON array of items and return list with at least {'path', 'channels'?} per item."""
    data = None
    if from_stdin:
        text = os.sys.stdin.read()
        data = json.loads(text)
    elif path is not None:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        return []

    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'path' in item:
                p = Path(str(item['path']))
                if p.suffix.lower() == '.wav':
                    items.append({'path': str(p), 'channels': item.get('channels')})
            elif isinstance(item, str):
                p = Path(item)
                if p.suffix.lower() == '.wav':
                    items.append({'path': str(p)})
    return items


def _iter_paths(inputs: List[Path], recursive: bool) -> Iterable[Path]:
    for p in inputs:
        if p.is_file() and p.suffix.lower() == '.wav':
            yield p
        elif p.is_dir():
            if recursive:
                yield from (fp for fp in p.rglob('*.wav') if fp.is_file())
            else:
                yield from (fp for fp in p.glob('*.wav') if fp.is_file())

# ---------- Downmix core ----------

SQRT1_2 = 1 / np.sqrt(2.0)  # ~0.7071

def downmix_block(block: np.ndarray) -> np.ndarray:
    """
    Downmix a block of shape (frames, channels) -> (frames, 2) as float32.
    Uses channel-order heuristics described in the module docstring.
    """
    if block.ndim != 2:
        raise ValueError('Audio block must be 2D (frames, channels)')
    nframes, ch = block.shape
    if ch == 1:
        stereo = np.repeat(block, 2, axis=1)
        return stereo.astype(np.float32, copy=False)
    if ch == 2:
        return block.astype(np.float32, copy=False)

    # Prepare L and R
    L = np.zeros((nframes,), dtype=np.float32)
    R = np.zeros((nframes,), dtype=np.float32)

    # Copy base L/R if present
    L += block[:, 0].astype(np.float32)
    R += block[:, 1].astype(np.float32)

    if ch == 3:
        # L, R, C
        C = block[:, 2].astype(np.float32)
        L += SQRT1_2 * C
        R += SQRT1_2 * C
    elif ch == 4:
        # L, R, Ls, Rs (quad)
        Ls = block[:, 2].astype(np.float32)
        Rs = block[:, 3].astype(np.float32)
        L += SQRT1_2 * Ls
        R += SQRT1_2 * Rs
    elif ch == 5:
        # L, R, C, Ls, Rs
        C = block[:, 2].astype(np.float32)
        Ls = block[:, 3].astype(np.float32)
        Rs = block[:, 4].astype(np.float32)
        L += SQRT1_2 * (C + Ls)
        R += SQRT1_2 * (C + Rs)
    elif ch == 6:
        # L, R, C, LFE, Ls, Rs (5.1)
        C = block[:, 2].astype(np.float32)
        LFE = block[:, 3].astype(np.float32)
        Ls = block[:, 4].astype(np.float32)
        Rs = block[:, 5].astype(np.float32)
        L += SQRT1_2 * C + 0.5 * LFE + SQRT1_2 * Ls
        R += SQRT1_2 * C + 0.5 * LFE + SQRT1_2 * Rs
    elif ch == 7:
        # Heuristic: L, R, C, LFE, Ls, Rs, Cb? -> treat 7th as back-center into both
        C = block[:, 2].astype(np.float32)
        LFE = block[:, 3].astype(np.float32)
        Ls = block[:, 4].astype(np.float32)
        Rs = block[:, 5].astype(np.float32)
        Cb = block[:, 6].astype(np.float32)
        L += SQRT1_2 * C + 0.5 * LFE + 0.5 * (Ls + Cb)
        R += SQRT1_2 * C + 0.5 * LFE + 0.5 * (Rs + Cb)
    elif ch >= 8:
        # Assume 7.1 base in first 8: L,R,C,LFE,Ls,Rs,Lrs,Rrs; extra channels averaged quietly
        C = block[:, 2].astype(np.float32)
        LFE = block[:, 3].astype(np.float32)
        Ls = block[:, 4].astype(np.float32)
        Rs = block[:, 5].astype(np.float32)
        Lrs = block[:, 6].astype(np.float32)
        Rrs = block[:, 7].astype(np.float32)
        L += SQRT1_2 * C + 0.5 * LFE + 0.5 * (Ls + Lrs)
        R += SQRT1_2 * C + 0.5 * LFE + 0.5 * (Rs + Rrs)
        if ch > 8:
            extra = block[:, 8:].astype(np.float32)
            if extra.size > 0:
                avg = np.mean(extra, axis=1)
                L += 0.25 * avg
                R += 0.25 * avg
    else:
        # Fallback: average all channels into both
        avg = np.mean(block.astype(np.float32), axis=1)
        L += avg
        R += avg

    return np.stack([L, R], axis=1)

# ---------- Processing ----------

def mixdown_file(in_path: Path, out_dir: Optional[Path], suffix: str, overwrite: bool,
                 subtype: str, normalize: bool, chunk_size: int) -> Dict[str, Any]:
    out_dir = out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}{suffix}.wav"
    if out_path.exists() and not overwrite:
        return {"path": str(in_path), "out": str(out_path), "skipped": True, "reason": "exists"}

    with sf.SoundFile(str(in_path), mode='r') as sfi:
        sr = sfi.samplerate
        ch = sfi.channels
        if ch == 2:
            # Pass through stereo copy
            mode = 'w'
            with sf.SoundFile(str(out_path), mode=mode, samplerate=sr, channels=2, subtype=subtype) as sfo:
                while True:
                    data = sfi.read(frames=chunk_size, dtype='float32', always_2d=True)
                    if data.size == 0:
                        break
                    out = data  # already stereo float32
                    if normalize:
                        peak = np.max(np.abs(out))
                        if peak > 1.0 and peak > 0:
                            out = out / peak
                    sfo.write(out)
            return {"path": str(in_path), "out": str(out_path), "sr": sr, "in_channels": ch, "out_channels": 2}

        mode = 'w'
        with sf.SoundFile(str(out_path), mode=mode, samplerate=sr, channels=2, subtype=subtype) as sfo:
            while True:
                data = sfi.read(frames=chunk_size, dtype='float32', always_2d=True)
                if data.size == 0:
                    break
                out = downmix_block(data)
                if normalize:
                    peak = np.max(np.abs(out))
                    if peak > 1.0 and peak > 0:
                        out = out / peak
                sfo.write(out)

    return {"path": str(in_path), "out": str(out_path), "sr": sr, "in_channels": ch, "out_channels": 2}

# ---------- CLI ----------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Downmix multi-channel WAVs to stereo WAVs.')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--from-json', type=Path, help='Read input list from JSON file (output of scan_wav_channels.py).')
    src.add_argument('--from-stdin', action='store_true', help='Read JSON from stdin.')
    ap.add_argument('paths', nargs='*', help='WAV files or directories to process (ignored if --from-json/--from-stdin is used).')
    ap.add_argument('-r', '--recursive', action='store_true', help='Recurse into directories when using paths.')
    ap.add_argument('--only-multi', action='store_true', help='Only process files with channels >= 3 (when reading JSON with channel info).')

    ap.add_argument('--out-dir', type=Path, help='Output directory (default: alongside source files).')
    ap.add_argument('--suffix', default='_stereo', help='Suffix to append to output filename (default: _stereo).')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    ap.add_argument('--subtype', default='PCM_16', help='Output subtype: PCM_16, PCM_24, PCM_32, FLOAT, etc. (default: PCM_16).')
    ap.add_argument('--normalize', dest='normalize', action='store_true', default=True, help='Normalize to avoid clipping (default: on).')
    ap.add_argument('--no-normalize', dest='normalize', action='store_false', help='Disable normalization.')
    ap.add_argument('--chunk-size', type=int, default=262144, help='Frames per processing block (default: 262144).')

    args = ap.parse_args(argv)

    inputs: List[Path]
    files: List[Path] = []
    if args.from_json or args.from_stdin:
        items = _load_json_items(args.from_json, args.from_stdin)
        for it in items:
            p = Path(it['path'])
            if p.suffix.lower() != '.wav':
                continue
            if args.only_multi:
                ch = it.get('channels')
                if isinstance(ch, int):
                    if ch >= 3:
                        files.append(p)
                else:
                    # Fallback: probe header when channel info missing
                    try:
                        with sf.SoundFile(str(p), mode='r') as sfi:
                            if sfi.channels >= 3:
                                files.append(p)
                    except Exception:
                        pass
            else:
                files.append(p)
    else:
        inputs = [Path(p) for p in args.paths]
        files = list(_iter_paths(inputs, recursive=args.recursive))

    if not files:
        print('No input WAV files found.')
        return 0

    results: List[Dict[str, Any]] = []
    for i, fpath in enumerate(files, start=1):
        try:
            res = mixdown_file(
                in_path=fpath,
                out_dir=args.out_dir,
                suffix=args.suffix,
                overwrite=args.overwrite,
                subtype=args.subtype,
                normalize=args.normalize,
                chunk_size=args.chunk_size,
            )
        except Exception as e:
            res = {"path": str(fpath), "error": str(e)}
        results.append(res)

    # Print a compact report
    ok = sum(1 for r in results if 'out' in r and not r.get('skipped'))
    skipped = sum(1 for r in results if r.get('skipped'))
    failed = sum(1 for r in results if r.get('error'))
    print(f"Processed: {ok}, Skipped: {skipped}, Failed: {failed}")
    for r in results:
        if r.get('error'):
            print(f"ERROR: {r['path']}: {r['error']}")
        elif r.get('skipped'):
            print(f"SKIP: {r['path']} -> {r['out']} (exists)")
        else:
            print(f"OK: {r['path']} -> {r['out']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
