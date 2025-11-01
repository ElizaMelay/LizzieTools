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

By default preserves the original file's subtype (bit depth/format) and sample rate. Can normalize to avoid clipping.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
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

# Channel mask bits (WAVE_FORMAT_EXTENSIBLE / KSAUDIO_SPEAKER_*)
CHAN_BITS = [
    (0x00000001, 'FL'),
    (0x00000002, 'FR'),
    (0x00000004, 'FC'),
    (0x00000008, 'LFE'),
    (0x00000010, 'BL'),
    (0x00000020, 'BR'),
    (0x00000040, 'FLC'),
    (0x00000080, 'FRC'),
    (0x00000100, 'BC'),
    (0x00000200, 'SL'),
    (0x00000400, 'SR'),
    (0x00000800, 'TC'),
    (0x00001000, 'TFL'),
    (0x00002000, 'TFC'),
    (0x00004000, 'TFR'),
    (0x00008000, 'TBL'),
    (0x00010000, 'TBC'),
    (0x00020000, 'TBR'),
    (0x00040000, 'TSL'),
    (0x00080000, 'TSR'),
    (0x00100000, 'BLC'),
    (0x00200000, 'BRC'),
]


def parse_wav_channel_mask(path: Path) -> Optional[int]:
    """Return WAVEFORMATEXTENSIBLE dwChannelMask if present, else None."""
    try:
        with path.open('rb') as f:
            if f.read(4) not in (b'RIFF', b'RF64'):
                return None
            _ = f.read(4)  # size
            if f.read(4) != b'WAVE':
                return None
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    return None
                cid = hdr[:4]
                csz = int.from_bytes(hdr[4:], 'little')
                if cid == b'fmt ':
                    data = f.read(csz)
                    if len(data) < 18:
                        return None
                    fmt_tag = int.from_bytes(data[0:2], 'little')
                    if fmt_tag == 0xFFFE and csz >= 40:  # WAVE_FORMAT_EXTENSIBLE
                        # structure: WAVEFORMATEX (16) + wValidBitsPerSample(2) + dwChannelMask(4) + SubFormat(16)
                        mask = int.from_bytes(data[20:24], 'little')
                        return mask
                    return None
                # skip chunk + pad
                f.seek(csz + (csz & 1), os.SEEK_CUR)
    except Exception:
        return None


def make_mask_transform(mask: int, ch: int):
    """Return a transform(block)->stereo using channel mask order if consistent with ch, else None."""
    if mask is None:
        return None
    # Build ordered channel roles by standard mask bit order
    roles = [name for (bit, name) in CHAN_BITS if (mask & bit)]
    if len(roles) != ch:
        # Inconsistent; cannot trust mask
        return None

    # Map role weights to L/R
    def role_weights(role: str) -> Tuple[float, float]:
        if role == 'FL':
            return 1.0, 0.0
        if role == 'FR':
            return 0.0, 1.0
        if role == 'FC' or role == 'TFC' or role == 'TC':
            return SQRT1_2, SQRT1_2
        if role == 'LFE':
            return 0.5, 0.5
        if role in ('BL', 'SL', 'TBL', 'TSL', 'BLC'):
            return SQRT1_2, 0.0
        if role in ('BR', 'SR', 'TBR', 'TSR', 'BRC'):
            return 0.0, SQRT1_2
        if role == 'BC':
            return 0.5, 0.5
        if role == 'FLC' or role == 'TFL':
            return 0.5, 0.0
        if role == 'FRC' or role == 'TFR':
            return 0.0, 0.5
        # default gentle spread
        return 0.5, 0.5

    # Precompute indices and weights
    idxs = list(range(ch))
    lw = np.array([role_weights(r)[0] for r in roles], dtype=np.float32)
    rw = np.array([role_weights(r)[1] for r in roles], dtype=np.float32)

    def transform(block: np.ndarray) -> np.ndarray:
        if block.ndim != 2 or block.shape[1] != ch:
            raise ValueError('Unexpected block shape for mask-based transform')
        b = block.astype(np.float32, copy=False)
        L = np.sum(b * lw[None, :], axis=1)
        R = np.sum(b * rw[None, :], axis=1)
        return np.stack([L, R], axis=1)

    return transform

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

def _unique_sibling_path(base_path: Path) -> Path:
    """Return a non-colliding path by appending (n) before the suffix if needed."""
    if not base_path.exists():
        return base_path
    i = 1
    while True:
        candidate = base_path.with_name(f"{base_path.stem} ({i}){base_path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _compute_peak(path: Path, chunk_size: int, transform) -> float:
    """Compute a global peak of the transformed signal by scanning once."""
    peak = 0.0
    with sf.SoundFile(str(path), mode='r') as sfi:
        while True:
            data = sfi.read(frames=chunk_size, dtype='float32', always_2d=True)
            if data.size == 0:
                break
            out = transform(data)
            if out.size:
                p = float(np.max(np.abs(out)))
                if p > peak:
                    peak = p
    return peak


def _write_streamed(sfi: sf.SoundFile, out_path: Path, subtype: str, chunk_size: int,
                    transform, scale: float = 1.0) -> None:
    sr = sfi.samplerate
    with sf.SoundFile(str(out_path), mode='w', samplerate=sr, channels=2, subtype=subtype) as sfo:
        while True:
            data = sfi.read(frames=chunk_size, dtype='float32', always_2d=True)
            if data.size == 0:
                break
            out = transform(data)
            if scale != 1.0:
                out = out * scale
            sfo.write(out)


def _ffmpeg_path() -> Optional[str]:
    exe = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    return exe


def _copy_metadata_with_ffmpeg(dst_audio: Path, src_meta: Path) -> Optional[str]:
    """Copy metadata from src_meta to dst_audio in place using ffmpeg remux. Returns error string if failed, else None."""
    ff = _ffmpeg_path()
    if not ff:
        return 'ffmpeg not found; metadata not preserved'
    tmp_out = dst_audio.with_suffix('.tmp.wav')
    cmd = [ff, '-v', 'error', '-y', '-i', str(dst_audio), '-i', str(src_meta), '-map', '0:a', '-map_metadata', '1', '-c', 'copy', str(tmp_out)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode != 0 or (not tmp_out.exists()):
            return f'ffmpeg metadata copy failed: {r.stderr.decode(errors="ignore").strip()}'
        # Replace original
        dst_audio.unlink(missing_ok=True)
        tmp_out.replace(dst_audio)
        return None
    except Exception as e:
        return str(e)


def mixdown_file(in_path: Path, out_dir: Optional[Path], suffix: str, overwrite: bool,
                 subtype: Optional[str], normalize: bool, chunk_size: int,
                 preserve_originals_subdir: Optional[str] = None) -> Dict[str, Any]:
    """Downmix a single file.

    If preserve_originals_subdir is provided and the file has >=3 channels, the original file is moved to a
    sibling subfolder (creating it if needed), and the stereo mix is written back to the original file path.
    Otherwise, the stereo mix is written to out_dir/stem+suffix.wav.
    """
    # First, peek channel count safely
    try:
        with sf.SoundFile(str(in_path), mode='r') as peek:
            sr = peek.samplerate
            ch = peek.channels
    except Exception as e:
        return {"path": str(in_path), "error": str(e)}

    # In preserve-originals mode, only act on multi-channel files
    if preserve_originals_subdir and ch >= 3:
        backup_dir = in_path.parent / preserve_originals_subdir
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = _unique_sibling_path(backup_dir / in_path.name)

        # Move the original to backup, then read from backup and write to original path
        shutil.move(str(in_path), str(backup_path))

        # If a stereo already exists at original path (rare), enforce overwrite behavior
        out_path = in_path
        if out_path.exists() and not overwrite:
            return {"path": str(in_path), "out": str(out_path), "skipped": True, "reason": "exists"}

        with sf.SoundFile(str(backup_path), mode='r') as sfi:
            # Determine output subtype: preserve unless explicitly set
            out_subtype = subtype or sfi.subtype or 'PCM_16'
            # Choose transform
            # Prefer mask-based transform if available
            mask = parse_wav_channel_mask(backup_path)
            transform = (lambda x: x) if sfi.channels == 2 else (make_mask_transform(mask, sfi.channels) or downmix_block)
            # Normalize using global peak if requested
            if normalize:
                peak = _compute_peak(backup_path, chunk_size=chunk_size, transform=transform)
                scale = (1.0 / peak) if peak > 1.0 else 1.0
                _ = sfi.seek(0)
                _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=scale)
            else:
                _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=1.0)

        meta_err = _copy_metadata_with_ffmpeg(out_path, backup_path)
        result = {"path": str(in_path), "out": str(out_path), "backup": str(backup_path), "sr": sr, "in_channels": ch, "out_channels": 2}
        if meta_err:
            result["metadata_warning"] = meta_err
        return result

    # Not preserving originals: write to side-by-side output
    out_dir = out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}{suffix}.wav"
    if out_path.exists() and not overwrite:
        return {"path": str(in_path), "out": str(out_path), "skipped": True, "reason": "exists"}

    with sf.SoundFile(str(in_path), mode='r') as sfi:
        out_subtype = subtype or sfi.subtype or 'PCM_16'
        mask = parse_wav_channel_mask(in_path)
        transform = (lambda x: x) if ch == 2 else (make_mask_transform(mask, ch) or downmix_block)
        if normalize:
            peak = _compute_peak(in_path, chunk_size=chunk_size, transform=transform)
            scale = (1.0 / peak) if peak > 1.0 else 1.0
            _ = sfi.seek(0)
        else:
            scale = 1.0
        _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=scale)
        meta_err = _copy_metadata_with_ffmpeg(out_path, in_path)
        result = {"path": str(in_path), "out": str(out_path), "sr": sfi.samplerate, "in_channels": ch, "out_channels": 2}
        if meta_err:
            result["metadata_warning"] = meta_err
        return result

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
    ap.add_argument('--preserve-originals', nargs='?', const='multitrack_originals', default=None,
                    help="Move original multi-channel files into this subfolder near the source, then write the stereo mix back to the original file path. If used without a value, defaults to 'multitrack_originals'. When set, ignores --out-dir and --suffix for affected files.")
    ap.add_argument('--subtype', default=None, help='Output subtype (e.g., PCM_16, PCM_24, PCM_32, FLOAT). Default: preserve original subtype.')
    ap.add_argument('--normalize', dest='normalize', action='store_true', default=True, help='Normalize to avoid clipping using global peak (default: on).')
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
                preserve_originals_subdir=args.preserve_originals,
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
