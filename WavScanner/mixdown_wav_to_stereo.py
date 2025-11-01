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
    (0x00000001, 'FL'),  # Front Left
    (0x00000002, 'FR'),  # Front Right
    (0x00000004, 'FC'),  # Front Center
    (0x00000008, 'LFE'), # Low-Frequency Effects
    (0x00000010, 'BL'),  # Back Left
    (0x00000020, 'BR'),  # Back Right
    (0x00000040, 'FLC'), # Front Left of Center
    (0x00000080, 'FRC'), # Front Right of Center
    (0x00000100, 'BC'),  # Back Center
    (0x00000200, 'SL'),  # Side Left
    (0x00000400, 'SR'),  # Side Right
    (0x00000800, 'TC'),  # Top Center
    (0x00001000, 'TFL'), # Top Front Left
    (0x00002000, 'TFC'), # Top Front Center
    (0x00004000, 'TFR'), # Top Front Right
    (0x00008000, 'TBL'), # Top Back Left
    (0x00010000, 'TBC'), # Top Back Center
    (0x00020000, 'TBR'), # Top Back Right
    (0x00040000, 'TSL'), # Top Side Left
    (0x00080000, 'TSR'), # Top Side Right
    (0x00100000, 'BLC'), # Bottom Left of Center
    (0x00200000, 'BRC'), # Bottom Right of Center
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


FRIENDLY_ROLE = {
    'FL': 'Front Left',
    'FR': 'Front Right',
    'FC': 'Front Center',
    'LFE': 'LFE (subwoofer)',
    'BL': 'Back Left',
    'BR': 'Back Right',
    'FLC': 'Front Left of Center',
    'FRC': 'Front Right of Center',
    'BC': 'Back Center',
    'SL': 'Side Left',
    'SR': 'Side Right',
    'TC': 'Top Center',
    'TFL': 'Top Front Left',
    'TFC': 'Top Front Center',
    'TFR': 'Top Front Right',
    'TBL': 'Top Back Left',
    'TBC': 'Top Back Center',
    'TBR': 'Top Back Right',
    'TSL': 'Top Side Left',
    'TSR': 'Top Side Right',
    'BLC': 'Bottom Left of Center',
    'BRC': 'Bottom Right of Center',
}

# Friendly names for Ambisonics (first-order B-Format)
FRIENDLY_AMBI_ROLE = {
    'W': 'Omni (W)',
    'X': 'Front-Back (X)',
    'Y': 'Left-Right (Y)',
    'Z': 'Up-Down (Z)',
}


def roles_from_mask(mask: Optional[int], ch: int) -> Optional[List[str]]:
    if mask is None:
        return None
    roles = [name for (bit, name) in CHAN_BITS if (mask & bit)]
    if len(roles) != ch:
        return None
    return roles


def friendly_mask_list(mask: Optional[int], ch: int) -> Optional[str]:
    roles = roles_from_mask(mask, ch)
    if not roles:
        return None
    return ", ".join(f"{FRIENDLY_ROLE.get(r, r)} ({r})" for r in roles)


def describe_downmix_from_roles(roles: List[str]) -> Tuple[str, str]:
    """Return (left_desc, right_desc) describing contributions with dB weights."""
    def role_weight_db_to_side(role: str, side: str) -> Optional[float]:
        # Convert weights used in make_mask_transform to dB text
        if role == 'FL' and side == 'L':
            return 0.0
        if role == 'FR' and side == 'R':
            return 0.0
        if role in ('FC', 'TFC', 'TC'):
            return -3.0
        if role == 'LFE':
            return -6.0
        if role in ('BL', 'SL', 'TBL', 'TSL', 'BLC') and side == 'L':
            return -3.0
        if role in ('BR', 'SR', 'TBR', 'TSR', 'BRC') and side == 'R':
            return -3.0
        if role == 'FLC' and side == 'L':
            return -6.0
        if role == 'FRC' and side == 'R':
            return -6.0
        if role == 'BC':
            return -6.0
        return None

    left_parts = []
    right_parts = []
    for r in roles:
        dbL = role_weight_db_to_side(r, 'L')
        if dbL is not None:
            left_parts.append(f"{FRIENDLY_ROLE.get(r, r)} ({r}, {dbL:+.1f} dB)")
        dbR = role_weight_db_to_side(r, 'R')
        if dbR is not None:
            right_parts.append(f"{FRIENDLY_ROLE.get(r, r)} ({r}, {dbR:+.1f} dB)")
    return ", ".join(left_parts) or "(none)", ", ".join(right_parts) or "(none)"


def make_roles_transform(roles: Optional[List[str]]) -> Optional[Any]:
    """Build a transform based on a roles list (e.g., ['FL','FR','FC','LFE']). Returns None if roles is invalid."""
    if not roles:
        return None
    ch = len(roles)
    # Reuse same weight mapping logic as mask-based transform
    def role_weights(role: str) -> Tuple[float, float]:
        if role == 'FL':
            return 1.0, 0.0
        if role == 'FR':
            return 0.0, 1.0
        if role in ('FC', 'TFC', 'TC'):
            return SQRT1_2, SQRT1_2
        if role == 'LFE':
            return 0.5, 0.5
        if role in ('BL', 'SL', 'TBL', 'TSL', 'BLC'):
            return SQRT1_2, 0.0
        if role in ('BR', 'SR', 'TBR', 'TSR', 'BRC'):
            return 0.0, SQRT1_2
        if role == 'BC':
            return 0.5, 0.5
        if role == 'FLC':
            return 0.5, 0.0
        if role == 'FRC':
            return 0.0, 0.5
        return 0.5, 0.5

    lw = np.array([role_weights(r)[0] for r in roles], dtype=np.float32)
    rw = np.array([role_weights(r)[1] for r in roles], dtype=np.float32)

    def transform(block: np.ndarray) -> np.ndarray:
        if block.ndim != 2 or block.shape[1] != ch:
            raise ValueError('Unexpected block shape for roles-based transform')
        b = block.astype(np.float32, copy=False)
        L = np.sum(b * lw[None, :], axis=1)
        R = np.sum(b * rw[None, :], axis=1)
        return np.stack([L, R], axis=1)

    return transform


def heuristic_roles_for_channels(ch: int) -> Optional[List[str]]:
    """Return a guessed roles list for common layouts, else None."""
    if ch == 1:
        return ['FC']
    if ch == 2:
        return ['FL', 'FR']
    if ch == 3:
        return ['FL', 'FR', 'FC']
    if ch == 4:
        # Assume L, R, Ls, Rs
        return ['FL', 'FR', 'SL', 'SR']
    if ch == 5:
        return ['FL', 'FR', 'FC', 'SL', 'SR']
    if ch == 6:
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
    if ch == 7:
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR', 'BC']
    if ch >= 8:
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR', 'BL', 'BR']
    return None


def _ffprobe_layout(path: Path) -> Optional[str]:
    ffprobe = _ffprobe_path()
    if not ffprobe:
        return None
    cmd = [ffprobe, '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channel_layout', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode != 0:
            return None
        layout = r.stdout.decode('utf-8', errors='ignore').strip()
        return layout or None
    except Exception:
        return None


def roles_from_ffprobe_layout(layout: Optional[str], ch: int) -> Optional[List[str]]:
    if not layout:
        return None
    l = layout.lower()
    # Common layouts
    if l in ('mono',):
        return ['FC'] if ch == 1 else None
    if l in ('stereo',):
        return ['FL', 'FR'] if ch == 2 else None
    if l in ('2.1', 'stereo+lfe') and ch == 3:
        return ['FL', 'FR', 'LFE']
    if l in ('3.0',) and ch == 3:
        return ['FL', 'FR', 'FC']
    if l in ('3f/lfe', '3.1') and ch == 4:
        return ['FL', 'FR', 'FC', 'LFE']
    if l in ('quad',) and ch == 4:
        return ['FL', 'FR', 'BL', 'BR']
    if l in ('4.0',) and ch == 4:
        return ['FL', 'FR', 'FC', 'BC']
    if l in ('5.0',) and ch == 5:
        return ['FL', 'FR', 'FC', 'SL', 'SR']
    if l in ('5.1',) and ch == 6:
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR']
    if l in ('7.1', '7.1(rear)') and ch == 8:
        return ['FL', 'FR', 'FC', 'LFE', 'SL', 'SR', 'BL', 'BR']
    if l in ('7.1(wide)',) and ch == 8:
        return ['FL', 'FR', 'FC', 'LFE', 'FLC', 'FRC', 'SL', 'SR']
    return None


def detect_ambisonics(path: Path, ch: int) -> Tuple[Optional[List[str]], Optional[str], Optional[Dict[str, Any]]]:
    """Detect first-order Ambisonics.
    Returns (roles, kind, info) where:
      - roles: channel roles if already B-format, else None
      - kind: 'FuMa' or 'AmbiX' for B-format, else None
      - info: {'format': 'A'|'B', 'evidence': [...], ...}
    """
    if ch < 4:
        return None, None, None
    tags = _ffprobe_tags(path) or {}
    name = path.name.lower()
    tag_values = [str(v).lower() for v in tags.values() if isinstance(v, (str, int, float))]
    hay = ' '.join([name] + tag_values)
    indicators: List[str] = []
    aformat: List[str] = []
    ambix_hits: List[str] = []
    fuma_hits: List[str] = []
    # filename indicators
    if 'ambisonic' in name:
        indicators.append("filename contains 'ambisonic'")
    if 'b-format' in name or 'bformat' in name:
        indicators.append("filename mentions 'B-Format'")
    if 'ambix' in name:
        indicators.append("filename contains 'ambix'")
        ambix_hits.append("filename contains 'ambix'")
    if 'fuma' in name:
        indicators.append("filename contains 'fuma'")
        fuma_hits.append("filename contains 'fuma'")
    if 'acn' in name or 'sn3d' in name:
        indicators.append("filename mentions 'ACN/SN3D'")
        ambix_hits.append("filename mentions 'ACN/SN3D'")
    if 'wxyz' in name:
        indicators.append("filename mentions 'WXYZ'")
        fuma_hits.append("filename mentions 'WXYZ'")
    if 'a format' in name or 'a-format' in name or 'aformat' in name:
        aformat.append("filename mentions 'A-Format'")
    # tag indicators
    for k, v in (tags or {}).items():
        try:
            vv = str(v).lower()
        except Exception:
            vv = ''
        if any(t in vv for t in ['ambisonic', 'b-format', 'bformat', 'ambix', 'fuma']):
            indicators.append(f"tag {k} contains '{vv}'")
        if any(t in vv for t in ['a format', 'a-format', 'aformat']):
            aformat.append(f"tag {k} contains '{vv}'")
        if any(t in vv for t in ['ambix', 'acn', 'sn3d']):
            ambix_hits.append(f"tag {k} contains '{vv}'")
        if any(t in vv for t in ['fuma', 'wxyz']):
            fuma_hits.append(f"tag {k} contains '{vv}'")
    is_ambi = bool(indicators or aformat)
    if not is_ambi:
        return None, None, None
    # If A-format is indicated, don't assume B-format roles
    if aformat:
        info = {
            'evidence': indicators + aformat,
            'tags_used': sorted(list(tags.keys())) if isinstance(tags, dict) else None,
            'format': 'A',
            'kind': None,
        }
        return None, None, info
    # Else assume B-format; decide FuMa vs AmbiX (ACN/SN3D) by keywords
    decider = 'default'
    kind = 'AmbiX'
    if ambix_hits and not fuma_hits:
        kind = 'AmbiX'
        decider = 'keyword-ambix'
    elif fuma_hits and not ambix_hits:
        kind = 'FuMa'
        decider = 'keyword-fuma'
    else:
        # ambiguous; default to AmbiX (caller may still bias/override)
        decider = 'ambiguous'
    roles = ['W', 'Y', 'Z', 'X'] if kind == 'AmbiX' else ['W', 'X', 'Y', 'Z']
    info = {
        'evidence': indicators,
        'tags_used': sorted(list(tags.keys())) if isinstance(tags, dict) else None,
        'format': 'B',
        'kind': kind,
        'decider': decider,
        'ambix_hits': ambix_hits,
        'fuma_hits': fuma_hits,
    }
    return roles, kind, info


def make_ambisonics_transform(roles: List[str], kind: str):
    """Simple first-order ambisonics stereo decode (horizontal Blumlein-like).
    Uses L = W + 0.7071*X + 0.7071*Y, R = W + 0.7071*X - 0.7071*Y.
    Z is ignored for horizontal stereo.
    """
    idx = {r: i for i, r in enumerate(roles)}
    if 'W' not in idx or 'X' not in idx or 'Y' not in idx:
        return None
    def transform(block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32, copy=False)
        W = b[:, idx['W']]
        X = b[:, idx['X']]
        Y = b[:, idx['Y']]
        L = W + SQRT1_2 * X + SQRT1_2 * Y
        R = W + SQRT1_2 * X - SQRT1_2 * Y
        return np.stack([L, R], axis=1)
    return transform


def describe_downmix_from_ambisonics(roles: List[str]) -> Tuple[str, str]:
    # L = W + -3 dB X + -3 dB Y; R = W + -3 dB X + -3 dB (-Y)
    left = []
    right = []
    if 'W' in roles:
        left.append(f"{FRIENDLY_AMBI_ROLE['W']} (+0.0 dB)")
        right.append(f"{FRIENDLY_AMBI_ROLE['W']} (+0.0 dB)")
    if 'X' in roles:
        left.append(f"{FRIENDLY_AMBI_ROLE['X']} (-3.0 dB)")
        right.append(f"{FRIENDLY_AMBI_ROLE['X']} (-3.0 dB)")
    if 'Y' in roles:
        left.append(f"{FRIENDLY_AMBI_ROLE['Y']} (-3.0 dB)")
        right.append(f"{FRIENDLY_AMBI_ROLE['Y']} (-3.0 dB, inverted to right)")
    return ', '.join(left), ', '.join(right)

def _compute_channel_rms(path: Path, frames_limit: int = 262144) -> Optional[List[float]]:
    """Compute per-channel RMS over up to frames_limit frames from the start of the file.
    Returns list of floats length=channels, or None on error.
    """
    try:
        with sf.SoundFile(str(path), mode='r') as sfi:
            ch = sfi.channels
            sumsqs = np.zeros((ch,), dtype=np.float64)
            count = 0
            remaining = frames_limit
            block = 65536
            while remaining > 0:
                n = min(block, remaining)
                data = sfi.read(frames=n, dtype='float32', always_2d=True)
                if data.size == 0:
                    break
                sumsqs += np.sum(data.astype(np.float32) ** 2.0, axis=0, dtype=np.float64)
                count += data.shape[0]
                remaining -= data.shape[0]
            if count == 0:
                return [0.0] * ch
            rms = np.sqrt(sumsqs / float(count))
            return [float(x) for x in rms]
    except Exception:
        return None

def _decide_ambix_fuma_by_energy(rms: List[float], min_diff_db: float = 3.0) -> Tuple[Optional[str], Dict[str, Any]]:
    """Given per-channel RMS for a 4-channel B-format file, decide AmbiX vs FuMa by
    finding the lowest-energy among channels 1..3 (X/Y/Z candidates).
    - If idx==2 -> AmbiX (Z at ch2)
    - If idx==3 -> FuMa (Z at ch3)
    - If idx==1 -> no confidence
    Requires that min is at least min_diff_db below both others.
    Returns (kind or None, info dict including confidence and indices).
    """
    info: Dict[str, Any] = {'method': 'energy-scan', 'rms': rms}
    if rms is None or len(rms) < 4:
        info['error'] = 'rms-unavailable'
        return None, info
    # Consider channels 1..3 (indexes)
    arr = np.array(rms, dtype=np.float64)
    subset = arr[1:4]
    min_idx_local = int(np.argmin(subset))  # 0..2
    min_idx = 1 + min_idx_local            # 1..3
    # Compute dB diffs vs other two
    min_val = subset[min_idx_local]
    others = np.delete(subset, min_idx_local)
    # avoid log of zero; add tiny eps
    eps = 1e-12
    db_min = 20.0 * np.log10(max(min_val, eps))
    db_others = 20.0 * np.log10(np.maximum(others, eps))
    diffs = db_others - db_min  # both should be >= min_diff_db to be confident
    confidence_db = float(np.min(diffs)) if diffs.size > 0 else 0.0
    info.update({'min_index': min_idx, 'confidence_db': confidence_db})
    if confidence_db < float(min_diff_db):
        # Not confidently lower
        return None, info
    if min_idx == 2:
        return 'AmbiX', info
    if min_idx == 3:
        return 'FuMa', info
    return None, info


def _load_matrix_from_json(path: Path) -> Optional[np.ndarray]:
    """Load a 4x4 matrix from a JSON file (list of 4 lists of 4 numbers)."""
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == 4 and all(isinstance(row, list) and len(row) == 4 for row in data):
            m = np.array(data, dtype=np.float32)
            return m
    except Exception:
        return None
    return None


def _aformat_preset_matrix(preset: Optional[str]) -> Optional[np.ndarray]:
    """Return a 4x4 A→B matrix for known presets. Currently supports 'generic'."""
    if not preset:
        return None
    p = preset.lower()
    if p == 'generic':
        # Assume channel order: [FL, FR, BL, BR] on a horizontal square.
        # SN3D-ish scaling; Z omitted.
        return np.array([
            [0.5,  0.5,  0.5,  0.5],  # W
            [0.5,  0.5, -0.5, -0.5],  # X (front-back)
            [0.5, -0.5,  0.5, -0.5],  # Y (left-right)
            [0.0,  0.0,  0.0,  0.0],  # Z
        ], dtype=np.float32)
    return None


def make_ambisonics_aformat_transform(matrix: np.ndarray, kind: str):
    """Convert A-format (4 capsules) to B-format via 4x4 matrix, then decode to stereo (FOA horizontal)."""
    if matrix.shape != (4, 4):
        raise ValueError('A→B matrix must be 4x4')
    def transform(block: np.ndarray) -> np.ndarray:
        if block.ndim != 2 or block.shape[1] != 4:
            raise ValueError('A-format block must be (frames, 4)')
        b = block.astype(np.float32, copy=False)
        B = b @ matrix.T  # frames x 4 -> W,X,Y,Z
        W = B[:, 0]
        X = B[:, 1]
        Y = B[:, 2]
        # Z = B[:, 3]  # unused for horizontal decode
        L = W + SQRT1_2 * X + SQRT1_2 * Y
        R = W + SQRT1_2 * X - SQRT1_2 * Y
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


def _ffmpeg_apply_metadata(dst_audio: Path, tags: Dict[str, Any]) -> Optional[str]:
    """Apply tags onto dst_audio by remuxing and setting -metadata key=value pairs. Returns error string or None."""
    ff = _ffmpeg_path()
    if not ff:
        return 'ffmpeg not found; metadata not preserved'
    tmp_out = dst_audio.with_suffix('.tmp2.wav')
    cmd = [ff, '-v', 'error', '-y', '-i', str(dst_audio), '-map', '0:a', '-map_metadata', '-1', '-c', 'copy']
    for k, v in (tags or {}).items():
        # Flatten values to strings
        if v is None:
            continue
        sval = str(v)
        # Avoid newlines that can confuse shells
        sval = sval.replace('\r', ' ').replace('\n', '  ')
        cmd.extend(['-metadata', f"{k}={sval}"])
    cmd.append(str(tmp_out))
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode != 0 or (not tmp_out.exists()):
            return f'ffmpeg metadata apply failed: {r.stderr.decode(errors="ignore").strip()}'
        dst_audio.unlink(missing_ok=True)
        tmp_out.replace(dst_audio)
        return None
    except Exception as e:
        return str(e)


def _ffprobe_path() -> Optional[str]:
    exe = shutil.which('ffprobe') or shutil.which('ffprobe.exe')
    return exe


def _ffprobe_tags(path: Path) -> Optional[Dict[str, Any]]:
    ffprobe = _ffprobe_path()
    if not ffprobe:
        return None
    cmd = [ffprobe, '-v', 'error', '-show_entries', 'format:stream', '-print_format', 'json', str(path)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout.decode('utf-8', errors='ignore'))
        tags: Dict[str, Any] = {}
        fmt = data.get('format', {})
        if isinstance(fmt, dict) and 'tags' in fmt and isinstance(fmt['tags'], dict):
            tags.update(fmt['tags'])
        for st in data.get('streams', []) or []:
            if isinstance(st, dict) and st.get('codec_type') == 'audio' and isinstance(st.get('tags'), dict):
                for k, v in st['tags'].items():
                    tags.setdefault(k, v)
        return tags
    except Exception:
        return None


BITS_PER_SUBTYPE = {
    'PCM_U8': 8,
    'PCM_S8': 8,
    'PCM_16': 16,
    'PCM_24': 24,
    'PCM_32': 32,
    'FLOAT': 32,
    'DOUBLE': 64,
}


def _bits_for_subtype(subtype: Optional[str]) -> Optional[int]:
    if not subtype:
        return None
    return BITS_PER_SUBTYPE.get(subtype)


def mixdown_file(in_path: Path, out_dir: Optional[Path], suffix: str, overwrite: bool,
                 subtype: Optional[str], normalize: bool, chunk_size: int,
                 preserve_originals_subdir: Optional[str] = None,
                 verbose: int = 0,
                 aformat_preset: Optional[str] = None,
                 aformat_matrix: Optional[Path] = None,
                 ambi_kind: str = 'auto',
                 ambi_default: str = 'ambix',
                 ambi_scan_frames: int = 262144,
                 ambi_scan_min_diff_db: float = 3.0) -> Dict[str, Any]:
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
            # Prefer Ambisonics first (bias)
            ambi_roles, ambi_kind, ambi_info = detect_ambisonics(backup_path, sfi.channels)
            detection_notes: Optional[Dict[str, Any]] = None
            layout = None
            mask = None
            roles = None
            transform = None
            transform_name = 'passthrough' if sfi.channels == 2 else None
            if ambi_info and ambi_info.get('format') == 'A':
                # A-format detected: attempt A→B using provided matrix/preset
                mat = None
                if aformat_matrix:
                    mat = _load_matrix_from_json(aformat_matrix)
                if mat is None and aformat_preset:
                    mat = _aformat_preset_matrix(aformat_preset)
                if mat is not None and sfi.channels == 4:
                    transform = make_ambisonics_aformat_transform(mat, 'FuMa')
                    roles = ['W', 'X', 'Y', 'Z']
                    transform_name = 'ambisonics-aformat+ab'
                    detection_notes = {'ambisonics': {'detected': True, 'format': 'A', 'kind': 'FuMa', 'info': ambi_info, 'aformat': {'preset': aformat_preset, 'matrix': str(aformat_matrix) if aformat_matrix else None}}}
                else:
                    # No matrix -> leave transform None to fall back
                    detection_notes = {'ambisonics': {'detected': True, 'format': 'A', 'kind': None, 'info': ambi_info, 'warning': 'A-format requires mic-specific A→B; provide --aformat-preset generic or --aformat-matrix'}}
            elif ambi_roles:
                # Allow override or default bias for ambiguous decisions, with optional energy scan
                chosen_kind = ambi_kind
                info_decider = (ambi_info or {}).get('decider') if isinstance(ambi_info, dict) else None
                detected_kind = ambi_kind if ambi_kind in ('fuma', 'ambix') else (ambi_kind if ambi_kind in ('FuMa','AmbiX') else None)
                # Normalize values
                ambi_kind_norm = (ambi_kind or 'auto').lower()
                if ambi_kind_norm in ('fuma', 'ambix'):
                    chosen_kind = 'AmbiX' if ambi_kind_norm == 'ambix' else 'FuMa'
                    reason = 'override'
                else:
                    # Try energy scan first when auto
                    kind_scan, scan_info = _decide_ambix_fuma_by_energy(_compute_channel_rms(backup_path) or [])
                    if kind_scan in ('AmbiX', 'FuMa'):
                        chosen_kind = kind_scan
                        reason = 'energy-scan'
                    elif info_decider == 'keyword-ambix':
                        chosen_kind = 'AmbiX'
                        reason = 'keyword'
                    elif info_decider == 'keyword-fuma':
                        chosen_kind = 'FuMa'
                        reason = 'keyword'
                    else:
                        # ambiguous/default bias
                        if (ambi_default or 'ambix').lower() == 'ambix':
                            chosen_kind = 'AmbiX'
                            reason = 'default-bias'
                        else:
                            chosen_kind = 'FuMa'
                            reason = 'default-bias'
                roles = ambi_roles
                tm = make_ambisonics_transform(roles, chosen_kind)
                transform = tm
                transform_name = f"ambisonics-{chosen_kind.lower()}"
                notes = {'ambisonics': {'detected': True, 'format': 'B', 'kind': chosen_kind, 'info': ambi_info, 'decision': reason}}
                # attach scan info if used
                try:
                    if 'scan_info' in locals() and isinstance(scan_info, dict):
                        notes['ambisonics']['scan'] = scan_info
                except Exception:
                    pass
                detection_notes = notes
            if transform is None:
                mask = parse_wav_channel_mask(backup_path)
                roles = roles_from_mask(mask, sfi.channels)
                if roles:
                    tm = make_roles_transform(roles)
                    transform = tm or downmix_block
                    transform_name = 'mask'
            if transform is None:
                layout = _ffprobe_layout(backup_path)
                roles = roles_from_ffprobe_layout(layout, sfi.channels)
                if roles:
                    tm = make_roles_transform(roles)
                    transform = tm or downmix_block
                    transform_name = 'ffprobe'
            if transform is None:
                roles = heuristic_roles_for_channels(sfi.channels)
                tm = make_roles_transform(roles) if roles else None
                transform = tm or downmix_block
                transform_name = 'heuristic'
            # Normalize using global peak if requested
            if normalize:
                peak = _compute_peak(backup_path, chunk_size=chunk_size, transform=transform)
                scale = (1.0 / peak) if peak > 1.0 else 1.0
                _ = sfi.seek(0)
                _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=scale)
            else:
                _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=1.0)

        # Metadata copy (post-write)
        meta_before = _ffprobe_tags(backup_path) if verbose >= 2 else None
        meta_err = _copy_metadata_with_ffmpeg(out_path, backup_path)
        # If simple copy missed keys, try explicit apply from source tags
        if meta_err is None and verbose >= 2:
            after_try = _ffprobe_tags(out_path)
            if isinstance(meta_before, dict) and isinstance(after_try, dict):
                missing = [k for k in meta_before.keys() if k not in after_try]
                if missing:
                    meta_err2 = _ffmpeg_apply_metadata(out_path, meta_before)
                    if meta_err2:
                        meta_err = meta_err2
        meta_after = _ffprobe_tags(out_path) if verbose >= 2 else None

        bits_in = _bits_for_subtype(sfi.subtype)
        bits_out = _bits_for_subtype(out_subtype)
        result: Dict[str, Any] = {
            "path": str(in_path),
            "out": str(out_path),
            "backup": str(backup_path),
            "sr": sr,
            "in_channels": ch,
            "out_channels": 2,
        }
        details: Dict[str, Any] = {
            "source": {
                "channels": ch,
                "samplerate": sr,
                "subtype": sfi.subtype,
                "bits": bits_in,
                "mask": f"0x{mask:08X}" if mask is not None else None,
                "roles": roles,
            },
            "dest": {
                "subtype": out_subtype,
                "bits": bits_out,
                "normalize": normalize,
                "scale": scale if normalize else 1.0,
                "transform": transform_name,
            },
        }
        if detection_notes:
            details.update(detection_notes)
        if verbose >= 2:
            details["metadata"] = {
                "ffmpeg": bool(_ffmpeg_path()),
                "ffprobe": bool(_ffprobe_path()),
                "before_keys": sorted(list(meta_before.keys())) if isinstance(meta_before, dict) else None,
                "after_keys": sorted(list(meta_after.keys())) if isinstance(meta_after, dict) else None,
            }
        result["details"] = details
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
        # Prefer Ambisonics first (bias)
        ambi_roles, ambi_kind, ambi_info = detect_ambisonics(in_path, ch)
        detection_notes: Optional[Dict[str, Any]] = None
        layout = None
        mask = None
        roles = None
        transform = None
        transform_name = 'passthrough' if ch == 2 else None
        if ambi_info and ambi_info.get('format') == 'A':
            mat = None
            if aformat_matrix:
                mat = _load_matrix_from_json(aformat_matrix)
            if mat is None and aformat_preset:
                mat = _aformat_preset_matrix(aformat_preset)
            if mat is not None and ch == 4:
                transform = make_ambisonics_aformat_transform(mat, 'FuMa')
                roles = ['W', 'X', 'Y', 'Z']
                transform_name = 'ambisonics-aformat+ab'
                detection_notes = {'ambisonics': {'detected': True, 'format': 'A', 'kind': 'FuMa', 'info': ambi_info, 'aformat': {'preset': aformat_preset, 'matrix': str(aformat_matrix) if aformat_matrix else None}}}
            else:
                detection_notes = {'ambisonics': {'detected': True, 'format': 'A', 'kind': None, 'info': ambi_info, 'warning': 'A-format requires mic-specific A→B; provide --aformat-preset generic or --aformat-matrix'}}
        elif ambi_roles:
            # Allow override or default bias for ambiguous decisions, with optional energy scan
            ambi_kind_norm = (ambi_kind or 'auto').lower()
            info_decider = (ambi_info or {}).get('decider') if isinstance(ambi_info, dict) else None
            if ambi_kind_norm in ('fuma', 'ambix'):
                chosen_kind = 'AmbiX' if ambi_kind_norm == 'ambix' else 'FuMa'
                reason = 'override'
            else:
                # Try energy scan first when auto
                kind_scan, scan_info = _decide_ambix_fuma_by_energy(_compute_channel_rms(in_path) or [])
                if kind_scan in ('AmbiX', 'FuMa'):
                    chosen_kind = kind_scan
                    reason = 'energy-scan'
                elif info_decider == 'keyword-ambix':
                    chosen_kind = 'AmbiX'
                    reason = 'keyword'
                elif info_decider == 'keyword-fuma':
                    chosen_kind = 'FuMa'
                    reason = 'keyword'
                else:
                    # ambiguous/default bias → prefer AmbiX by default
                    if (ambi_default or 'ambix').lower() == 'ambix':
                        chosen_kind = 'AmbiX'
                        reason = 'default-bias'
                    else:
                        chosen_kind = 'FuMa'
                        reason = 'default-bias'
            roles = ambi_roles
            tm = make_ambisonics_transform(roles, chosen_kind)
            transform = tm
            transform_name = f"ambisonics-{chosen_kind.lower()}"
            detection_notes = {'ambisonics': {'detected': True, 'format': 'B', 'kind': chosen_kind, 'info': ambi_info, 'decision': reason}}
            # attach scan info if used
            try:
                if 'scan_info' in locals() and isinstance(scan_info, dict):
                    detection_notes['ambisonics']['scan'] = scan_info
            except Exception:
                pass
        if transform is None:
            mask = parse_wav_channel_mask(in_path)
            roles = roles_from_mask(mask, ch)
            if roles:
                tm = make_roles_transform(roles)
                transform = (lambda x: x) if ch == 2 else (tm or downmix_block)
                transform_name = 'mask'
        if transform is None:
            layout = _ffprobe_layout(in_path)
            roles = roles_from_ffprobe_layout(layout, ch)
            if roles:
                tm = make_roles_transform(roles)
                transform = (lambda x: x) if ch == 2 else (tm or downmix_block)
                transform_name = 'ffprobe'
        if transform is None:
            roles = heuristic_roles_for_channels(ch)
            tm = make_roles_transform(roles) if roles else None
            transform = (lambda x: x) if ch == 2 else (tm or downmix_block)
            transform_name = 'heuristic'
        if normalize:
            peak = _compute_peak(in_path, chunk_size=chunk_size, transform=transform)
            scale = (1.0 / peak) if peak > 1.0 else 1.0
            _ = sfi.seek(0)
        else:
            scale = 1.0
        _write_streamed(sfi, out_path, subtype=out_subtype, chunk_size=chunk_size, transform=transform, scale=scale)
        meta_before = _ffprobe_tags(in_path) if verbose >= 2 else None
        meta_err = _copy_metadata_with_ffmpeg(out_path, in_path)
        if meta_err is None and verbose >= 2:
            after_try = _ffprobe_tags(out_path)
            if isinstance(meta_before, dict) and isinstance(after_try, dict):
                missing = [k for k in meta_before.keys() if k not in after_try]
                if missing:
                    meta_err2 = _ffmpeg_apply_metadata(out_path, meta_before)
                    if meta_err2:
                        meta_err = meta_err2
        meta_after = _ffprobe_tags(out_path) if verbose >= 2 else None
        bits_in = _bits_for_subtype(sfi.subtype)
        bits_out = _bits_for_subtype(out_subtype)
        result: Dict[str, Any] = {"path": str(in_path), "out": str(out_path), "sr": sfi.samplerate, "in_channels": ch, "out_channels": 2}
        details: Dict[str, Any] = {
            "source": {
                "channels": ch,
                "samplerate": sfi.samplerate,
                "subtype": sfi.subtype,
                "bits": bits_in,
                "mask": f"0x{mask:08X}" if mask is not None else None,
                "roles": roles,
            },
            "dest": {
                "subtype": out_subtype,
                "bits": bits_out,
                "normalize": normalize,
                "scale": scale if normalize else 1.0,
                "transform": transform_name,
            },
        }
        if detection_notes:
            details.update(detection_notes)
        if verbose >= 2:
            details["metadata"] = {
                "ffmpeg": bool(_ffmpeg_path()),
                "ffprobe": bool(_ffprobe_path()),
                "before_keys": sorted(list(meta_before.keys())) if isinstance(meta_before, dict) else None,
                "after_keys": sorted(list(meta_after.keys())) if isinstance(meta_after, dict) else None,
            }
        result["details"] = details
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
    ap.add_argument('--aformat-preset', choices=['generic'], default=None, help='If Ambisonics A-format is detected, use this preset 4x4 A→B matrix before decoding (experimental).')
    ap.add_argument('--aformat-matrix', type=Path, default=None, help='Path to JSON file containing a 4x4 A→B matrix (rows=W,X,Y,Z; cols=channels 1..4). Overrides --aformat-preset.')
    ap.add_argument('--ambisonics-kind', choices=['auto', 'fuma', 'ambix'], default='auto', help='Force Ambisonics B-format kind (auto/fuma/ambix).')
    ap.add_argument('--ambisonics-default', choices=['fuma', 'ambix'], default='ambix', help='When auto detection is ambiguous, prefer this kind (default: ambix).')
    ap.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v or -vv).')

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
                verbose=args.verbose,
                aformat_preset=args.aformat_preset,
                aformat_matrix=args.aformat_matrix,
                ambi_kind=args.ambisonics_kind,
                ambi_default=args.ambisonics_default,
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

        if args.verbose >= 1 and r.get('details'):
            d = r['details']
            src = d.get('source', {})
            dst = d.get('dest', {})
            print(f"  Original: {src.get('channels')} channels @ {src.get('samplerate')} Hz, {src.get('subtype')} ({src.get('bits')}‑bit)")
            # Ambisonics detection info (if applicable)
            ambi = d.get('ambisonics')
            if ambi and ambi.get('detected'):
                fmt = ambi.get('format')
                kind = ambi.get('kind')
                info = ambi.get('info') or {}
                ev = info.get('evidence') or []
                decider = info.get('decider')
                if fmt == 'A':
                    ainfo = ambi.get('aformat') or {}
                    if ainfo.get('matrix') or ainfo.get('preset'):
                        used = ainfo.get('preset') or 'custom-matrix'
                        print(f"  Ambisonics: A-format detected based on: {', '.join(ev) if ev else 'indicators'}; converted A→B using preset '{used}'")
                    else:
                        print(f"  Ambisonics: A-format detected based on: {', '.join(ev) if ev else 'indicators'}; no A→B provided — using non-ambisonic downmix fallback")
                else:
                    reason_txt = ''
                    decision = ambi.get('decision')
                    if decision:
                        reason_txt = f" (decision: {decision})"
                    elif decider:
                        reason_txt = f" (detected by {decider})"
                    print(f"  Ambisonics: detected as {kind} based on: {', '.join(ev) if ev else 'unspecified indicators'}{reason_txt}")
                    # If we decided via an energy scan or fell back to default-bias, explain why and show levels
                    scan = ambi.get('scan') if isinstance(ambi.get('scan'), dict) else None
                    if decision in ('energy-scan', 'default-bias'):
                        if decision == 'energy-scan':
                            print("    Not enough metadata to determine AmbiX vs FuMa; analyzed audio levels to decide.")
                        elif decision == 'default-bias':
                            print("    Metadata was ambiguous; no decisive keywords; using default preference (AmbiX by default).")
                        if scan and isinstance(scan.get('rms'), (list, tuple)):
                            rms = [float(x) for x in scan.get('rms')]
                            # Compute dB per channel
                            eps = 1e-12
                            db = [20.0 * np.log10(x if x > eps else eps) for x in rms]
                            ch_list = ', '.join([f"ch{i+1}={rms[i]:.6f} ({db[i]:+.1f} dB)" for i in range(len(rms))])
                            print(f"    Energy scan RMS (first ~262k frames): {ch_list}")
                            if 'min_index' in scan:
                                mi = scan.get('min_index')
                                conf = scan.get('confidence_db')
                                print(f"    Lowest among channels 2..4: ch{mi} (confidence {conf:+.1f} dB) → selected {kind}.")
            # Friendly layout and downmix plan
            roles = src.get('roles')
            shown_layout = False
            if roles:
                # If roles came from mask or ffprobe, present them as layout
                print("  Channel layout:", ", ".join(f"{FRIENDLY_ROLE.get(r, r)} ({r})" for r in roles))
                shown_layout = True
            else:
                # Try mask-based layout string
                src_mask = src.get('mask')
                src_mask_int = int(src_mask, 16) if isinstance(src_mask, str) and src_mask.startswith('0x') else None
                if src_mask_int is not None:
                    friendly = friendly_mask_list(src_mask_int, src.get('channels') or 0)
                    if friendly:
                        print(f"  Channel layout: {friendly}")
                        roles = roles_from_mask(src_mask_int, src.get('channels') or 0)
                        shown_layout = True
            # If still no roles, use heuristic and tell the user
            if not roles:
                roles = heuristic_roles_for_channels(src.get('channels') or 0)
                if roles:
                    print("  Guessed channel layout (common pattern):", ", ".join(f"{FRIENDLY_ROLE.get(r, r)} ({r})" for r in roles))
                else:
                    print("  Channel layout: not signaled; using generic averaging")
            # Always show a downmix plan
            if roles:
                if isinstance(dst.get('transform'), str) and dst.get('transform', '').startswith('ambisonics-'):
                    left_desc, right_desc = describe_downmix_from_ambisonics(roles)
                else:
                    left_desc, right_desc = describe_downmix_from_roles(roles)
                print(f"  Downmix plan → Left:  {left_desc}")
                print(f"                 Right: {right_desc}")
            norm_txt = f"on (scale {dst.get('scale'):.3f})" if dst.get('normalize') else "off"
            print(f"  Output: stereo, {dst.get('subtype')} ({dst.get('bits')}‑bit), transform: {dst.get('transform')}, normalization: {norm_txt}")
            if args.verbose >= 2 and d.get('metadata'):
                md = d['metadata']
                print(f"  Metadata propagation via ffmpeg: {'yes' if md.get('ffmpeg') else 'no'}; ffprobe available: {'yes' if md.get('ffprobe') else 'no'}")
                # Re-probe full tags for values and provide a summary
                src_tags = _ffprobe_tags(Path(r['path'])) or {}
                dst_tags = _ffprobe_tags(Path(r['out'])) or {}
                src_keys = set(src_tags.keys())
                dst_keys = set(dst_tags.keys())
                missing_keys = sorted(list(src_keys - dst_keys))
                common_keys = sorted(list(src_keys & dst_keys))
                changed_keys = [k for k in common_keys if str(src_tags.get(k)) != str(dst_tags.get(k))]
                propagated_keys = [k for k in common_keys if str(src_tags.get(k)) == str(dst_tags.get(k))]
                print(f"  Metadata summary: before={len(src_keys)}, after={len(dst_keys)}, propagated={len(propagated_keys)}, changed={len(changed_keys)}, missing={len(missing_keys)}")
                for k in sorted(src_keys):
                    v_before = src_tags.get(k)
                    v_after = dst_tags.get(k)
                    if v_after is None:
                        print(f"    [missing] {k}: {v_before}")
                    elif str(v_after) == str(v_before):
                        print(f"    [propagated] {k}: {v_after}")
                    else:
                        print(f"    [changed] {k}: before='{v_before}' after='{v_after}'")

    # If any metadata could not be preserved, print guidance on ffmpeg setup
    if any(r.get('metadata_warning') for r in results):
        print("\nNote: Some files did not have metadata preserved.")
        print("- This tool relies on ffmpeg to copy RIFF/BWF/iXML tags into the new file.")
        print("- Install ffmpeg and ensure 'ffmpeg' and 'ffprobe' are on your PATH, then re-run.")
        print("Windows quick options:")
        print("  1) Chocolatey:   choco install ffmpeg")
        print("  2) Winget:        winget install Gyan.FFmpeg or winget install ffmpeg")
        print("  3) Manual:        Download a static build (e.g., from gyan.dev), unzip, add the 'bin' folder to PATH")
        print("Verify:  ffmpeg -version   and   ffprobe -version")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
