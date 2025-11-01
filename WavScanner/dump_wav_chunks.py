#!/usr/bin/env python3
"""
Dump RIFF/WAV chunk structure and metadata without including raw audio data.

- Supports RIFF and RF64 (ds64) headers
- Lists all chunks with offsets and sizes; applies ds64 extended sizes when present
- Parses common chunk types: fmt, ds64, fact, LIST/INFO (key/value), bext (basic fields), iXML (as text)
- Skips dumping the 'data' payload (reports its size and offset only)
- Optional hex preview of non-audio chunk payloads (configurable size)

Usage:
  python dump_wav_chunks.py <file.wav> [--json] [--hexdump 64]

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _read_u16le(b: bytes, off: int = 0) -> int:
    return int.from_bytes(b[off:off+2], 'little', signed=False)

def _read_i16le(b: bytes, off: int = 0) -> int:
    return int.from_bytes(b[off:off+2], 'little', signed=True)

def _read_u32le(b: bytes, off: int = 0) -> int:
    return int.from_bytes(b[off:off+4], 'little', signed=False)

def _read_u64le(b: bytes, off: int = 0) -> int:
    return int.from_bytes(b[off:off+8], 'little', signed=False)

class RiffReader:
    def __init__(self, path: Path):
        self.path = path
        self.f = None
        self.is_rf64 = False
        self.riff_size_32: Optional[int] = None
        self.ds64: Dict[str, int] = {}
        self.riff_end: Optional[int] = None

    def __enter__(self):
        self.f = self.path.open('rb')
        self._read_header()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.f:
                self.f.close()
        finally:
            self.f = None

    def _read_exact(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError('Unexpected EOF')
        return b

    def _read_header(self):
        f = self.f
        sig = self._read_exact(4)
        if sig not in (b'RIFF', b'RF64'):
            raise ValueError('Not a RIFF/RF64 file')
        size = _read_u32le(self._read_exact(4))
        wave = self._read_exact(4)
        if wave != b'WAVE':
            raise ValueError('Not a WAVE file')
        self.is_rf64 = (sig == b'RF64')
        self.riff_size_32 = size
        # Riff end based on header size (RIFF size excludes 8-byte header)
        if not self.is_rf64 and size != 0:
            self.riff_end = 8 + size
        else:
            # For RF64 or unknown, fall back to file size
            try:
                self.riff_end = os.fstat(f.fileno()).st_size
            except Exception:
                self.riff_end = None

    def _within_riff(self) -> bool:
        if self.riff_end is None:
            return True
        return self.f.tell() + 8 <= self.riff_end  # ensure header fits

    def iter_chunks(self):
        f = self.f
        # If RF64, first chunk should be ds64; but we'll just loop
        while True:
            # Stop if we cannot read a full header
            if not self._within_riff():
                break
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid = hdr[0:4]
            csz = _read_u32le(hdr, 4)
            off = f.tell()  # payload start
            payload = None
            # For 'data' with RF64 and size==0xFFFFFFFF, extended size from ds64 applies
            extended_size: Optional[int] = None
            if cid == b'ds64':
                payload = f.read(csz)
                # Parse first 24 bytes (or 28 by spec including table len)
                if len(payload) >= 24:
                    riff_sz = _read_u64le(payload, 0)
                    data_sz = _read_u64le(payload, 8)
                    samp_ct = _read_u64le(payload, 16)
                    self.ds64['RIFF'] = riff_sz
                    self.ds64['data'] = data_sz
                    self.ds64['samples'] = samp_ct
            elif cid == b'fmt ':
                payload = f.read(csz)
            elif cid == b'LIST':
                payload = f.read(csz)
            elif cid == b'bext':
                payload = f.read(csz)
            elif cid == b'iXML':
                payload = f.read(csz)
            elif cid == b'fact':
                payload = f.read(csz)
            elif cid == b'junk':
                payload = f.read(csz)
            elif cid == b'umid':
                payload = f.read(csz)
            elif cid == b'minf':
                payload = f.read(csz)
            elif cid == b'regn':
                payload = f.read(csz)
            elif cid == b'DGDA':
                payload = f.read(csz)
            else:
                # Unknown or 'data' or anything else; we will not load 'data'
                if cid == b'data' and csz == 0xFFFFFFFF and 'data' in self.ds64:
                    extended_size = int(self.ds64['data'])
                # seek past
                f.seek(csz, os.SEEK_CUR)

            # Pad to even
            if (csz & 1):
                f.seek(1, os.SEEK_CUR)

            yield (cid, csz, off, payload, extended_size)


def _parse_fmt(payload: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(payload) >= 16:
        out['audio_format_tag'] = _read_u16le(payload, 0)
        out['channels'] = _read_u16le(payload, 2)
        out['sample_rate'] = _read_u32le(payload, 4)
        out['avg_bytes_per_sec'] = _read_u32le(payload, 8)
        out['block_align'] = _read_u16le(payload, 12)
        out['bits_per_sample'] = _read_u16le(payload, 14)
        if out['audio_format_tag'] == 0xFFFE and len(payload) >= 40:
            out['valid_bits_per_sample'] = _read_u16le(payload, 18)
            out['channel_mask'] = _read_u32le(payload, 20)
            # GUID bytes 24..39
            guid = payload[24:40]
            out['subformat_guid'] = guid.hex()
    out['raw_len'] = len(payload)
    return out


def _parse_list_info(payload: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {'type': None, 'entries': []}
    if len(payload) < 4:
        return out
    ltype = payload[0:4]
    out['type'] = ltype.decode('ascii', errors='replace')
    pos = 4
    while pos + 8 <= len(payload):
        cid = payload[pos:pos+4]
        csz = _read_u32le(payload, pos+4)
        pos += 8
        val = payload[pos:pos+csz]
        pos += csz
        if (csz & 1) and pos < len(payload):
            pos += 1  # pad
        try:
            key = cid.decode('ascii', errors='replace')
        except Exception:
            key = cid.hex()
        # INFO strings are typically null-terminated
        sval = val.rstrip(b'\x00').decode('utf-8', errors='replace')
        out['entries'].append({'id': key, 'value': sval})
    return out


def _parse_bext(payload: bytes) -> Dict[str, Any]:
    # BWF Broadcast Extension (v0..v2). We'll parse common fixed fields.
    out: Dict[str, Any] = {}
    try:
        def rstr(off: int, n: int) -> str:
            return payload[off:off+n].split(b'\x00', 1)[0].decode('utf-8', errors='replace')
        if len(payload) >= 256:
            out['description'] = rstr(0, 256)
        if len(payload) >= 256+32:
            out['originator'] = rstr(256, 32)
        if len(payload) >= 256+32+32:
            out['originator_reference'] = rstr(288, 32)
        if len(payload) >= 256+32+32+10:
            out['origination_date'] = rstr(320, 10)
        if len(payload) >= 256+32+32+10+8:
            out['origination_time'] = rstr(330, 8)
        if len(payload) >= 340+4:
            out['time_reference_low'] = _read_u32le(payload, 340)
        if len(payload) >= 344+4:
            out['time_reference_high'] = _read_u32le(payload, 344)
        if len(payload) >= 348+2:
            out['version'] = _read_u16le(payload, 348)
        # UMID 64 bytes at 350 for v1+, reserved then coding history string
        if len(payload) > 384:
            # coding history starts after fixed area; attempt to locate by searching for first zero padding area
            chist = payload[602:] if len(payload) > 602 else b''
            if chist:
                out['coding_history'] = chist.replace(b'\r\n', b' | ').decode('utf-8', errors='replace')
    except Exception:
        pass
    out['raw_len'] = len(payload)
    return out


def _parse_ixml(payload: bytes) -> Dict[str, Any]:
    try:
        txt = payload.decode('utf-8', errors='replace')
    except Exception:
        txt = ''
    return {'xml': txt, 'raw_len': len(payload)}


def _hexdump(b: bytes, max_bytes: int) -> str:
    n = min(len(b), max(0, max_bytes))
    data = b[:n]
    return data.hex()


def dump_wav(path: Path, hexdump: int = 0) -> Dict[str, Any]:
    info: Dict[str, Any] = {'path': str(path), 'chunks': [], 'rf64': False, 'ds64': {}}
    with RiffReader(path) as rr:
        info['rf64'] = rr.is_rf64
        if rr.ds64:
            info['ds64'] = rr.ds64.copy()
        for cid, csz, off, payload, ext_sz in rr.iter_chunks():
            cid_txt = cid.decode('ascii', errors='replace')
            entry: Dict[str, Any] = {
                'id': cid_txt,
                'offset': off,
                'size': csz,
            }
            if ext_sz is not None:
                entry['extended_size'] = ext_sz
            summary: Dict[str, Any] = {}
            if cid == b'fmt ' and payload is not None:
                summary = _parse_fmt(payload)
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'ds64' and payload is not None:
                # already parsed into rr.ds64; include a quick view
                summary = rr.ds64.copy()
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'LIST' and payload is not None:
                summary = _parse_list_info(payload)
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'fact' and payload is not None:
                if len(payload) >= 4:
                    summary = {'sample_length': _read_u32le(payload, 0), 'raw_len': len(payload)}
                else:
                    summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'bext' and payload is not None:
                summary = _parse_bext(payload)
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'iXML' and payload is not None:
                summary = _parse_ixml(payload)
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'junk' and payload is not None:
                # Just report raw length
                summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'umid' and payload is not None:
                summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'minf' and payload is not None:
                summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'regn' and payload is not None:
                summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'DGDA' and payload is not None:
                summary = {'raw_len': len(payload)}
                if hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
            elif cid == b'data':
                # Do not read payload; just report sizes
                pass
            else:
                # Unknown chunk; include a short hex preview if requested
                if payload is not None and hexdump:
                    entry['hex'] = _hexdump(payload, hexdump)
                if payload is not None:
                    summary = {'raw_len': len(payload)}
            if summary:
                entry['summary'] = summary
            info['chunks'].append(entry)
    return info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Dump RIFF/WAV chunk structure without audio data')
    ap.add_argument('path', type=Path, help='Path to WAV file to inspect')
    ap.add_argument('--json', action='store_true', help='Output JSON instead of human-readable text')
    ap.add_argument('--hexdump', type=int, default=0, help='Include a hex preview of the first N bytes of each non-audio chunk (0 disables)')
    args = ap.parse_args(argv)

    info = dump_wav(args.path, hexdump=args.hexdump)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"File: {info['path']}")
    print(f"RF64: {'yes' if info.get('rf64') else 'no'}")
    if info.get('ds64'):
        ds = info['ds64']
        riff_sz = ds.get('RIFF')
        data_sz = ds.get('data')
        samp_ct = ds.get('samples')
        print(f"ds64: RIFFSize={riff_sz} dataSize={data_sz} sampleCount={samp_ct}")
    for ch in info['chunks']:
        eid = ch['id']
        off = ch['offset']
        size = ch['size']
        ext = ch.get('extended_size')
        ext_txt = f" (ext {ext})" if ext is not None else ''
        print(f"- {eid} @ 0x{off:08X} size={size}{ext_txt}")
        if 'summary' in ch:
            summ = ch['summary']
            if eid == 'fmt ':
                af = summ.get('audio_format_tag')
                chn = summ.get('channels')
                sr = summ.get('sample_rate')
                bps = summ.get('bits_per_sample')
                mask = summ.get('channel_mask')
                subformat = summ.get('subformat_guid')
                if subformat:
                    print(f"    fmt: tag=0x{af:04X} channels={chn} rate={sr} bits={bps} mask={('0x%08X'%mask) if mask is not None else 'n/a'} subformat={subformat}")
                else:
                    print(f"    fmt: tag=0x{af:04X} channels={chn} rate={sr} bits={bps} mask={('0x%08X'%mask) if mask is not None else 'n/a'}")
            elif eid == 'LIST':
                ltype = summ.get('type')
                print(f"    LIST type={ltype}")
                for ent in summ.get('entries', []) or []:
                    print(f"      {ent.get('id')}: {ent.get('value')}")
            elif eid == 'bext':
                keys = ['description','originator','originator_reference','origination_date','origination_time','version']
                for k in keys:
                    v = summ.get(k)
                    if v:
                        print(f"    bext.{k}: {v}")
            elif eid == 'iXML':
                xml = (summ.get('xml') or '').strip()
                if xml:
                    #first_line = xml.splitlines()[0]
                    #print(f"    iXML: {first_line[:120]}{'…' if len(first_line)>120 else ''}")
                    print(f"    iXML:\n{xml}")
            elif eid == 'fact':
                sl = summ.get('sample_length')
                if sl is not None:
                    print(f"    fact.sample_length: {sl}")
            else:
                # Unknown; print raw_len if we had payload
                if 'raw_len' in summ:
                    print(f"    payload_len: {summ['raw_len']}")
        if 'hex' in ch:
            print(f"    hex: {ch['hex']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
