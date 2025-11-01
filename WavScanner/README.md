# WavScanner

Scan a folder of `.wav` files and report which ones are multi‑channel audio.

- Default threshold: channels >= 3 (i.e., more than stereo)
- Recursive scanning supported
- Outputs: human-readable table (default), JSON, or CSV
- Only reads headers; no audio decoding

## Quick start (Windows PowerShell)

```powershell
# From this folder
python .\scan_wav_channels.py .

# Recurse into subfolders and output JSON
python .\scan_wav_channels.py C:\Audio\Samples -r --json

# List WAVs with 4 or more channels
python .\scan_wav_channels.py D:\sessions --min-channels 4

# Print all WAV files with their channel counts
python .\scan_wav_channels.py . --print-all

# Save CSV to a file
python .\scan_wav_channels.py . -r --csv > multi_channel_wavs.csv
```

## Output fields

- path: full file path
- channels: number of channels (if header readable)
- sample_rate: sample rate in Hz (best-effort)
- bits_per_sample: bits per sample (best-effort)
- format_tag: WAVE format code (e.g., 1 for PCM, 65534 for WAVE_FORMAT_EXTENSIBLE)
- container: RIFF or RF64
- error: set if the header couldn't be parsed

## Notes

- This tool parses the RIFF/WAVE header (`fmt ` chunk) directly. It works with standard RIFF and RF64 files and doesn't depend on external libraries.
- Compressed WAVs are fine for header inspection; the tool does not try to decode audio data.
- Non-WAV files are ignored. Only files with the `.wav` extension are scanned.

## Mixdown multi-channel to stereo

Use `mixdown_wav_to_stereo.py` to convert multi-channel WAVs into stereo with sensible downmixing:

```powershell
# Install deps (inside the repo venv)
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe -m pip install -r .\requirements.txt

# Pipe scanner JSON directly into mixdown (PowerShell pipeline)
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\scan_wav_channels.py C:\Path\To\Wavs -r --json |
	C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py --from-stdin --only-multi --out-dir .\stereo

# Or downmix from a saved JSON file
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py --from-json .\scan_output.json --only-multi --out-dir .\stereo

# Or downmix all WAVs in a folder (recursively)
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r --out-dir .\stereo

# Keep filenames and write next to sources with a suffix
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r --suffix _stereo --overwrite
```

Options:
- `--from-json` or `--from-stdin` to read the list from the scanner’s JSON output
- `--only-multi` to process only files with 3+ channels (uses `channels` from JSON when present; otherwise probes headers)
- `--subtype` to choose output format (PCM_16 default; PCM_24/PCM_32/FLOAT supported)
- `--normalize/--no-normalize` to control clipping protection
- `--chunk-size` frames per block for large files

### Metadata preservation (ffmpeg required)

To preserve RIFF/BWF/iXML/INFO metadata (e.g., Title, Artist, Description, CodingHistory), this tool uses `ffmpeg` to copy metadata onto the newly written stereo WAV. If `ffmpeg` isn't found, the audio will still be produced but metadata won't be preserved. The script will output a note at the end when metadata wasn't preserved and how to set up ffmpeg.

Install ffmpeg on Windows and ensure `ffmpeg` and `ffprobe` are on your PATH:

```powershell
# Option A: Chocolatey
choco install ffmpeg

# Option B: winget
winget install Gyan.FFmpeg
# or
winget install ffmpeg

# Option C: Manual
# Download a static build from a trusted source (e.g., https://www.gyan.dev/ffmpeg/builds/)
# Unzip and add the 'bin' folder to your System PATH

# Verify
ffmpeg -version
ffprobe -version
```

Once installed, rerun the mixdown. For debugging, use verbosity flags and get human-friendly explanations:

```powershell
# -v: Friendly per-file summary (channels and sample rate; input subtype -> output subtype; whether channel mask used;
#     transform selected; normalization and scale)
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r -v

# -vv: Deeper internals. Shows WAVEFORMATEXTENSIBLE mask as human-readable layout and the downmix plan (with dB weights).
#      Lists every metadata key found with its value and whether it was propagated, changed, or missing after copy.
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r -vv
```

In-place replacement while preserving originals:

```powershell
# Moves each multi-channel original into a sibling subfolder (default 'multitrack_originals')
# and writes the stereo mix back to the original file path
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r --preserve-originals --overwrite

# Use a custom subfolder name for originals
C:/LizzieTools/WavScanner/.venv/Scripts/python.exe .\mixdown_wav_to_stereo.py C:\Path\To\Wavs -r --preserve-originals _original_multitrack --overwrite
```

Notes:
- `--preserve-originals [folder]` moves only multi-channel sources (3+ ch). Stereo files are left untouched.
- When `--preserve-originals` is set, `--out-dir` and `--suffix` are ignored for affected files.

Downmix logic prefers the WAVEFORMATEXTENSIBLE channel mask when present to map channels correctly (e.g., 3F/LFE). If no mask is present, heuristics follow common layouts (3.0/4.0/5.0/5.1/7.1) with ITU-like weights (e.g., C at -3 dB into L/R, LFE at -6 to -12 dB). When layout is unknown, it falls back to a reasonable average.
