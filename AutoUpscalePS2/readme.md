
# AutoUpscalePS2

Automatic batch upscaling & preparation of PlayStation 2 textures dumped from PCSX2 using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Handles: upscale → optional small/large variant consolidation → mip patching → final replacement folder. Includes a companion similarity analysis tool for low‑resolution LOD duplicates.

---

## Quick Start

```powershell
# Game folder mode (auto uses dumps/intermediates/replacements)
python upscale.py -r "C:\realesrgan\realesrgan-ncnn-vulkan.exe" -g "C:\Users\you\Documents\PCSX2\textures\<GAME_SERIAL>"

# Add ID-based replacement (recommended for LOD/distance quality improvement)
python upscale.py -r <realesrgan> -g <game_path> --id-replace
```

Upscaled textures appear in `replacements/` for PCSX2 to load (ensure PCSX2 "Load Textures" is enabled; use the hotkey to reload as needed).

---

## Core Features

| Area | What it does |
|------|--------------|
| Upscaling | One Real-ESRGAN pass across all dumped textures |
| Mip Patching | Highest mip copied onto all lower mip variants (`*-mipN`) |
| ID Variant Consolidation (optional) | Replaces smaller per-ID textures with perceptually similar larger ones in the same ID group |
| Similarity Detection (helper script) | Reports visually similar non-mip low-res vs high-res pairs (hash + optional pixel diff) |
| Safety & Preview | Dry run mode + multi-level verbosity (`-v`, `-vv`, `-vvv`) |

---

## Requirements

* Python 3.7+
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (native NCNN or Python script)
* [PCSX2](https://pcsx2.net/) (for texture dumping/loading)
* Pillow (`pip install Pillow`) for similarity features

---

## PCSX2 Configuration

1. Enable dumping: `Settings > Graphics > Texture Replacement > Dump Textures`.
2. Enable loading: `Settings > Graphics > Texture Replacement > Load Textures`.
3. Optional performance: enable `Asynchronous Texture Loading` + `Precache Textures` (esp. with Vulkan).
4. Bind a hotkey: `Graphics > Reload Texture Replacements` to apply new replacements after each upscale run.

---

## Folder Layout (Game Mode `-g`)

Given `-g C:\...\textures\<GAME_SERIAL>`:

```
<GAME_SERIAL>\
	dumps\          # Created by PCSX2 (input to pipeline)
	intermediates\  # Created by script (upscaled + patched working set)
	replacements\   # Final output PCSX2 consumes
```

You may override with explicit `-i -m -o` paths instead of `-g`.

---

## Verbosity Levels

| Flag | Adds |
|------|------|
| (none) | Summary only |
| -v | Configuration + step summaries |
| -vv | Discovery details (commands, grouping) |
| -vvv | Per-file actions (copy, hash outcomes) |
| -vvvv | Raw Real-ESRGAN stderr passthrough |

---

## Command Overview (Main Script)

| Flag | Purpose |
|------|---------|
| `-g / --game PATH` | Use standard PCSX2 texture folder layout |
| `-i / -m / -o` | Manual input/intermediate/output paths |
| `-r / --realesrgan` | Real-ESRGAN executable / script or directory |
| `--realesrgan-args "..."` | Extra args passed through untouched |
| `--dry-run` | Simulate all steps (no writes) |
| `--id-replace` | Enable ID-based small→large texture replacement |
| `--id-small-threshold N` | Max dimension marking SMALL (default 256) |
| `--id-large-threshold N` | Min dimension marking LARGE (default 256) |
| `--id-hash-size S` | aHash size 4–16 (default 8) |
| `--id-hash-threshold D` | Max Hamming distance to allow replacement (default 6) |

---

## ID-Based Small Texture Replacement

Many games embed alternate LOD texture variants instead of relying purely on mip chains. These often differ only by an internal numeric second segment in the filename (`prefix-<ID>-rest.png`). This feature consolidates smaller variants to the best matching larger one—only when they are visually similar.

Process per ID group:
1. Split into LARGE (≥ `--id-large-threshold`) and SMALL (< `--id-small-threshold`).
2. Compute aHash for each LARGE & SMALL (configurable size `--id-hash-size`).
3. For each SMALL choose the LARGE with minimum Hamming distance.
4. Replace only if distance ≤ `--id-hash-threshold`.

Tuning tips:
* Start with defaults (`8 / 6`). If many legitimate matches are skipped, raise threshold to 7–9.
* If false positives occur, either lower threshold or raise hash size to 10–12 (and re-adjust threshold ~10–15% of bits).
* Use `--dry-run -vv` to inspect which pairs would be replaced vs skipped (`Similarity skips`).

Example:
```powershell
python upscale.py -r <realesrgan> -g <game_path> --id-replace \
	--id-small-threshold 256 --id-large-threshold 512 \
	--id-hash-size 8 --id-hash-threshold 8 -vv
```

---

## Similarity Analysis Helper (`detect_similar_images.py`)

Use this separate tool to audit non-mip low-res textures that are near duplicates of higher-res bases (useful for manual cleanup or validating ID replacement).

Basic examples:
```powershell
python detect_similar_images.py -g "C:\Users\you\Documents\PCSX2\textures\<GAME_SERIAL>"
python detect_similar_images.py -i "...\intermediates" --hash-threshold 6 --verify-diff --diff-threshold 12
```

Key arguments:
| Flag | Meaning |
|------|---------|
| `-g / -i` | Select intermediates folder automatically or directly |
| `--base-size N` | Boundary between high vs low (default 1024). Use 256 to mirror ID replacement scale |
| `--hash-size S` | aHash dimension (NxN) |
| `--hash-threshold D` | Hamming distance cutoff for candidate match |
| `--verify-diff` | Adds mean absolute pixel diff check (costlier, more precise) |
| `--diff-threshold X` | Pixel diff threshold (0–255 scale) |
| `--top-k K` | Show K closest highs per low (ignores threshold for listing) |
| `--pair A B` | Direct compare two images |
| `--visual-dir DIR` | Generate composite + heatmap images (+ optional gallery) |
| `--csv file.csv` | Export matches |

Interpretation (aHash 8×8 guidance):
* 0–4: Near-identical
* 5–10: Similar (minor detail/color changes)
* 11–20: Loosely related / maybe different variant
* 21+: Usually unrelated

Tip: For parity with ID replacement logic, run with `--base-size 256 --hash-size 8`.

Visual output (when using `--visual-dir`):
* Left: Low texture scaled (nearest) – reveals original pixel grid.
* Middle: High texture.
* Right: Red heatmap (difference intensity per pixel).

---

## Workflow Summary

1. Play game to populate `dumps/`.
2. Run upscale pipeline (optionally with `--id-replace`).
3. Reload texture replacements in PCSX2 (hotkey).
4. (Optional) Run similarity helper to audit leftover variants.
5. Iterate as more textures dump.

---

## Troubleshooting

| Symptom | Suggestion |
|---------|------------|
| No images found | Verify `dumps/` not empty; correct `-g` path or `-i` directory |
| Real-ESRGAN non-zero exit | Check executable path & needed model files; run command manually with `-vv` for full args |
| Few ID replacements | Increase `--id-hash-threshold` or lower small/large thresholds; confirm groupings with `-vv` |
| Incorrect replacements | Lower threshold or raise `--id-hash-size`; inspect with dry run first |
| Pillow import errors | `pip install Pillow` in your active Python environment |

---

## Cheat Sheet

| Goal | Command (PowerShell) |
|------|----------------------|
| Basic upscale | `python upscale.py -r <realesrgan> -g <game>` |
| Include ID replacement | `python upscale.py -r <realesrgan> -g <game> --id-replace` |
| Dry run preview | `python upscale.py -r <realesrgan> -g <game> --id-replace --dry-run` |
| Similarity audit (defaults) | `python detect_similar_images.py -g <game>` |
| Similarity audit @256 base | `python detect_similar_images.py -g <game> --base-size 256` |
| Visual gallery | `python detect_similar_images.py -g <game> --visual-dir visuals --hash-threshold 8` |

---

## License

See [LICENSE](LICENSE) for details.
