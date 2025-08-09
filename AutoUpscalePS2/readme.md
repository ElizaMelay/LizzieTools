
# AutoUpscalePS2

A Python tool for automatically upscaling PlayStation 2 game textures dumped from PCSX2 using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
This script processes texture files, upscales them, patches mip levels, and prepares them for re-injection.

## Features

- Batch upscale PS2 textures (4x by default via Real-ESRGAN)
- Uses highest mip level for all mip levels in textures with mips
- Automates upscaling, mip patching, and copying to output
- Detects similar standalone low‑resolution LOD textures (non-mip) that closely match a higher resolution base texture (via perceptual hashing)

## Dependencies

- Python 3.7+
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (native or Python version)
- [PCSX2](https://pcsx2.net/) (for texture dumping/loading)

## PCSX2 Setup

1. **Enable Texture Dumping:**
	- In PCSX2, go to `Settings > Graphics > Texture Replacement > Dump Textures` and enable it.
	- Run your game; textures as they are loaded and rendered will appear in the `dumps` folder. 

2. **Enable Texture Replacement:**
	- In PCSX2, go to `Settings > Graphics > Texture Replacement > Load Textures` and enable it.
    - Enabling `Asynchronous Texture Loading` and `Precache Textures` is recommend to avoid slowdown and hitches, especially using Vulkan renderer

## Folder Structure

When using the `-g`/`--game` argument, simply specify the path to your PCSX2 game texture folder (for example, `C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>`). The script will automatically look for and use the following subfolders:

- `dumps\` — Contains the original textures dumped by PCSX2 as you play the game.
- `intermediates\` — Used by the script to store upscaled and patched textures (created automatically if it doesn't exist).
- `replacements\` — Where the final upscaled textures are placed for PCSX2 to load as replacements.

These folders are not required to be created manually (except for `dumps`, which is created by PCSX2 when dumping textures). The script will create `intermediates` and `replacements` as needed. This structure is typical for PCSX2 texture workflows, but you can also specify custom paths using the original arguments if desired.

## Verbosity

You can control how much output the script prints using the `-v`/`--verbose` flag. Add more `v`s for more detail (e.g., `-v`, `-vv`, `-vvv`). The default is minimal output; higher levels show more information about the upscaling process, mip patching, and file copying.

## Usage

1. Install dependencies.
2. Create an `intermediates` folder next to your `dumps` and `replacements` folders.
3. Run the script:
	```
	python upscale.py -r <path_to_realesrgan> -g <path_to_game_textures>
	```
	- Example:
	  ```
	  python upscale.py -r "C:\realesrgan\realesrgan-ncnn-vulkan.exe" -g "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>"
	  ```
	- You can pass extra arguments to Real-ESRGAN with `--realesrgan-args "<args>"`.

4. The upscaled textures will be ready in the `replacements` folder for PCSX2.

Note: This script and PCSX2 do not run continuously. As the game is played, more textures are dumped to disk and need to be upscaled. PCSX2 needs to have replacement textures manually refreshed. A hotkey can be bound in `Settings > Hotkeys` under `Graphics > Reload Texture Replacements`.

## Detecting visually similar LOD textures

Some games ship reduced-detail meshes that reference their own *different* texture files instead of relying on mip chains. After upscaling you might have multiple near-duplicate textures (e.g. a 1024x1024 and a separate 256x256 file) where only the larger one truly needs to remain. The helper script `detect_similar_images.py` searches the `intermediates` folder for low‑resolution, non-mip textures that are perceptually similar to a larger texture.

Run examples:
```
python detect_similar_images.py -g "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>"
python detect_similar_images.py -i "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>\intermediates" --hash-threshold 6 --verify-diff --diff-threshold 12
```

Output lines look like:
```
LOW 256x256 foo_small.png -> HIGH 1024x1024 foo_big.png (hamming=4, diff=7.9)
```
Columns show the matched low/high texture names, sizes, hash distance, and optional mean absolute difference (if `--verify-diff` used). A CSV can be written with `--csv report.csv`.

Key options:
- `--base-size N` (default 1024) threshold to consider a texture high-res
- `--hash-size` (default 8) perceptual hash dimension (NxN)
- `--hash-threshold` maximum Hamming distance for a potential match
- `--verify-diff` enables pixel diff confirmation (slower, more precise)
- `--diff-threshold` mean absolute pixel difference limit when verifying

Use the report to decide whether to remove / alias / copy the higher resolution texture over the low one for consistency.

## License

See [LICENSE](LICENSE) for details.
