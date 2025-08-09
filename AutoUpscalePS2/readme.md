
# AutoUpscalePS2

A Python tool for automatically upscaling PlayStation 2 game textures dumped from PCSX2 using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
This script processes texture files, upscales them, patches mip levels, and prepares them for re-injection.

## Features

- Batch upscale PS2 textures (4x by default via Real-ESRGAN)
- Uses highest mip level for all mip levels in textures with mips
- Automates upscaling, mip patching, and copying to output

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

## License

See [LICENSE](LICENSE) for details.
