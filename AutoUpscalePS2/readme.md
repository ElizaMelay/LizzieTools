
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

Note: These folders are typically located in `C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>\`
- `dumps\` - original dumped textures from PCSX2
- `intermediates\` - upscaled and patched textures (created by script, should be a sibling of `dumps` and `replacements`)
- `replacements\` - final textures for PCSX2 to load

## Usage

1. Install dependencies.
2. Create an `intermediates` folder next to your `dumps` and `replacements` folders.
3. Run the script:
	```
	python upscale.py -r <path_to_realesrgan> -i <path_to_dumps> -m <path_to_intermediates> -o <path_to_replacements>
	```
	- Example:
	  ```
	  python upscale.py -r "C:\realesrgan\realesrgan-ncnn-vulkan.exe" -i "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>\dumps" -m "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>\intermediates" -o "C:\Users\username\Documents\PCSX2\textures\<GAME_SERIAL>\replacements"
	  ```
	- You can pass extra arguments to Real-ESRGAN with `--realesrgan-args "<args>"`.

4. The upscaled textures will be ready in the `replacements` folder for PCSX2.

Note: This script and PCSX2 do not run continuously. As the game is played, more textures are dumped to disk and need to be upscaled. PCSX2 needs to have replacement textures manually refreshed. A hotkey can be bound in `Settings > Hotkeys` under `Graphics > Reload Texture Replacements`.

## License

See [LICENSE](LICENSE) for details.
