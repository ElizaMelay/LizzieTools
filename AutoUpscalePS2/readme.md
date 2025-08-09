
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
- PCSX2 (for texture dumping/loading)

## PCSX2 Setup

1. **Enable Texture Dumping:**
	- In PCSX2, go to `Config > Advanced > Texture Dumping` and enable it.
	- Run your game; dumped textures will appear in the `dumps` folder.

2. **Enable Texture Replacement:**
	- In PCSX2, go to `Config > Advanced > Texture Replacement` and enable it.
	- Place upscaled textures in the `replacements` folder.

## Folder Structure

- `dumps/` — original dumped textures from PCSX2
- `intermediate/` — upscaled and patched textures (created by script, should be a sibling of `dumps` and `replacements`)
- `replacements/` — final textures for PCSX2 to load

## Usage

1. Install dependencies.
2. Create an `intermediate` folder next to your `dumps` and `replacements` folders.
3. Run the script:
	```
	python upscale.py -r <path_to_realesrgan> -i <path_to_dumps> -m <path_to_intermediate> -o <path_to_replacements>
	```
	- Example:
	  ```
	  python upscale.py -r realesrgan-ncnn-vulkan.exe -i dumps -m intermediate -o replacements
	  ```
	- You can pass extra arguments to Real-ESRGAN with `--realesrgan-args "<args>"`.

4. The upscaled textures will be ready in the `replacements` folder for PCSX2.

## License

See [LICENSE](LICENSE) for details.
