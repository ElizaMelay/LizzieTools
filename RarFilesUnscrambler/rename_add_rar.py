import argparse
import os


def add_rar_extension(folder):
    folder = os.path.abspath(folder)
    print(f"Processing folder: {folder}")

    count = 0
    for entry in os.scandir(folder):
        if not entry.is_file():
            continue

        path = entry.path
        dirname, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        # Only rename files with no extension at all
        if ext:
            continue

        new_name = filename + ".rar"
        new_path = os.path.join(dirname, new_name)

        # Avoid overwriting anything accidentally
        if os.path.exists(new_path):
            print(f"SKIP (exists): {new_name}")
            continue

        os.rename(path, new_path)
        print(f"RENAMED: {filename} -> {new_name}")
        count += 1

    print(f"Done. Renamed {count} file(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Add .rar extension to all files with no extension in a folder."
    )
    parser.add_argument(
        "directory",
        help="Directory containing the scrambled files",
    )
    args = parser.parse_args()
    add_rar_extension(args.directory)


if __name__ == "__main__":
    main()