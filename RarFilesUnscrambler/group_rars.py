import argparse
import os
from collections import defaultdict

RAR_MAGIC = b"Rar!\x1a\x07"  # RAR4/RAR5, version byte after this


def is_rar(path):
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        return header.startswith(RAR_MAGIC)
    except OSError:
        return False


def rar_version(f):
    # f is an open file at start
    f.seek(0)
    header = f.read(8)
    if not header.startswith(RAR_MAGIC) or len(header) < 8:
        return None
    ver_byte = header[7]
    if ver_byte == 0:
        return 4
    if ver_byte == 1:
        return 5
    return None


def parse_rar4_signature(path):
    """
    Try to extract a stable signature from a RAR4 archive:
    (version, main_flags, first_file_crc, first_file_unpacked_size, first_file_name_length)
    Returns None on failure.
    """
    try:
        with open(path, "rb") as f:
            if rar_version(f) != 4:
                return None

            # Skip magic (7 bytes) + version byte already read
            f.seek(7 + 1)

            # RAR4 block header layout:
            #  2 bytes CRC
            #  1 byte type
            #  2 bytes flags
            #  2 bytes size
            # (optional) 4 bytes ADD_SIZE if flags & 0x8000
            def read_block_header():
                header = f.read(7)
                if len(header) < 7:
                    return None
                crc = int.from_bytes(header[0:2], "little")
                btype = header[2]
                flags = int.from_bytes(header[3:5], "little")
                size = int.from_bytes(header[5:7], "little")
                add_size = 0
                if flags & 0x8000:
                    add_size_bytes = f.read(4)
                    if len(add_size_bytes) < 4:
                        return None
                    add_size = int.from_bytes(add_size_bytes, "little")
                return {
                    "crc": crc,
                    "type": btype,
                    "flags": flags,
                    "size": size,
                    "add_size": add_size,
                }

            # First block is MAIN_HEADER
            main_hdr = read_block_header()
            if not main_hdr or main_hdr["type"] != 0x73:  # 's'
                return None
            main_flags = main_hdr["flags"]
            # Skip rest of MAIN_HEADER body
            to_skip = main_hdr["size"] - 7 + main_hdr["add_size"]
            if to_skip < 0:
                return None
            f.seek(to_skip, os.SEEK_CUR)

            # Now scan for first FILE_HEADER block (type = 0x74 't')
            while True:
                bh = read_block_header()
                if not bh:
                    return None
                btype = bh["type"]
                body_size = bh["size"] - 7 + bh["add_size"]
                if body_size < 0:
                    return None

                if btype == 0x74:  # FILE_HEADER
                    # RAR4 file header body starts with:
                    #   4 bytes PACK_SIZE
                    #   4 bytes UNP_SIZE
                    #   1 byte HOST_OS
                    #   4 bytes FILE_CRC
                    #   4 bytes FTIME
                    #   1 byte UNP_VER
                    #   1 byte METHOD
                    #   2 bytes NAME_SIZE
                    #   4 bytes ATTR
                    body = f.read(min(body_size, 4 + 4 + 1 + 4 + 4 + 1 + 1 + 2 + 4))
                    if len(body) < 4 + 4 + 1 + 4 + 4 + 1 + 1 + 2 + 4:
                        return None
                    unpacked_size = int.from_bytes(body[4:8], "little")
                    file_crc = int.from_bytes(body[9:13], "little")
                    name_size = int.from_bytes(body[4 + 4 + 1 + 4 + 4 + 1 + 1:4 + 4 + 1 + 4 + 4 + 1 + 1 + 2], "little")
                    return ("rar4", main_flags, file_crc, unpacked_size, name_size)
                else:
                    # Skip this block body
                    f.seek(body_size, os.SEEK_CUR)
    except OSError:
        return None


def parse_rar5_signature(path):
    """
    Try to extract a stable signature from a RAR5 archive.
    RAR5 has a more complex header format; we approximate by reading:
      - main header flags
      - first file header's name length and uncompressed size
    Returns (version, main_flags, first_file_unp_size, first_file_name_len) or None.
    """
    try:
        with open(path, "rb") as f:
            if rar_version(f) != 5:
                return None

            # RAR5:
            # magic (7 bytes) + version (1 byte) already read
            f.seek(8)

            # Variable-length integer decoder as per RAR5 spec (simplified).
            def read_vint():
                first = f.read(1)
                if not first:
                    return None
                b = first[0]
                if b < 0x80:
                    return b
                # Continuation:
                value = b & 0x7F
                shift = 7
                while True:
                    nxt = f.read(1)
                    if not nxt:
                        return None
                    nb = nxt[0]
                    value |= (nb & 0x7F) << shift
                    shift += 7
                    if nb < 0x80:
                        break
                return value

            # Read main header
            # main header size (vint), type (vint), flags (vint)
            main_size = read_vint()
            main_type = read_vint()
            main_flags = read_vint()
            if main_size is None or main_type is None or main_flags is None:
                return None

            # Skip the rest of main header body
            if main_size > 0:
                f.seek(main_size, os.SEEK_CUR)

            # Now scan blocks to find first FILE block (type 2)
            while True:
                blk_size = read_vint()
                if blk_size is None:
                    return None
                blk_type = read_vint()
                blk_flags = read_vint()
                if blk_type is None or blk_flags is None:
                    return None

                # For RAR5, file header body starts with:
                #  - optional extra area size, data area size (vint each)
                #  - then file-specific fields including name length and unp size in extra area
                # The full spec is complex; we'll try to read name length and unp size heuristically
                if blk_type == 2:  # FILE_HEADER
                    # For simplicity, read a chunk of the block and derive a basic signature
                    body = f.read(blk_size)
                    if not body:
                        return None
                    # Very crude: use body length and first 32 bytes as signature,
                    # but include main_flags to differentiate archives.
                    # This is still far more robust than comparing only first bytes of the file.
                    sig_len = len(body)
                    sig_prefix = body[:32]
                    sig_hash = hash((sig_len, sig_prefix))
                    return ("rar5", main_flags, sig_hash, sig_len)

                # Otherwise skip this block's body
                if blk_size > 0:
                    f.seek(blk_size, os.SEEK_CUR)
    except OSError:
        return None


def rar_content_signature(path):
    """
    Return a per-archive signature that should be identical for all parts
    of a given multi-part archive and different for other archives.
    """
    # First try RAR4
    sig = parse_rar4_signature(path)
    if sig is not None:
        return sig
    # Then RAR5
    sig = parse_rar5_signature(path)
    if sig is not None:
        return sig
    # Fallback: group as unknown
    size = os.path.getsize(path)
    return ("unknown", size, os.path.basename(path))


def scan_directory(root):
    rar_files = []
    for entry in os.scandir(root):
        if not entry.is_file():
            continue
        path = entry.path
        if is_rar(path):
            rar_files.append(path)
    return rar_files


def group_rars(paths):
    groups = defaultdict(list)
    for path in paths:
        sig = rar_content_signature(path)
        size = os.path.getsize(path)
        groups[sig].append((path, size))

    grouped = []
    for key, files in groups.items():
        files_sorted = sorted(files, key=lambda x: (-x[1], x[0]))
        grouped.append((key, files_sorted))

    grouped.sort(key=lambda g: -len(g[1]))
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Group scrambled multi-part RAR files into sets."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing scrambled files (default: current)",
    )
    parser.add_argument(
        "--write-commands",
        action="store_true",
        help="Write a .txt file per group with suggested extraction command.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.directory)
    print(f"Scanning directory: {root}")

    rar_files = scan_directory(root)
    if not rar_files:
        print("No RAR files detected.")
        return

    print(f"Detected {len(rar_files)} potential RAR parts.")
    groups = group_rars(rar_files)

    for idx, (key, files) in enumerate(groups, start=1):
        kind = key[0]
        print("=" * 60)
        print(f"Set #{idx}: {kind}, parts={len(files)}")
        for path, size in files:
            print(f"  {size:>12} bytes  {os.path.basename(path)}")

        if args.write_commands:
            primary = files[0][0]
            cmd_path = os.path.join(
                root, f"rar_set_{idx:03d}_extract_command.txt"
            )
            content = []
            content.append("# Suggested extraction commands for this set")
            content.append("# Adjust paths / tool (7z or unrar) as needed.")
            content.append("")
            content.append("# Using 7-Zip (CLI):")
            content.append(f'7z x "{os.path.basename(primary)}"')
            content.append("")
            content.append("# Using WinRAR/UnRAR (CLI):")
            content.append(f'unrar x "{os.path.basename(primary)}"')
            with open(cmd_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            print(f"  -> Wrote helper: {os.path.basename(cmd_path)}")

    print("=" * 60)
    print(f"Identified {len(groups)} group(s).")


if __name__ == "__main__":
    main()