import argparse
from pathlib import Path
from PIL import Image


SUPPORTED_EXTS = {
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".tif", ".tiff",
    ".gif",
    ".bmp"
}


def convert_image_to_bmp(input_path: Path, output_path: Path):
    try:
        with Image.open(input_path) as img:
            # 处理 GIF、多帧图像，只取第一帧
            img.seek(0)

            # BMP 不支持透明通道，这里统一转成 RGB
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="BMP")

        print(f"Converted: {input_path} -> {output_path}")

    except Exception as e:
        print(f"Failed: {input_path}, error: {e}")


def convert_single_file(input_path: Path, output_path: Path | None):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported image format: {input_path.suffix}")

    if output_path is None:
        output_path = input_path.with_suffix(".bmp")

    convert_image_to_bmp(input_path, output_path)


def convert_directory(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTS:
            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".bmp")
            convert_image_to_bmp(img_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert common image formats to BMP."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input image file or input directory."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output BMP file or output directory."
    )

    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Convert all supported images in a directory."
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.directory:
        output_dir = Path(args.output) if args.output else input_path / "bmp_output"
        convert_directory(input_path, output_dir)
    else:
        output_path = Path(args.output) if args.output else None
        convert_single_file(input_path, output_path)


if __name__ == "__main__":
    main()