from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


INPUT = Path("assets/portrait.png")
OUTPUT = Path("portrait.svg")

COLS = 90

# Dark → light
RAMP = "@%#cs*+=-:.` "


def escape_xml(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Image not found: {INPUT}")

    print(f"Loading: {INPUT}")

    # ---------------------------------------------------------
    # 1. Load image
    # ---------------------------------------------------------

    image = Image.open(INPUT).convert("RGBA")

    # ---------------------------------------------------------
    # 2. Remove background
    # ---------------------------------------------------------

    print("Removing background...")

    foreground = remove(
        image,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )

    # ---------------------------------------------------------
    # 3. Crop to the detected subject
    # ---------------------------------------------------------

    alpha = foreground.getchannel("A")

    bbox = alpha.getbbox()

    if bbox:
        left, top, right, bottom = bbox

        # Add a little breathing room
        pad_x = int((right - left) * 0.08)
        pad_y = int((bottom - top) * 0.08)

        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(foreground.width, right + pad_x)
        bottom = min(foreground.height, bottom + pad_y)

        foreground = foreground.crop(
            (left, top, right, bottom)
        )

    # ---------------------------------------------------------
    # 4. Put subject on WHITE background
    # ---------------------------------------------------------

    background = Image.new(
        "RGBA",
        foreground.size,
        (255, 255, 255, 255),
    )

    background.alpha_composite(foreground)

    rgb = background.convert("RGB")

    # ---------------------------------------------------------
    # 5. Convert to grayscale
    # ---------------------------------------------------------

    img = np.array(rgb)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    # ---------------------------------------------------------
    # 6. Resize for monospace characters
    # ---------------------------------------------------------

    h, w = gray.shape

    rows = max(
        1,
        int(COLS * (h / w) * 0.48)
    )

    gray = cv2.resize(
        gray,
        (COLS, rows),
        interpolation=cv2.INTER_AREA,
    )

    # ---------------------------------------------------------
    # 7. Smooth noise but preserve edges
    # ---------------------------------------------------------

    gray = cv2.bilateralFilter(
        gray,
        7,
        50,
        50,
    )

    # ---------------------------------------------------------
    # 8. Local contrast
    # ---------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    # ---------------------------------------------------------
    # 9. Darken midtones
    # ---------------------------------------------------------

    normalized = gray.astype(np.float32) / 255.0

    normalized = normalized ** 1.7

    gray = (
        normalized * 255
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # 10. Convert brightness → ASCII
    # ---------------------------------------------------------

    ascii_rows = []

    for row in gray:

        line = ""

        for value in row:

            index = int(
                (value / 255)
                * (len(RAMP) - 1)
            )

            line += RAMP[index]

        ascii_rows.append(line)

    # ---------------------------------------------------------
    # 11. Generate SVG
    # ---------------------------------------------------------

    font_size = 12.9
    char_width = 7.74
    line_height = 15

    width = COLS * char_width
    height = rows * line_height

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',

        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.0f}" '
            f'height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}">'
        ),

        '<rect width="100%" height="100%" fill="white"/>',

        (
            f'<g font-family="JetBrains Mono, '
            f'Liberation Mono, monospace" '
            f'font-size="{font_size}px" '
            f'fill="#111">'
        ),
    ]

    for i, line in enumerate(ascii_rows):

        y = (i + 1) * line_height

        svg.append(
            f'<text x="0" y="{y}">'
            f'{escape_xml(line)}'
            f'</text>'
        )

    svg.extend([
        "</g>",
        "</svg>",
    ])

    OUTPUT.write_text(
        "\n".join(svg),
        encoding="utf-8",
    )

    print()
    print("====================================")
    print("Portrait generated successfully")
    print("====================================")
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Grid  : {COLS} × {rows}")
    print()


if __name__ == "__main__":
    main()
