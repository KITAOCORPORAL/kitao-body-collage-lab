"""Preview rendering for element masks and contact sheets."""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_io import pil_to_comfy_image

COLORS = [
    (235, 64, 52),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
    (231, 76, 120),
]


def render_element_preview(image, elements):
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    overlay = base.copy()
    for index, item in enumerate(elements):
        mask = np.asarray(item["mask"], dtype=bool)
        overlay[mask] = COLORS[index % len(COLORS)]
    composed = cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
    preview = Image.fromarray(composed, mode="RGB")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for index, item in enumerate(elements):
        color = COLORS[index % len(COLORS)]
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 600))
        text = f'{item["id"]} {float(item["confidence"]):.2f}'
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_y = max(0, y1 - (text_bbox[3] - text_bbox[1]) - 4)
        draw.rectangle((x1, text_y, x1 + text_bbox[2] - text_bbox[0] + 6, y1), fill=color)
        draw.text((x1 + 3, text_y + 1), text, fill=(0, 0, 0), font=font)
    return pil_to_comfy_image(preview)


def render_mask_contact_sheet(elements, tile_size=256):
    count = max(1, len(elements))
    columns = min(4, max(1, math.ceil(math.sqrt(count))))
    rows = math.ceil(count / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "#181818")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    if not elements:
        draw.text((16, 16), "No masks detected", fill="white", font=font)
        return pil_to_comfy_image(sheet)

    for index, item in enumerate(elements):
        mask = Image.fromarray(np.asarray(item["mask"], dtype=np.uint8) * 255, mode="L")
        mask.thumbnail((tile_size - 16, tile_size - 36), Image.Resampling.NEAREST)
        x = (index % columns) * tile_size + (tile_size - mask.width) // 2
        y = (index // columns) * tile_size + 8
        colored = Image.new("RGB", mask.size, COLORS[index % len(COLORS)])
        sheet.paste(colored, (x, y), mask)
        label = f'{item["id"]}  area={item["area"]}'
        draw.text(((index % columns) * tile_size + 8, (index // columns + 1) * tile_size - 24), label, fill="white", font=font)
    return pil_to_comfy_image(sheet)
