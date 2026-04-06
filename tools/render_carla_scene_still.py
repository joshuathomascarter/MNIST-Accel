#!/usr/bin/env python3
"""Render a competition-ready still image from a CARLA scene summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
from matplotlib import font_manager


CANVAS_W = 1600
CANVAS_H = 900


CAMERA_COLORS = {
    "front": "#ffb703",
    "front_left": "#fb8500",
    "front_right": "#8ecae6",
    "rear_left": "#90be6d",
    "rear": "#577590",
    "rear_right": "#f28482",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a stylized still image for the CARLA BEV demo")
    parser.add_argument("--input", required=True, help="Path to the CARLA scene summary JSON")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--variant", choices=["hero", "annotated"], default="annotated",
                        help="Hero mode is clean for a title slide; annotated mode adds technical overlays")
    return parser.parse_args()


def load_scene(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVu Sans:bold" if bold else "DejaVu Sans"
    try:
        font_path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=True)
        return ImageFont.truetype(font_path, size=size)
    except Exception:
        return ImageFont.load_default()


def rgba(hex_color: str, alpha: int = 255):
    rgb = ImageColor.getrgb(hex_color)
    return (rgb[0], rgb[1], rgb[2], alpha)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int):
    words = text.split()
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        width = draw.textbbox((0, 0), trial, font=font)[2]
        if current and width > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_vertical_gradient(draw: ImageDraw.ImageDraw, top: str, bottom: str):
    top_rgb = ImageColor.getrgb(top)
    bottom_rgb = ImageColor.getrgb(bottom)
    for y in range(CANVAS_H):
        t = y / max(CANVAS_H - 1, 1)
        color = tuple(int(top_rgb[idx] * (1 - t) + bottom_rgb[idx] * t) for idx in range(3))
        draw.line([(0, y), (CANVAS_W, y)], fill=color)


def draw_buildings(draw: ImageDraw.ImageDraw):
    building_specs = [
        (70, 70, 470, 270, "#22333b"),
        (1130, 70, 1530, 270, "#2a4d69"),
        (70, 630, 470, 830, "#3d405b"),
        (1130, 630, 1530, 830, "#283618"),
    ]
    for x0, y0, x1, y1, color in building_specs:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=36, fill=color, outline=rgba("#f4f1de", 60), width=3)
        for row in range(y0 + 28, y1 - 20, 34):
            for col in range(x0 + 24, x1 - 18, 42):
                draw.rounded_rectangle((col, row, col + 18, row + 14), radius=4, fill=rgba("#fefae0", 46))


def draw_roads(base: Image.Image):
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((610, 0, 990, CANVAS_H), fill="#2d3142")
    draw.rectangle((0, 290, CANVAS_W, 610), fill="#2d3142")
    draw.rectangle((650, 0, 950, CANVAS_H), fill="#3a3f58")
    draw.rectangle((0, 330, CANVAS_W, 570), fill="#3a3f58")

    sidewalk = rgba("#c8d5b9", 255)
    draw.rectangle((560, 0, 610, CANVAS_H), fill=sidewalk)
    draw.rectangle((990, 0, 1040, CANVAS_H), fill=sidewalk)
    draw.rectangle((0, 240, CANVAS_W, 290), fill=sidewalk)
    draw.rectangle((0, 610, CANVAS_W, 660), fill=sidewalk)

    lane_yellow = rgba("#ffd166", 255)
    lane_white = rgba("#f8f9fa", 185)
    for y in range(40, CANVAS_H, 90):
        draw.rounded_rectangle((792, y, 808, y + 50), radius=5, fill=lane_yellow)
    for x in range(50, CANVAS_W, 110):
        draw.rounded_rectangle((x, 442, x + 64, 458), radius=5, fill=lane_white)

    for offset in range(0, 180, 24):
        draw.rectangle((560 + offset, 260, 572 + offset, 320), fill=rgba("#ffffff", 205))
        draw.rectangle((560 + offset, 580, 572 + offset, 640), fill=rgba("#ffffff", 205))
    for offset in range(0, 180, 24):
        draw.rectangle((1280, 260 + offset, 1340, 272 + offset), fill=rgba("#ffffff", 205))
        draw.rectangle((260, 260 + offset, 320, 272 + offset), fill=rgba("#ffffff", 205))


def draw_vehicle(base: Image.Image, center, size, angle_deg, body_color, outline="#0b0f14"):
    width, height = size
    vehicle = Image.new("RGBA", (int(width * 2), int(height * 2)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vehicle, "RGBA")
    x0 = vehicle.width / 2 - width / 2
    y0 = vehicle.height / 2 - height / 2
    x1 = vehicle.width / 2 + width / 2
    y1 = vehicle.height / 2 + height / 2
    draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=body_color, outline=outline, width=4)
    draw.rounded_rectangle((x0 + 10, y0 + 10, x1 - 10, y0 + 30), radius=10, fill=rgba("#dce6f2", 200))
    draw.rounded_rectangle((x0 + 12, y1 - 28, x1 - 12, y1 - 10), radius=8, fill=rgba("#0f172a", 120))
    rotated = vehicle.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    px = int(center[0] - rotated.width / 2)
    py = int(center[1] - rotated.height / 2)
    base.alpha_composite(rotated, (px, py))


def draw_tree(draw: ImageDraw.ImageDraw, center_x, center_y):
    draw.ellipse((center_x - 18, center_y - 18, center_x + 18, center_y + 18), fill="#6a994e", outline="#386641")
    draw.rectangle((center_x - 4, center_y + 10, center_x + 4, center_y + 28), fill="#7f5539")


def draw_camera_fovs(base: Image.Image, scene: dict, ego_center):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    direction_map = {
        "front": (-90, 36, 230),
        "front_left": (-132, 34, 210),
        "front_right": (-48, 34, 210),
        "rear_left": (150, 32, 190),
        "rear": (90, 30, 170),
        "rear_right": (32, 32, 190),
    }

    cameras = scene.get("cameras", [])
    max_tokens = max(sum(cam["token_counts"].values()) for cam in cameras)
    for camera in cameras:
        name = camera["name"]
        center_angle, spread, radius = direction_map.get(name, (-90, 30, 180))
        total_tokens = sum(camera["token_counts"].values())
        radius = int(radius + 60 * (total_tokens / max_tokens))
        pts = [ego_center]
        for delta in (-spread, 0, spread):
            angle = math.radians(center_angle + delta)
            pts.append((
                ego_center[0] + radius * math.cos(angle),
                ego_center[1] + radius * math.sin(angle),
            ))
        color = CAMERA_COLORS.get(name, "#ffffff")
        draw.polygon(pts, fill=rgba(color, 58), outline=rgba(color, 155))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=4))
    base.alpha_composite(overlay)


def draw_mesh_inset(base: Image.Image, scene: dict):
    draw = ImageDraw.Draw(base, "RGBA")
    x0, y0, x1, y1 = 1160, 80, 1520, 420
    draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill=rgba("#0b132b", 220), outline=rgba("#ffffff", 80), width=2)
    title_font = get_font(26, bold=True)
    small_font = get_font(18)
    draw.text((x0 + 22, y0 + 16), "4x4 Mesh Mapping", font=title_font, fill="#f8fafc")

    camera_tiles = {int(camera["tile"]) for camera in scene["cameras"]}
    fusion_tiles = set()
    for camera in scene["cameras"]:
        fusion_tiles.add(int(camera["primary_fusion_tile"]))
        fusion_tiles.add(int(camera["secondary_fusion_tile"]))
    head_tiles = {int(tile) for tile in scene.get("head_tiles", [])}
    root_tile = int(scene.get("planner_root", 10))

    grid_x, grid_y = x0 + 28, y0 + 70
    cell = 68
    for row in range(4):
        for col in range(4):
            node = row * 4 + col
            fill = "#1f2937"
            if node == 0:
                fill = "#6d597a"
            if node in camera_tiles:
                fill = "#bc6c25"
            if node in fusion_tiles:
                fill = "#2a9d8f"
            if node in head_tiles:
                fill = "#355070"
            if node == root_tile:
                fill = "#d62828"
            cx0 = grid_x + col * cell
            cy0 = grid_y + row * cell
            draw.rounded_rectangle((cx0, cy0, cx0 + 52, cy0 + 52), radius=10, fill=fill, outline=rgba("#ffffff", 55), width=2)
            draw.text((cx0 + 15, cy0 + 11), str(node), font=get_font(20, bold=True), fill="#ffffff")

    legend = [
        ("cam", "#bc6c25"),
        ("bev", "#2a9d8f"),
        ("head", "#355070"),
        ("root", "#d62828"),
    ]
    ly = y1 - 50
    lx = x0 + 24
    for idx, (name, color) in enumerate(legend):
        offset = idx * 82
        draw.rounded_rectangle((lx + offset, ly, lx + offset + 20, ly + 20), radius=5, fill=color)
        draw.text((lx + offset + 28, ly - 2), name, font=small_font, fill="#f8fafc")


def draw_info_panel(base: Image.Image, scene: dict):
    draw = ImageDraw.Draw(base, "RGBA")
    title_font = get_font(40, bold=True)
    sub_font = get_font(24)
    stat_font = get_font(22, bold=True)
    small_font = get_font(18)

    draw.rounded_rectangle((56, 48, 650, 204), radius=28, fill=rgba("#081c15", 208), outline=rgba("#ffffff", 72), width=2)
    lines = wrap_text(draw, scene["scene_name"], title_font, 520)
    title_y = 70
    for idx, line in enumerate(lines[:2]):
        draw.text((84, title_y + idx * 44), line, font=title_font, fill="#f8fafc")
    subtitle_y = 132 if len(lines) == 1 else 168
    draw.text((86, subtitle_y), "Trace-driven BEV fusion demo visual", font=sub_font, fill=rgba("#d8f3dc", 230))

    stats = [
        ("cameras", str(len(scene["cameras"]))),
        ("BEV cells", str(scene.get("bev_cells", 192))),
        ("fusion root", str(scene.get("planner_root", 10))),
    ]
    panel_y = 742
    for idx, (label, value) in enumerate(stats):
        sx0 = 60 + idx * 214
        sx1 = sx0 + 192
        draw.rounded_rectangle((sx0, panel_y, sx1, panel_y + 104), radius=24, fill=rgba("#111827", 214), outline=rgba("#ffffff", 52), width=2)
        draw.text((sx0 + 18, panel_y + 18), label, font=small_font, fill=rgba("#cbd5e1", 225))
        draw.text((sx0 + 18, panel_y + 48), value, font=stat_font, fill="#ffffff")

    callout = (
        "Competition use: place this still on the left of the slide, then\n"
        "show baseline vs INR packet-collapse numbers on the right."
    )
    draw.rounded_rectangle((980, 742, 1530, 842), radius=24, fill=rgba("#14213d", 214), outline=rgba("#ffffff", 52), width=2)
    draw.text((1006, 770), callout, font=small_font, fill="#f8fafc")


def draw_hero_panel(base: Image.Image, scene: dict):
    draw = ImageDraw.Draw(base, "RGBA")
    title_font = get_font(52, bold=True)
    sub_font = get_font(26)
    chip_font = get_font(18, bold=True)
    draw.rounded_rectangle((64, 64, 670, 236), radius=30, fill=rgba("#03170f", 214), outline=rgba("#ffffff", 84), width=2)

    lines = wrap_text(draw, scene["scene_name"], title_font, 540)
    title_y = 92
    for idx, line in enumerate(lines[:2]):
        draw.text((92, title_y + idx * 56), line, font=title_font, fill="#f8fafc")
    subtitle_y = 154 if len(lines) == 1 else 206
    draw.text((94, subtitle_y), "Aggregation-aware NoC demo visual", font=sub_font, fill=rgba("#d8f3dc", 235))

    chip_x = 92
    for label in ["6 cameras", "BEV fusion", "4x4 mesh"]:
        width = draw.textbbox((0, 0), label, font=chip_font)[2] + 34
        draw.rounded_rectangle((chip_x, 786, chip_x + width, 824), radius=18, fill=rgba("#081c15", 208), outline=rgba("#ffffff", 60), width=2)
        draw.text((chip_x + 17, 796), label, font=chip_font, fill="#f8fafc")
        chip_x += width + 14


def draw_annotations(base: Image.Image, scene: dict):
    draw = ImageDraw.Draw(base, "RGBA")
    label_font = get_font(20, bold=True)
    small_font = get_font(16)

    annotations = [
        ((790, 662), (612, 704), "ego vehicle", "camera FOV origin"),
        ((910, 468), (1028, 356), "fusion hotspot", "traffic converges near root"),
        ((1226, 246), (1006, 286), "mesh inset", "camera/BEV/head tile mapping"),
    ]

    for start, end, title, subtitle in annotations:
        draw.line([start, end], fill=rgba("#fefae0", 230), width=4)
        draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill="#fefae0")
        box_w = max(draw.textbbox((0, 0), title, font=label_font)[2], draw.textbbox((0, 0), subtitle, font=small_font)[2]) + 26
        box_h = 58
        x0, y0 = end
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=18, fill=rgba("#111827", 220), outline=rgba("#ffffff", 62), width=2)
        draw.text((x0 + 14, y0 + 10), title, font=label_font, fill="#f8fafc")
        draw.text((x0 + 14, y0 + 33), subtitle, font=small_font, fill=rgba("#cbd5e1", 220))


def render_scene(scene: dict, variant: str = "annotated") -> Image.Image:
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), "#000000")
    draw = ImageDraw.Draw(image, "RGBA")
    draw_vertical_gradient(draw, "#93c5fd", "#386641")
    draw_buildings(draw)
    draw_roads(image)

    for pos in [(520, 120), (1080, 120), (520, 760), (1080, 760), (1220, 560), (380, 560)]:
        draw_tree(draw, *pos)

    ego_center = (800, 700)
    draw_camera_fovs(image, scene, ego_center)

    vehicle_specs = [
        (ego_center, (82, 152), 0, "#3a86ff"),
        ((800, 150), (82, 152), 180, "#ef476f"),
        ((1180, 450), (82, 152), 90, "#ffbe0b"),
        ((430, 450), (82, 152), 270, "#2ec4b6"),
        ((690, 450), (82, 152), 270, "#adb5bd"),
        ((980, 450), (82, 152), 90, "#84a59d"),
        ((800, 520), (82, 152), 0, "#6c757d"),
    ]
    for spec in vehicle_specs:
        draw_vehicle(image, *spec)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    overlay_alpha = 12 if variant == "hero" else 18
    overlay_draw.rectangle((0, 0, CANVAS_W, CANVAS_H), fill=rgba("#000000", overlay_alpha))
    image.alpha_composite(overlay)

    if variant == "hero":
        draw_hero_panel(image, scene)
    else:
        draw_mesh_inset(image, scene)
        draw_info_panel(image, scene)
        draw_annotations(image, scene)
    return image.convert("RGB")


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = load_scene(input_path)
    image = render_scene(scene, variant=args.variant)
    image.save(output_path, format="PNG")
    print(f"Input scene:  {input_path}")
    print(f"Output still: {output_path}")
    print(f"Scene name:   {scene['scene_name']}")
    print(f"Variant:      {args.variant}")


if __name__ == "__main__":
    main()