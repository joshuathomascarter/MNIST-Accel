#!/usr/bin/env python3
"""Export a compact CARLA scene summary into NoC BEV trace files.

This script does not depend on the CARLA Python package. It consumes a small
JSON scene summary that can be produced from a simulated driving scene or
curated by hand for competition demos. The output matches the compact trace
format consumed by tools/noc_allocator_full_comparison.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_HEAD_TILES = [8, 11, 12, 15]
DEFAULT_FUSION_OVERLAP_EDGES = [
    [5, 6],
    [6, 9],
    [9, 10],
    [10, 5],
    [6, 10],
    [9, 5],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CARLA-style BEV scene summary into NoC trace JSON files")
    parser.add_argument("--input", required=True, help="Path to the CARLA scene summary JSON")
    parser.add_argument("--frame-out", help="Output path for the frame-level BEV trace JSON")
    parser.add_argument("--reduce-out", help="Output path for the reduction-only BEV trace JSON")
    parser.add_argument("--secondary-ratio", type=float, default=None,
                        help="Fallback fraction of tokens that should go to the secondary fusion tile")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def default_outputs(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem.replace("_scene", "")
    return (
        input_path.with_name(f"{stem}_bev_frame.json"),
        input_path.with_name(f"{stem}_bev_reduce.json"),
    )


def secondary_count(total_tokens: int, camera: dict, scale_name: str, default_ratio: float) -> int:
    explicit_secondary = camera.get("secondary_token_counts", {}).get(scale_name)
    if explicit_secondary is not None:
        return max(0, min(total_tokens, int(explicit_secondary)))

    ratio = camera.get("secondary_ratio", default_ratio)
    return max(0, min(total_tokens, int(round(total_tokens * ratio))))


def build_frame_trace(scene: dict, source_name: str, default_secondary_ratio: float | None) -> dict:
    controller = int(scene.get("controller", 0))
    planner_root = int(scene.get("planner_root", 10))
    head_tiles = [int(tile) for tile in scene.get("head_tiles", DEFAULT_HEAD_TILES)]
    cameras = scene["cameras"]
    scales = scene["scales"]
    packets = []

    packets.append({
        "phase": "camera_setup",
        "src": controller,
        "dsts": [int(camera["tile"]) for camera in cameras],
        "is_sparse": False,
        "num_flits": int(scene.get("camera_setup_flits", 10)),
        "release_cycle": int(scene.get("camera_setup_cycle", 0)),
        "pair_release_stride": 1,
    })

    fallback_ratio = float(scene.get("secondary_ratio", 0.25 if default_secondary_ratio is None else default_secondary_ratio))

    for scale in scales:
        scale_name = scale["name"]
        dense_flits = int(scale["dense_flits"])
        release_cycle = int(scale["release_cycle"])
        completion_cycle = int(scale["completion_cycle"])

        packets.append({
            "phase": f"{scale_name}_dense_ingress",
            "src": controller,
            "dsts": [int(camera["tile"]) for camera in cameras],
            "is_sparse": False,
            "num_flits": dense_flits,
            "release_cycle": release_cycle,
            "pair_release_stride": 1,
        })

        for camera in cameras:
            total_tokens = int(camera["token_counts"][scale_name])
            secondary_tokens = secondary_count(total_tokens, camera, scale_name, fallback_ratio)
            primary_tokens = total_tokens - secondary_tokens
            src_tile = int(camera["tile"])
            primary_tile = int(camera["primary_fusion_tile"])
            secondary_tile = int(camera["secondary_fusion_tile"])

            if primary_tokens > 0:
                packets.append({
                    "phase": f"{scale_name}_{camera['name']}_primary",
                    "src": src_tile,
                    "dst": primary_tile,
                    "is_sparse": True,
                    "num_flits": 2,
                    "count": primary_tokens,
                    "release_cycle": release_cycle + 12 + (src_tile % 5),
                    "release_stride": 1,
                })
            if secondary_tokens > 0:
                packets.append({
                    "phase": f"{scale_name}_{camera['name']}_secondary",
                    "src": src_tile,
                    "dst": secondary_tile,
                    "is_sparse": True,
                    "num_flits": 2,
                    "count": secondary_tokens,
                    "release_cycle": release_cycle + 13 + (src_tile % 5),
                    "release_stride": 1,
                })

        packets.append({
            "phase": f"{scale_name}_completion",
            "srcs": [int(camera["tile"]) for camera in cameras],
            "dst": planner_root,
            "is_sparse": True,
            "num_flits": 1,
            "release_cycle": completion_cycle,
            "pair_release_stride": 1,
        })

    overlap_cycle = int(scene.get("fusion_overlap_cycle", int(scales[-1]["completion_cycle"]) + 10))
    for edge_index, edge in enumerate(scene.get("fusion_overlap_edges", DEFAULT_FUSION_OVERLAP_EDGES)):
        packets.append({
            "phase": f"fusion_overlap_{edge[0]}_{edge[1]}",
            "src": int(edge[0]),
            "dst": int(edge[1]),
            "is_sparse": True,
            "num_flits": 2,
            "release_cycle": overlap_cycle + edge_index,
        })

    planner_query_cycle = int(scene.get("planner_query_cycle", overlap_cycle + len(scene.get("fusion_overlap_edges", DEFAULT_FUSION_OVERLAP_EDGES)) + 3))
    head_response_cycle = int(scene.get("head_response_cycle", planner_query_cycle + 4))
    head_weight_cycle = int(scene.get("head_weight_cycle", head_response_cycle + 4))
    head_weight_flits = int(scene.get("head_weight_flits", 8))

    packets.append({
        "phase": "planner_queries",
        "src": planner_root,
        "dsts": head_tiles,
        "is_sparse": True,
        "num_flits": 2,
        "release_cycle": planner_query_cycle,
        "pair_release_stride": 1,
    })
    packets.append({
        "phase": "head_responses",
        "srcs": head_tiles,
        "dst": planner_root,
        "is_sparse": True,
        "num_flits": 1,
        "release_cycle": head_response_cycle,
        "pair_release_stride": 1,
    })
    packets.append({
        "phase": "head_weights",
        "src": controller,
        "dsts": head_tiles,
        "is_sparse": False,
        "num_flits": head_weight_flits,
        "release_cycle": head_weight_cycle,
        "pair_release_stride": 1,
    })

    return {
        "name": f"{scene.get('scene_name', source_name)} BEV frame trace",
        "description": scene.get(
            "description",
            "Trace exported from a compact CARLA scene summary for the competition demo.",
        ),
        "planner_root": planner_root,
        "source_scene": source_name,
        "packets": packets,
    }


def build_reduce_trace(scene: dict, source_name: str) -> dict:
    planner_root = int(scene.get("planner_root", 10))
    cameras = scene["cameras"]
    bev_cells = int(scene.get("bev_cells", 192))
    base_release_cycle = int(scene.get("reduce_release_cycle", 48))
    camera_release_stride = int(scene.get("camera_release_stride", 1))
    reduce_release_stride = int(scene.get("reduce_release_stride", 1))

    packets = []
    for camera_index, camera in enumerate(cameras):
        packets.append({
            "phase": f"{camera['name']}_reduce",
            "src": int(camera["tile"]),
            "dst": planner_root,
            "is_sparse": True,
            "num_flits": 1,
            "count": bev_cells,
            "release_cycle": base_release_cycle + camera_index * camera_release_stride,
            "release_stride": reduce_release_stride,
            "reduce_group_start": 0,
            "reduce_group_stride": 1,
        })

    return {
        "name": f"{scene.get('scene_name', source_name)} BEV reduction trace",
        "description": scene.get(
            "description",
            "Reduction-only trace exported from a compact CARLA scene summary.",
        ),
        "planner_root": planner_root,
        "num_bev_cells": bev_cells,
        "source_scene": source_name,
        "packets": packets,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    frame_default, reduce_default = default_outputs(input_path)
    frame_out = Path(args.frame_out).resolve() if args.frame_out else frame_default
    reduce_out = Path(args.reduce_out).resolve() if args.reduce_out else reduce_default

    scene = load_json(input_path)
    frame_trace = build_frame_trace(scene, input_path.name, args.secondary_ratio)
    reduce_trace = build_reduce_trace(scene, input_path.name)

    dump_json(frame_out, frame_trace)
    dump_json(reduce_out, reduce_trace)

    print(f"Input scene:   {input_path}")
    print(f"Frame trace:   {frame_out}")
    print(f"Reduce trace:  {reduce_out}")
    print(f"Scene name:    {frame_trace['name']}")
    print(f"Cameras:       {len(scene['cameras'])}")
    print(f"BEV cells:     {reduce_trace['num_bev_cells']}")


if __name__ == "__main__":
    main()