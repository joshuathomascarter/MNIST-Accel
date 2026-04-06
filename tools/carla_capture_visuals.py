#!/usr/bin/env python3
"""Capture a real CARLA still image and short clip from a running simulator.

This script is intended for a supported CARLA host, not for the current macOS
development machine. It connects to a running CARLA server, spawns an ego car,
attaches an RGB camera, saves a PNG still, and optionally exports a GIF clip.
"""

from __future__ import annotations

import argparse
import queue
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import carla
except ImportError:  # pragma: no cover - runtime guard for unsupported hosts.
    carla = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture CARLA screenshot and short clip from a running server")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server RPC port")
    parser.add_argument("--traffic-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--town", default="Town05", help="Town to load before capture")
    parser.add_argument("--output-dir", default="docs/figs/carla_real_capture", help="Output directory for captured media")
    parser.add_argument("--still-name", default="carla_town05_real.png", help="PNG still filename")
    parser.add_argument("--clip-name", default="carla_town05_real.gif", help="GIF clip filename")
    parser.add_argument("--frames", type=int, default=90, help="Number of frames to capture into the clip")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup ticks before capture starts")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate for GIF export")
    parser.add_argument("--width", type=int, default=1280, help="RGB camera width")
    parser.add_argument("--height", type=int, default=720, help="RGB camera height")
    parser.add_argument("--fov", type=float, default=100.0, help="RGB camera field of view")
    parser.add_argument("--vehicles", type=int, default=18, help="Number of NPC vehicles to spawn")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for actor selection")
    parser.add_argument("--no-clip", action="store_true", help="Only export a still PNG")
    return parser.parse_args()


def require_carla() -> None:
    if carla is None:
        raise SystemExit(
            "CARLA Python API is not installed. Install the wheel from the CARLA package on a supported Ubuntu/Windows host, then rerun this script."
        )


def make_image(carla_image) -> Image.Image:
    arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_image.height, carla_image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]
    return Image.fromarray(rgb)


def choose_vehicle_blueprint(world):
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    preferred = [bp for bp in blueprints if bp.id.endswith("tesla.model3")]
    pool = preferred or list(blueprints)
    if not pool:
        raise RuntimeError("No vehicle blueprints available in CARLA world")
    blueprint = random.choice(pool)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    return blueprint


def build_camera_blueprint(world, width: int, height: int, fov: float):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    return bp


def spawn_npc_vehicles(client, world, traffic_manager, count: int):
    actors = []
    spawn_points = world.get_map().get_spawn_points()
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    random.shuffle(spawn_points)
    for transform in spawn_points:
        if len(actors) >= count:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        actors.append(actor)
    return actors


def set_sync_mode(world, traffic_manager, fps: int):
    settings = world.get_settings()
    original = carla.WorldSettings(
        no_rendering_mode=settings.no_rendering_mode,
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
        substepping=settings.substepping,
        max_substep_delta_time=settings.max_substep_delta_time,
        max_substeps=settings.max_substeps,
        max_culling_distance=settings.max_culling_distance,
        deterministic_ragdolls=settings.deterministic_ragdolls,
        tile_stream_distance=settings.tile_stream_distance,
        actor_active_distance=settings.actor_active_distance,
        spectator_as_ego=settings.spectator_as_ego,
    )
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(new_settings)
    traffic_manager.set_synchronous_mode(True)
    return original


def restore_sync_mode(world, traffic_manager, original_settings):
    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)


def main() -> None:
    args = parse_args()
    require_carla()
    random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    still_path = output_dir / args.still_name
    clip_path = output_dir / args.clip_name

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    if args.town and not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)

    traffic_manager = client.get_trafficmanager(args.traffic_port)
    traffic_manager.set_random_device_seed(args.seed)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    original_settings = set_sync_mode(world, traffic_manager, args.fps)
    actor_cleanup = []
    sensor_cleanup = []
    image_queue: queue.Queue = queue.Queue()

    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in current CARLA map")

        ego_bp = choose_vehicle_blueprint(world)
        ego = None
        for transform in spawn_points:
            ego = world.try_spawn_actor(ego_bp, transform)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("Unable to spawn ego vehicle")
        ego.set_autopilot(True, traffic_manager.get_port())
        actor_cleanup.append(ego)

        npc_actors = spawn_npc_vehicles(client, world, traffic_manager, args.vehicles)
        actor_cleanup.extend(npc_actors)

        camera_bp = build_camera_blueprint(world, args.width, args.height, args.fov)
        camera_tf = carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12.0))
        camera = world.spawn_actor(camera_bp, camera_tf, attach_to=ego)
        camera.listen(image_queue.put)
        sensor_cleanup.append(camera)

        for _ in range(args.warmup):
            world.tick()
            image_queue.get(timeout=5.0)

        frames = []
        for frame_index in range(max(args.frames, 1)):
            world.tick()
            image = image_queue.get(timeout=5.0)
            pil_image = make_image(image)
            if frame_index == 0:
                pil_image.save(still_path)
            if not args.no_clip:
                frames.append(pil_image)

        if frames and not args.no_clip:
            first, *rest = frames
            first.save(
                clip_path,
                save_all=True,
                append_images=rest,
                duration=int(1000 / max(args.fps, 1)),
                loop=0,
            )

        print(f"Connected town: {world.get_map().name}")
        print(f"Still saved:    {still_path}")
        if not args.no_clip:
            print(f"Clip saved:     {clip_path}")
        print(f"Ego vehicle:    {ego.type_id}")
        print(f"NPC vehicles:   {len(npc_actors)}")

    finally:
        for sensor in sensor_cleanup:
            sensor.stop()
        for actor in reversed(sensor_cleanup + actor_cleanup):
            actor.destroy()
        restore_sync_mode(world, traffic_manager, original_settings)


if __name__ == "__main__":
    main()