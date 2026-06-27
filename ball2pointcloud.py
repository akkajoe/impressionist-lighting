#!/usr/bin/env python3
import os
import time
import json
import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
import skimage
import torch
from tqdm.auto import tqdm

# optional EXR support
try:
    import ezexr
except Exception:
    ezexr = None

# optional fast JSON
try:
    import orjson
    _HAS_ORJSON = True
except Exception:
    _HAS_ORJSON = False


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_dir", type=str, required=True,
                        help="directory that contains the chrome ball images")
    parser.add_argument("--envmap_dir", type=str, required=True,
                        help="directory to output environment maps and *_points.json(.gz)")
    parser.add_argument("--envmap_height", type=int, default=256,
                        help="environment map height in pixels (lat-long; width=2*height)")
    parser.add_argument("--scale", type=int, default=4,
                        help="scale factor applied before downsampling envmap")
    parser.add_argument("--threads", type=int, default=8,
                        help="num processes for parallel processing")

    # Sphere geometry for 3D positions
    parser.add_argument("--sphere_centre", type=str, default="0, 0, 0",
                        help="Sphere centre (cx, cy, cz) in camera coords")
    parser.add_argument("--sphere_radius", type=float, default=0.1,
                        help="Chrome ball sphere radius")

    # gzip toggle
    parser.add_argument("--zip_json", action="store_true",
                        help="gzip the JSON")

    # single-file path
    parser.add_argument("--file", type=str, default=None,
                        help="If set, process only this single file inside --ball_dir")

    # limit number of files
    parser.add_argument("--max_files", type=int, default=0,
                        help="limit number of files from directory to process (0 = all)")
    return parser


def create_envmap_grid(size: int):
    """
    BLENDER CONVENTION
    Create the grid of environment map (lat-long) that contains the position in spherical coordinates.

    Returns:
      theta_phi: (size, 2*size, 2) with [theta, phi] in radians
    """
    theta = torch.linspace(0, 2 * np.pi, size * 2) # width dimension (0..2pi)
    phi = torch.linspace(0, np.pi, size) # height dimension (0..pi)
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    return theta_phi.numpy()


def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    From reflection relation, mirror normal aligns with normalize(I + R).

    incoming_vector (I): vector from surface point toward the camera
    reflect_vector (R): vector from surface point toward the light (surface->light)
    """
    v = incoming_vector + reflect_vector
    N = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    return N


def get_cartesian_from_spherical(theta: np.array, phi: np.array, r=1.0):
    """
    theta: horizontal angle in [0, 2pi]
    phi: vertical angle in [0, pi]
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def luminance_linear(env):
    # env: (H, W, 3) linear RGB
    return 0.2126 * env[..., 0] + 0.7152 * env[..., 1] + 0.0722 * env[..., 2]


def process_image(args: argparse.Namespace, file_name: str):
    I = np.array([1.0, 0.0, 0.0], dtype=np.float32) # camera looks along +X (convention)

    # Decide JSON path first and skip only if JSON exists
    base_no_ext = os.path.splitext(file_name)[0]
    base = base_no_ext + "_points"
    points_out_path = os.path.join(
        args.envmap_dir,
        base + (".json.gz" if args.zip_json else ".json")
    )
    if os.path.exists(points_out_path):
        # Already processed
        return None

    # Envmap image output
    envmap_output_path = os.path.join(args.envmap_dir, file_name)

    # Read ball image
    ball_path = os.path.join(args.ball_dir, file_name)
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".exr":
        if ezexr is None:
            print(f"[skip] {file_name}: ezexr not available to read EXR")
            return None
        try:
            ball_image = ezexr.imread(ball_path)  # expected linear
        except Exception as e:
            print(f"[skip] {file_name}: cannot read EXR ({e})")
            return None
        if ball_image.ndim == 2:
            ball_image = np.stack([ball_image] * 3, axis=-1)
        if ball_image.shape[-1] == 4:
            ball_image = ball_image[..., :3]
        ball_image = ball_image.astype(np.float32)
    else:
        try:
            ball_image = skimage.io.imread(ball_path)
            ball_image = skimage.img_as_float(ball_image).astype(np.float32)
        except Exception as e:
            print(f"[skip] {file_name}: cannot read image ({e})")
            return None
        if ball_image.ndim == 2:
            ball_image = np.stack([ball_image] * 3, axis=-1)
        if ball_image.shape[-1] == 4:
            ball_image = ball_image[..., :3]

    H_ball, W_ball = ball_image.shape[:2]

    # Env grid and vectors
    Hs = args.envmap_height * args.scale
    env_grid = create_envmap_grid(Hs) # (Hs, 2Hs, 2) -> theta, phi
    reflect_vec = get_cartesian_from_spherical(
        env_grid[..., 0], env_grid[..., 1]
    ).astype(np.float32) # (Hs, 2Hs, 3)
    normal = get_normal_vector(I[None, None, :], reflect_vec).astype(np.float32)

    # positions for grid_sample [0,1] -> [-1,1], using y,z
    pos = (normal + 1.0) / 2.0
    pos = 1.0 - pos
    pos = pos[..., 1:] # (Hs, 2Hs, 2)  # pos[...,0]=u, pos[...,1]=v in [0,1]

    # compute which ball-image pixel each env sample came from
    u = pos[..., 0]  # x in [0,1]
    v = pos[..., 1]  # y in [0,1]
    col = u * (W_ball - 1)
    row = v * (H_ball - 1)
    ball_ij = np.stack([row, col], axis=-1).astype(np.float32) # (Hs, 2Hs, 2)

    # Bilinear sampling to build env map
    with torch.no_grad():
        grid = torch.from_numpy(pos)[None].float() # [1, Hs, 2Hs, 2]
        grid = grid * 2 - 1 # [-1, 1]
        ball_t = torch.from_numpy(ball_image[None]).float() # [1, H, W, C]
        ball_t = ball_t.permute(0, 3, 1, 2) # [1, 3, H, W]
        env_t = torch.nn.functional.grid_sample(
            ball_t,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        env_map = (
            env_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        )  # (Hs, 2Hs, 3)

    # Core fields
    Hs, Ws = env_map.shape[:2]
    N = int(Hs * Ws)

    cx, cy, cz = [float(v) for v in args.sphere_centre.split(",")]
    c = np.array([cx, cy, cz], dtype=np.float32)
    r = float(args.sphere_radius)

    # Required fields (float32)
    incident = (-reflect_vec).astype(np.float32) # light to surface (Hs, Ws, 3)
    normals = normal.astype(np.float32) # (Hs, Ws, 3)
    X = (c[None, None, :] + r * normals).astype(np.float32) # (Hs, Ws, 3)
    luminance = luminance_linear(env_map).astype(np.float32) # (Hs, Ws)

    # Flatten to JSON-friendly lists
    dirs_light_to_surface = incident.reshape(-1, 3).tolist()
    normals_flat = normals.reshape(-1, 3).tolist()
    X_flat = X.reshape(-1, 3).tolist()
    luminance_flat = luminance.reshape(-1).tolist()
    ball_ij_flat = ball_ij.reshape(-1, 2).tolist()

    # JSON payload
    payload = {
        "envmap_shape": [int(Hs), int(Ws)],
        "ball_shape": [int(H_ball), int(W_ball)],
        "num_points": int(N),
        "sphere_centre": [cx, cy, cz],
        "sphere_radius": float(r),
        "dirs_light_to_surface": dirs_light_to_surface,
        "normals": normals_flat,
        "X": X_flat,
        "luminance": luminance_flat,
        "ball_ij": ball_ij_flat,
    }

    # JSON SAVE
    try:
        if _HAS_ORJSON:
            data_bin = orjson.dumps(payload)
            if args.zip_json:
                import gzip
                with gzip.open(points_out_path, "wb", compresslevel=1) as f:
                    f.write(data_bin)
            else:
                with open(points_out_path, "wb") as f:
                    f.write(data_bin)
        else:
            # Fallback to standard json
            if args.zip_json:
                import gzip
                with gzip.open(points_out_path, "wt", encoding="utf-8") as f:
                    json.dump(payload, f)
            else:
                with open(points_out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

        print(f"[ok] {file_name}: dumped {N:,} points to {points_out_path}")
    except Exception as e:
        print(f"[warning] Could not save JSON for {file_name}: {e}")

    # Optional: save a downsampled envmap image
    env_map_default = skimage.transform.resize(
        env_map,
        (args.envmap_height, args.envmap_height * 2),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    try:
        if ext == ".exr":
            if ezexr is None:
                print(f"[warn] {file_name}: cannot write EXR (ezexr missing), skipping env image")
            else:
                pass # skipping envmap image save
        else:
            env8 = skimage.img_as_ubyte(np.clip(env_map_default, 0.0, 1.0))
            pass  # skipping envmap image save
    except Exception as e:
        print(f"[warn] {file_name}: could not save env image ({e})")

    return None


def main():
    start_time = time.time()
    args = create_argparser().parse_args()

    os.makedirs(args.envmap_dir, exist_ok=True)

    # single-file path
    if args.file is not None:
        process_image(args, args.file)
        print("TOTAL TIME:", time.time() - start_time)
        return

    # multi-file path
    valid_ext = {".png", ".jpg", ".jpeg", ".exr"}
    files = sorted(
        f for f in os.listdir(args.ball_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    )

    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    process_func = partial(process_image, args)

    t = max(1, int(args.threads))
    if t == 1:
        for f in tqdm(files, total=len(files)):
            process_image(args, f)
    else:
        with Pool(t) as p:
            list(tqdm(p.imap(process_func, files), total=len(files)))

    print("TOTAL TIME:", time.time() - start_time)


if __name__ == "__main__":
    main()

