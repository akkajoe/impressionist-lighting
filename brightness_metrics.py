import os
import numpy as np
import skimage.io
import skimage.util
import torch
import pandas as pd

def create_envmap_grid(size):
    theta = torch.linspace(0, 2 * np.pi, size * 2)
    phi = torch.linspace(0, np.pi, size)
    theta, phi = torch.meshgrid(theta, phi, indexing='xy')
    return torch.cat([theta[..., None], phi[..., None]], dim=-1).numpy()

def get_cartesian_from_spherical(theta, phi, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def get_normal_vector(I, R):
    v = I + R
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def luminance_linear(env):
    return 0.2126*env[...,0] + 0.7152*env[...,1] + 0.0722*env[...,2]

def compute_diffusion_metrics(env_map, reflect_vec, top_percentile=95.0):
    L = luminance_linear(env_map)
    peak = float(L.max())
    mean = float(L.mean()) + 1e-12
    peak_mean_ratio = peak / mean
    total = float(L.sum()) + 1e-12
    thresh = np.percentile(L, top_percentile)
    mask = L >= thresh
    concentration = float(L[mask].sum()) / total
    bright_dirs = reflect_vec[mask]
    if len(bright_dirs) > 1:
        w = L[mask]
        w = w / (w.sum() + 1e-12)
        mean_dir = (bright_dirs * w[:, None]).sum(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
        dots = np.clip((bright_dirs * mean_dir[None]).sum(axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        angular_spread = float(np.std(angles))
    else:
        angular_spread = 0.0
    w3 = (L * mask)[..., None]
    vec = (reflect_vec * w3).reshape(-1, 3).sum(axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    az = float(np.degrees(np.arctan2(vec[1], vec[0])))
    el = float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0))))
    return {
        "peak_mean_ratio" : round(peak_mean_ratio, 3),
        "brightness_concentration": round(concentration, 3),
        "angular_spread_deg" : round(angular_spread, 3),
        "dominant_elevation_deg" : round(el, 3),
        "dominant_azimuth_deg" : round(az, 3),
    }

def process_ball(ball_path, envmap_height=128, scale=2, top_percentile=95.0):
    ball_image = skimage.io.imread(ball_path)
    ball_image = skimage.util.img_as_float(ball_image).astype(np.float32)
    if ball_image.ndim == 2:
        ball_image = np.stack([ball_image]*3, axis=-1)
    if ball_image.shape[-1] == 4:
        ball_image = ball_image[..., :3]
    srgb = np.clip(ball_image, 0.0, 1.0)
    ball_image = np.where(
        srgb <= 0.04045, srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)
    Hs = envmap_height * scale
    I  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env_grid = create_envmap_grid(Hs)
    reflect_vec = get_cartesian_from_spherical(env_grid[...,0], env_grid[...,1]).astype(np.float32)
    normal = get_normal_vector(I[None, None, :], reflect_vec)
    pos = (normal + 1.0) / 2.0
    pos = 1.0 - pos
    pos = pos[..., 1:]
    with torch.no_grad():
        grid   = torch.from_numpy(pos)[None].float() * 2 - 1
        ball_t = torch.from_numpy(ball_image[None]).permute(0,3,1,2).float()
        env_t  = torch.nn.functional.grid_sample(
            ball_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
        env_map = env_t[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    return compute_diffusion_metrics(env_map, reflect_vec, top_percentile)

def collect_all_ball_images(root_dir):
    valid_ext = {'.png', '.jpg', '.jpeg'}
    records = []

    # Try standard nested structure: root/artist/scene/square/*.png
    found_nested = False
    for artist in sorted(os.listdir(root_dir)):
        artist_path = os.path.join(root_dir, artist)
        if not os.path.isdir(artist_path): continue
        for scene in sorted(os.listdir(artist_path)):
            scene_path = os.path.join(artist_path, scene)
            if not os.path.isdir(scene_path): continue
            square_path = os.path.join(scene_path, 'square')
            if not os.path.isdir(square_path): continue
            for fname in sorted(os.listdir(square_path)):
                if os.path.splitext(fname)[1].lower() not in valid_ext: continue
                records.append({
                    'path' : os.path.join(square_path, fname),
                    'artist': artist,
                    'scene' : scene.lower(),
                    'file' : fname,
                })
                found_nested = True

    if found_nested:
        return records

    # Fallback: flat structure-> root/square/*.png
    square_path = os.path.join(root_dir, 'square')
    if os.path.isdir(square_path):
        for fname in sorted(os.listdir(square_path)):
            if os.path.splitext(fname)[1].lower() not in valid_ext: continue
            records.append({
                'path' : os.path.join(square_path, fname),
                'artist': 'unknown',
                'scene' : 'unknown',
                'file' : fname,
            })

    return records

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--case_studies", nargs="+", default=[])
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.root_dir
    os.makedirs(out_dir, exist_ok=True)

    records = collect_all_ball_images(args.root_dir)
    print(f"\nFound {len(records)} ball images across all artists and scenes.")

    all_results = []
    for i, rec in enumerate(records):
        try:
            m = process_ball(rec['path'], top_percentile=args.percentile)
            m['file'] = rec['file']
            m['artist'] = rec['artist']
            m['scene'] = rec['scene']
            all_results.append(m)
            if (i+1) % 100 == 0:
                print(f" Processed {i+1} / {len(records)}...")
        except Exception as e:
            print(f" [skip] {rec['file']}: {e}")

    df_all = pd.DataFrame(all_results)

    print("\nDATASET STATISTICS")
    for col in ['peak_mean_ratio','brightness_concentration','angular_spread_deg']:
        print(f"\n  {col}:")
        print(f" mean : {df_all[col].mean():.3f}")
        print(f" std : {df_all[col].std():.3f}")
        print(f" 25th% : {df_all[col].quantile(0.25):.3f}")
        print(f" median : {df_all[col].median():.3f}")
        print(f" 75th% : {df_all[col].quantile(0.75):.3f}")

    print("\nPER ARTIST: mean metrics")
    artist_stats = (
        df_all.groupby('artist')[['peak_mean_ratio','brightness_concentration','angular_spread_deg']]
        .mean().round(3).sort_values('peak_mean_ratio')
    )
    print(artist_stats.to_string())

    print("\nINDOOR vs OUTDOOR: mean metrics")
    scene_stats = (
        df_all.groupby('scene')[['peak_mean_ratio','brightness_concentration','angular_spread_deg']]
        .mean().round(3)
    )
    print(scene_stats.to_string())

    print("\nPER ARTIST PER SCENE")
    artist_scene = (
        df_all.groupby(['artist','scene'])[['peak_mean_ratio','brightness_concentration','angular_spread_deg']]
        .mean().round(3).reset_index()
    )
    print(artist_scene.to_string(index=False))

    if args.case_studies:
        print("\nCASE STUDIES vs DATASET")
        case_results = []
        for base in args.case_studies:
            match = df_all[df_all['file'].str.startswith(base)]
            if match.empty:
                print(f" NOT FOUND: {base}")
                continue
            row = match.mean(numeric_only=True)
            artist = match['artist'].iloc[0]
            scene = match['scene'].iloc[0]
            pct_pm = (df_all['peak_mean_ratio'] <= row['peak_mean_ratio']).mean()*100
            pct_con = (df_all['brightness_concentration'] <= row['brightness_concentration']).mean()*100
            pct_sp = (df_all['angular_spread_deg'] <= row['angular_spread_deg']).mean()*100
            print(f"\n  {base}")
            print(f" Artist : {artist}  Scene: {scene}")
            print(f" Peak/Mean : {row['peak_mean_ratio']:.3f} (dataset percentile: {pct_pm:.0f}%)")
            print(f" Conc. : {row['brightness_concentration']:.3f}  (dataset percentile: {pct_con:.0f}%)")
            print(f" Spread : {row['angular_spread_deg']:.3f}° (dataset percentile: {pct_sp:.0f}%)")
            print(f" Elevation : {row['dominant_elevation_deg']:.3f}°")
            case_results.append({
                'painting': base, 'artist': artist, 'scene': scene,
                'peak_mean_ratio': round(row['peak_mean_ratio'],3), 'peak_mean_pct': round(pct_pm,1),
                'brightness_concentration': round(row['brightness_concentration'],3), 'concentration_pct': round(pct_con,1),
                'angular_spread_deg': round(row['angular_spread_deg'],3), 'spread_pct': round(pct_sp,1),
                'dominant_elevation_deg': round(row['dominant_elevation_deg'],3),
            })
        pd.DataFrame(case_results).to_csv(os.path.join(out_dir,"diffusion_case_studies.csv"), index=False)

    df_all.to_csv(os.path.join(out_dir,"diffusion_metrics_all.csv"), index=False)
    artist_stats.to_csv(os.path.join(out_dir,"diffusion_metrics_artist.csv"))
    artist_scene.to_csv(os.path.join(out_dir,"diffusion_metrics_artist_scene.csv"), index=False)
    print(f"\nAll CSVs saved to: {out_dir}")
