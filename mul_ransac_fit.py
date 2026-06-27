#!/usr/bin/env python3
import os, sys, json, gzip, csv, argparse
import fcntl
import numpy as np
import skimage
from skimage import io as skio
from PIL import Image

def load_json_any(path):
    try:
        import orjson
        use_orjson = True
    except Exception:
        use_orjson = False

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return orjson.loads(data) if use_orjson else json.loads(data.decode("utf-8"))
    else:
        if use_orjson:
            with open(path, "rb") as f:
                return orjson.loads(f.read())
        else:
            with open(path, "r") as f:
                return json.load(f)


def unit_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def dir_to_az_el(d):
    x, y, z = d
    az = float(np.degrees(np.arctan2(y, x)))
    el = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    return az, el


# RANSAC
def fit_point_source_weighted(X, dirs, w):
    X64 = X.astype(np.float64, copy=False)
    d64 = dirs.astype(np.float64, copy=False)
    w64 = w.astype(np.float64, copy=False)

    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I3 = np.eye(3, dtype=np.float64)

    for i in range(len(X64)):
        d = d64[i].reshape(3, 1)
        P = I3 - d @ d.T
        wi = float(w64[i])
        A += wi * P
        b += wi * (P @ X64[i])

    return np.linalg.solve(A + 1e-10 * I3, b)


def weighted_mean_abs_angle_deg(L, X, dirs, w):
    v = unit_rows(L[None, :] - X)
    cosang = np.sum(v * dirs, axis=1).clip(-1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    W = w / (w.sum() + 1e-12)
    return float((W * ang).sum())


def multi_ransac_point_source_weighted(
    X, dirs, lum,
    n_iters=800,
    angle_thresh_deg=3.0,
    min_inliers_frac=0.005,
    n_candidates=3,
    seed=0,
    proposal_cap=200_000,
    distinct_deg=30.0
):
    """
    Returns up to n_candidates distinct light hypotheses.

    DISTINCTNESS GUARANTEE (fixed):
      - For each candidate, we compute a representative direction d as the inlier-weighted mean ray direction from light -> surface.
      - We keep a candidate only if its d differs from every previously kept candidate's d by at least distinct_deg (i.e., angle >= distinct_deg).
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    if N < 3:
        return []

    w_all = lum.astype(np.float64, copy=False)
    np.maximum(w_all, 0.0, out=w_all)
    w_all += 1e-12

    # proposal subset if huge
    if N > proposal_cap:
        prop_idx = rng.choice(N, size=proposal_cap, replace=False)
        Xp, dp = X[prop_idx], dirs[prop_idx]
    else:
        Xp, dp = X, dirs

    min_inliers = max(3, int(np.ceil(min_inliers_frac * N)))
    cos_thr = float(np.cos(np.deg2rad(angle_thresh_deg)))
    raw = []

    # RANSAC loop
    for _ in range(n_iters):
        samp = rng.choice(Xp.shape[0], size=3, replace=False)
        try:
            L_seed = fit_point_source_weighted(Xp[samp], dp[samp], np.ones(3))
        except np.linalg.LinAlgError:
            continue

        v_full = unit_rows(L_seed[None, :] - X)
        dots = np.sum(v_full * dirs, axis=1)
        inliers = np.where(dots > cos_thr)[0]
        if len(inliers) < min_inliers:
            continue

        L_ref = fit_point_source_weighted(X[inliers], dirs[inliers], w_all[inliers])

        mae_in = weighted_mean_abs_angle_deg(L_ref, X[inliers], dirs[inliers], w_all[inliers])
        mae_all = weighted_mean_abs_angle_deg(L_ref, X, dirs, w_all)
        frac = float(w_all[inliers].sum() / (w_all.sum() + 1e-12))
        score = frac / (mae_in + 1e-6)

        raw.append((score, L_ref, frac, mae_in, mae_all, inliers))

    if not raw:
        return []

    raw.sort(key=lambda t: t[0], reverse=True)

    cos_distinct = float(np.cos(np.deg2rad(distinct_deg)))

    def cand_dir(L, inn):
        """Inlier-weighted mean direction from light->surface (unit)"""
        v = unit_rows(L[None, :] - X[inn])  # (k,3)
        w = w_all[inn].astype(np.float64, copy=False)
        w = w / (w.sum() + 1e-12)
        d = (v * w[:, None]).sum(axis=0)
        d = d / (np.linalg.norm(d) + 1e-12)
        return d

    # unique holds tuples with extra stored direction at end
    unique = []
    for sc, L, frac, mae_in, mae_all, inn in raw:
        d = cand_dir(L, inn)

        too_close = False
        for kept in unique:
            d_prev = kept[-1]
            if float(np.dot(d, d_prev)) > cos_distinct:
                too_close = True
                break
        if too_close:
            continue

        unique.append((sc, L, frac, mae_in, mae_all, inn, d))
        if len(unique) >= n_candidates:
            break

    # Return the original 6-tuple structure
    return [(sc, L, frac, mae_in, mae_all, inn) for (sc, L, frac, mae_in, mae_all, inn, d) in unique]


# Overlay helpers
LIGHT_COLORS = np.array([
    [1.0, 0.2, 0.2],   # L1
    [0.2, 1.0, 0.2],   # L2
    [0.2, 0.5, 1.0],   # L3
], dtype=np.float32)


def derive_image_stem(path):
    name = os.path.basename(path)
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith("_points.json"):
        name = name[:-len("_points.json")]
    elif name.endswith("_points"):
        name = name[:-len("_points")]
    return os.path.splitext(name)[0]


def find_ball_image(ball_dirs, stem):
    if not ball_dirs:
        return None
    for d in ball_dirs:
        if not d:
            continue
        for ext in (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


def make_cluster_overlay(ball_dirs, vis_dir, json_path, cands, ball_ij):
    if not vis_dir:
        return

    stem = derive_image_stem(json_path)
    out_path = os.path.join(vis_dir, stem + "_clusters.png")

    # Do not overwrite old correct overlays
    if os.path.exists(out_path):
        return

    ball_path = find_ball_image(ball_dirs, stem)
    if ball_path is None:
        return

    ball_img = skio.imread(ball_path)
    if ball_img.ndim == 2:
        ball_img = np.stack([ball_img] * 3, axis=-1)
    if ball_img.shape[-1] == 4:
        ball_img = ball_img[..., :3]

    ball_vis = skimage.img_as_float32(ball_img)
    H, W = ball_vis.shape[:2]

    labels = np.full(len(ball_ij), -1, np.int32)
    for li, cand in enumerate(cands):
        inn = cand[-1] # inliers
        # assign higher-ranked first
        m = (labels[inn] == -1)
        labels[inn[m]] = li

    rows = np.clip(ball_ij[:, 0], 0, H - 1)
    cols = np.clip(ball_ij[:, 1], 0, W - 1)

    for li in range(min(len(cands), len(LIGHT_COLORS))):
        m = (labels == li)
        if not np.any(m):
            continue
        ball_vis[rows[m], cols[m], :] = (
            0.3 * ball_vis[rows[m], cols[m], :] +
            0.7 * LIGHT_COLORS[li][None, :]
        )

    os.makedirs(vis_dir, exist_ok=True)
    Image.fromarray(np.clip(ball_vis * 255.0, 0, 255).astype(np.uint8)).save(out_path)


# CSV helpers
CSV_KEYS = [
    "file", "rank", "score",
    "inlier_weight_frac", "mae_in_deg_weighted", "mae_all_deg_weighted",
    "dir_vx", "dir_vy", "dir_vz", "dir_az_deg", "dir_el_deg",
    "Lx", "Ly", "Lz", "num_inliers",
    "angle_thresh_deg", "n_iters", "N"
]


def read_existing_ranks_from_open_file(f, file_bn):
    f.seek(0)
    r = csv.DictReader(f)
    ranks = set()
    for row in r:
        if (row.get("file") or "").strip() == file_bn:
            try:
                ranks.add(int(float(row.get("rank", "0"))))
            except Exception:
                pass
    return ranks


def write_done_marker(done_dir, stem, found_k, requested_k):
    if not done_dir:
        return
    os.makedirs(done_dir, exist_ok=True)
    p = os.path.join(done_dir, stem + ".done")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"found_k={found_k}\n")
        f.write(f"requested_k={requested_k}\n")
    os.replace(tmp, p)


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_candidates", type=int, default=3)
    ap.add_argument("--ball_dir", action="append", default=[])
    ap.add_argument("--vis_dir", default=None)
    ap.add_argument("--done_dir", default=None, help="Directory to write per-file completion markers.")
    ap.add_argument("--front_nx_thresh", type=float, default=0.1)
    ap.add_argument("--bright_quantile", type=float, default=0.8)
    ap.add_argument("--distinct_deg", type=float, default=30.0)
    ap.add_argument("--angle_thresh_deg", type=float, default=4.0)
    ap.add_argument("--n_iters", type=int, default=600)
    ap.add_argument("--min_inliers_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proposal_cap", type=int, default=200000)
    args = ap.parse_args()

    D = load_json_any(args.input)

    X_all = np.array(D["X"], np.float32)
    dirs_all = unit_rows(np.array(D["dirs_light_to_surface"], np.float32))
    lum_all = np.array(D["luminance"], np.float32)
    normals_all = np.array(D["normals"], np.float32)
    ball_ij_all = np.array(D["ball_ij"], np.int64)

    # filtering
    front = normals_all[:, 0] > args.front_nx_thresh
    if not np.any(front):
        stem = derive_image_stem(args.input)
        write_done_marker(args.done_dir, stem, found_k=0, requested_k=args.n_candidates)
        print(f"[warn] {os.path.basename(args.input)}: no front-facing points")
        return

    q = np.quantile(lum_all[front], args.bright_quantile)
    mask = front & (lum_all >= q)
    if not np.any(mask):
        stem = derive_image_stem(args.input)
        write_done_marker(args.done_dir, stem, found_k=0, requested_k=args.n_candidates)
        print(f"[warn] {os.path.basename(args.input)}: empty after bright filter")
        return

    X = X_all[mask]
    dirs = dirs_all[mask]
    lum = lum_all[mask]
    ball_ij = ball_ij_all[mask]
    N = int(len(X))

    if N < 3:
        stem = derive_image_stem(args.input)
        write_done_marker(args.done_dir, stem, found_k=0, requested_k=args.n_candidates)
        print(f"[warn] {os.path.basename(args.input)}: <3 points after filter")
        return

    cands = multi_ransac_point_source_weighted(
        X, dirs, lum,
        n_iters=args.n_iters,
        angle_thresh_deg=args.angle_thresh_deg,
        min_inliers_frac=args.min_inliers_frac,
        n_candidates=args.n_candidates,
        seed=args.seed,
        proposal_cap=args.proposal_cap,
        distinct_deg=args.distinct_deg
    )

    # Overlay only if missing
    if cands and args.vis_dir:
        make_cluster_overlay(args.ball_dir, args.vis_dir, args.input, cands, ball_ij)

    file_bn = os.path.basename(args.input)
    stem = derive_image_stem(args.input)

    # append missing ranks only (under file lock)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # header if file empty
            f.seek(0, os.SEEK_END)
            empty = (f.tell() == 0)
            if empty:
                f.seek(0)
                w = csv.DictWriter(f, fieldnames=CSV_KEYS)
                w.writeheader()

            have = read_existing_ranks_from_open_file(f, file_bn)

            f.seek(0, os.SEEK_END)
            w = csv.DictWriter(f, fieldnames=CSV_KEYS)

            for rank, (sc, L, frac, mae_in, mae_all, inn) in enumerate(cands, start=1):
                if rank in have:
                    continue

                v = unit_rows(L[None, :] - X[inn])
                w_in = lum[inn].astype(np.float64, copy=False)
                w_in = w_in / (w_in.sum() + 1e-12)
                d = (v * w_in[:, None]).sum(axis=0)
                d /= (np.linalg.norm(d) + 1e-12)
                az, el = dir_to_az_el(d)

                w.writerow({
                    "file": file_bn,
                    "rank": int(rank),
                    "score": float(sc),
                    "inlier_weight_frac": float(frac),
                    "mae_in_deg_weighted": float(mae_in),
                    "mae_all_deg_weighted": float(mae_all),
                    "dir_vx": float(d[0]),
                    "dir_vy": float(d[1]),
                    "dir_vz": float(d[2]),
                    "dir_az_deg": float(az),
                    "dir_el_deg": float(el),
                    "Lx": float(L[0]),
                    "Ly": float(L[1]),
                    "Lz": float(L[2]),
                    "num_inliers": int(len(inn)),
                    "angle_thresh_deg": float(args.angle_thresh_deg),
                    "n_iters": int(args.n_iters),
                    "N": int(N),
                })

            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # mark complete even if <3 lights were found
    write_done_marker(args.done_dir, stem, found_k=len(cands), requested_k=args.n_candidates)

    print(f"[ok] {file_bn}: found {len(cands)} light(s), requested cap {args.n_candidates}")
    if args.vis_dir:
        out_path = os.path.join(args.vis_dir, stem + "_clusters.png")
        if os.path.exists(out_path):
            print(f"[ok] overlay present: {out_path} (not overwritten if already existed)")


if __name__ == "__main__":
    main()

