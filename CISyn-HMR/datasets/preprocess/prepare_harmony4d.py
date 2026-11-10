"""
Process data/harmony4d/{train,test} into MA-HMR annotation format.

Harmony4D raw structure (per sequence):
  {split}/{big_seq}/{seq}/
    processed_data/smpl/{:05d}.npy   -- per-frame SMPL dict keyed by 'aria01','aria02'
        each: global_orient (3,), body_pose (69,), transl (3,), betas (10,),
              vertices (6890,3), joints (45,3)
    exo/cam{NN}/images/{:05d}.jpg    -- exo camera images (3840x2160 GoPro)
    colmap/workplace/cameras.txt     -- COLMAP intrinsics  (OPENCV_FISHEYE)
    colmap/workplace/images.txt      -- COLMAP extrinsics  (quat + translation)
    colmap/workplace/scale.npy                        -- exo-only mode
    colmap/workplace/colmap_from_aria_transforms.pkl  -- ego_exo mode

Coordinate system:
  SMPL params live in the "primary world" frame:
    ego_exo mode  -> aria01 world  (z-up, roughly metric)
    exo-only mode -> metric world  (z-up after scale.npy alignment)
  Camera extrinsics are computed as  E = E_colmap @ primary_transform  so that
  they map from the SMPL world to OpenCV camera space (x-right, y-down, z-fwd).
  The primary_transform is a similarity (includes a scale factor), so we decompose
  the 3x3 part into a pure rotation via SVD and divide the translation by the same
  scale.  This preserves 2D projection and gives a proper rotation matrix that
  cv2.Rodrigues in base.py can consume.

  Pinhole approximation: by default we take (fx, fy, cx, cy) from the COLMAP
  OPENCV_FISHEYE model and ignore k1-k4 distortion.  With --undistort, images
  are rectified to a true pinhole model via cv2.fisheye and saved alongside the
  originals in images_undistorted/.

Output: data/harmony4d/harmony4d_smpl_{split}.npz
  annots dict: {rel_img_path -> [person_dict, ...]}
  rel_img_path is relative to data/harmony4d/
"""
import os
import pickle
import argparse
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

FAIL_LIST = []
HUMAN_NAMES = ['aria01', 'aria02']


# ---------------------------------------------------------------------------
# Quaternion helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def quat_to_rotmat(qw, qx, qy, qz):
    """Hamilton quaternion (scalar-first) to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# COLMAP text-file parsers
# ---------------------------------------------------------------------------

def parse_colmap_cameras(cameras_txt):
    """Parse cameras.txt -> {camera_id: dict}.

    Each dict has keys: fx, fy, cx, cy, width, height, model, dist_coeffs.
    dist_coeffs is (k1,k2,k3,k4) for OPENCV_FISHEYE, empty tuple otherwise.
    """
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            dist_coeffs = ()
            if model == 'OPENCV_FISHEYE':
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                if len(params) >= 8:
                    dist_coeffs = tuple(params[4:8])
            elif model in ('PINHOLE', 'SIMPLE_PINHOLE'):
                fx = params[0]
                fy = params[1] if model == 'PINHOLE' else params[0]
                cx = params[-2]
                cy = params[-1]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            cameras[cam_id] = dict(
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                model=model, dist_coeffs=dist_coeffs)
    return cameras


def get_undistort_maps(cam_info):
    """Compute fisheye undistortion maps and new pinhole intrinsics.

    Returns (map1, map2, new_K) or None if camera has no distortion.
    """
    if not cam_info['dist_coeffs'] or cam_info['model'] != 'OPENCV_FISHEYE':
        return None
    K = np.array([
        [cam_info['fx'], 0, cam_info['cx']],
        [0, cam_info['fy'], cam_info['cy']],
        [0, 0, 1],
    ], dtype=np.float64)
    D = np.array(cam_info['dist_coeffs'], dtype=np.float64)
    size = (cam_info['width'], cam_info['height'])
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    return map1, map2, new_K


def parse_colmap_images(images_txt):
    """Parse images.txt -> {name: (R_3x3, t_3x1, camera_id)}.

    COLMAP convention:  v_cam = R @ v_colmap_world + t
    """
    images = {}
    with open(images_txt) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]),
                           float(parts[3]), float(parts[4]))
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        R = quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        images[name] = (R, t, cam_id)
    return images


# ---------------------------------------------------------------------------
# Extrinsics helpers
# ---------------------------------------------------------------------------

def load_primary_transform(colmap_dir):
    """Return the 4x4 transform from SMPL world to COLMAP world."""
    scale_path = os.path.join(colmap_dir, 'scale.npy')
    aria_path = os.path.join(colmap_dir, 'colmap_from_aria_transforms.pkl')

    if os.path.isfile(scale_path):
        T = np.load(scale_path)
        return np.linalg.inv(T), 'exo'
    elif os.path.isfile(aria_path):
        with open(aria_path, 'rb') as f:
            colmap_from_aria = pickle.load(f)
        return colmap_from_aria['aria01'], 'ego_exo'
    else:
        raise FileNotFoundError(
            f'Neither scale.npy nor colmap_from_aria_transforms.pkl in {colmap_dir}')


def decompose_extrinsics(full_ext_4x4):
    """Decompose a 4x4 world-to-camera similarity into (R_pure, cam_trans).

    full_ext[:3,:3] may contain a uniform scale from the primary_transform.
    We factor it as  M = s * R  via SVD, then set cam_trans = t / s so that
    perspective projection (x/z, y/z) is invariant.
    """
    M = full_ext_4x4[:3, :3].astype(np.float64)
    t = full_ext_4x4[:3, 3].astype(np.float64)

    U, S, Vh = np.linalg.svd(M)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        Vh[-1] *= -1
        R = U @ Vh
    scale = S.mean()

    cam_trans = t / scale
    return R.astype(np.float32), cam_trans.astype(np.float32)


def build_exo_extrinsics(colmap_images, primary_transform):
    """Group per-camera calibrated extrinsics in SMPL-world frame.

    Returns {cam_name: [(time_stamp, R_pure, cam_trans, colmap_cam_id), ...]}.
    """
    exo_ext = defaultdict(list)
    for name, (R_col, t_col, cam_id) in colmap_images.items():
        cam_name = name.split('/')[0]
        if not cam_name.startswith('cam'):
            continue
        try:
            ts = int(name.split('/')[1].replace('.jpg', ''))
        except (IndexError, ValueError):
            continue

        raw_ext = np.eye(4, dtype=np.float64)
        raw_ext[:3, :3] = R_col
        raw_ext[:3, 3] = t_col
        full_ext = raw_ext @ primary_transform
        R_pure, cam_trans = decompose_extrinsics(full_ext)
        exo_ext[cam_name].append((ts, R_pure, cam_trans, cam_id))

    for cam_name in exo_ext:
        exo_ext[cam_name].sort(key=lambda x: x[0])
    return dict(exo_ext)


def nearest_extrinsics(calib_list, time_stamp):
    """Return (R, cam_trans, cam_id) for the calibrated frame nearest to time_stamp."""
    best = min(calib_list, key=lambda x: abs(x[0] - time_stamp))
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Sequence processor
# ---------------------------------------------------------------------------

def process_sequence(seq_path, split, big_seq, seq_name, annots,
                     undistort=False):
    """Process one sequence and add entries to annots dict."""
    colmap_dir = os.path.join(seq_path, 'colmap', 'workplace')
    cameras_txt = os.path.join(colmap_dir, 'cameras.txt')
    images_txt = os.path.join(colmap_dir, 'images.txt')
    smpl_dir = os.path.join(seq_path, 'processed_data', 'smpl')
    exo_dir = os.path.join(seq_path, 'exo')

    if not all(os.path.isfile(p) for p in [cameras_txt, images_txt]):
        print(f'  Skipping {big_seq}/{seq_name}: missing COLMAP files')
        return 0
    if not os.path.isdir(smpl_dir):
        print(f'  Skipping {big_seq}/{seq_name}: no SMPL data')
        return 0
    if not os.path.isdir(exo_dir):
        print(f'  Skipping {big_seq}/{seq_name}: no exo cameras')
        return 0

    try:
        primary_transform, seq_mode = load_primary_transform(colmap_dir)
    except FileNotFoundError as e:
        print(f'  Skipping {big_seq}/{seq_name}: {e}')
        return 0

    colmap_cameras = parse_colmap_cameras(cameras_txt)
    colmap_images = parse_colmap_images(images_txt)
    exo_ext = build_exo_extrinsics(colmap_images, primary_transform)

    smpl_files = sorted(
        f for f in os.listdir(smpl_dir) if f.endswith('.npy'))

    exo_camera_names = sorted(
        d for d in os.listdir(exo_dir)
        if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d)))

    # Pre-compute undistortion maps and new intrinsics per colmap camera_id
    undistort_cache = {}  # cam_id -> (map1, map2, new_fx, new_fy, new_cx, new_cy)
    if undistort:
        for cam_id, cam_info in colmap_cameras.items():
            result = get_undistort_maps(cam_info)
            if result is not None:
                map1, map2, new_K = result
                undistort_cache[cam_id] = (
                    map1, map2,
                    float(new_K[0, 0]), float(new_K[1, 1]),
                    float(new_K[0, 2]), float(new_K[1, 2]))

    # Track which images have already been undistorted on disk
    undistorted_on_disk = set()

    img_subdir = 'images_undistorted' if undistort else 'images'

    count = 0
    for smpl_file in smpl_files:
        time_stamp = int(smpl_file.replace('.npy', ''))
        smpl_data = np.load(
            os.path.join(smpl_dir, smpl_file), allow_pickle=True).item()

        for cam_name in exo_camera_names:
            if cam_name not in exo_ext:
                continue

            src_img_path = os.path.join(
                exo_dir, cam_name, 'images', f'{time_stamp:05d}.jpg')
            if not os.path.isfile(src_img_path):
                continue

            R, cam_trans, cam_id = nearest_extrinsics(
                exo_ext[cam_name], time_stamp)

            if cam_id not in colmap_cameras:
                continue

            # Resolve intrinsics (undistorted or original)
            if undistort and cam_id in undistort_cache:
                m1, m2, fx, fy, cx, cy = undistort_cache[cam_id]
                # Write undistorted image once
                dst_dir = os.path.join(exo_dir, cam_name, 'images_undistorted')
                dst_img_path = os.path.join(dst_dir, f'{time_stamp:05d}.jpg')
                if dst_img_path not in undistorted_on_disk:
                    if not os.path.isfile(dst_img_path):
                        os.makedirs(dst_dir, exist_ok=True)
                        img = cv2.imread(src_img_path)
                        img_ud = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                        cv2.imwrite(dst_img_path, img_ud)
                    undistorted_on_disk.add(dst_img_path)
            else:
                ci = colmap_cameras[cam_id]
                fx, fy, cx, cy = ci['fx'], ci['fy'], ci['cx'], ci['cy']

            focal = np.array([fx, fy], dtype=np.float32)
            princpt = np.array([cx, cy], dtype=np.float32)

            person_list = []
            for human_name in HUMAN_NAMES:
                if human_name not in smpl_data:
                    continue
                hd = smpl_data[human_name]
                try:
                    global_orient = hd['global_orient'].astype(np.float32)
                    body_pose = hd['body_pose'].astype(np.float32)
                    transl = hd['transl'].astype(np.float32)
                    betas = hd['betas'].astype(np.float32)

                    if global_orient.shape != (3,):
                        global_orient = global_orient.reshape(3)
                    if body_pose.shape != (69,):
                        body_pose = body_pose.reshape(69)
                    if transl.shape != (3,):
                        transl = transl.reshape(3)
                    if betas.shape[0] > 10:
                        betas = betas[:10]
                    elif betas.shape[0] < 10:
                        betas = np.pad(betas, (0, 10 - betas.shape[0]))

                    person_list.append({
                        'focal': focal,
                        'princpt': princpt,
                        'cam_rot': R,
                        'cam_trans': cam_trans,
                        'global_orient': global_orient,
                        'body_pose': body_pose,
                        'transl': transl,
                        'betas': betas,
                        'genders': 'neutral',
                    })
                except Exception as e:
                    FAIL_LIST.append(
                        f'{big_seq}/{seq_name}/{smpl_file}/{human_name}: {e}')
                    continue

            if person_list:
                annot_key = os.path.join(
                    split, big_seq, seq_name,
                    'exo', cam_name, img_subdir, f'{time_stamp:05d}.jpg')
                annots[annot_key] = person_list
                count += 1

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Harmony4D annotations for MA-HMR.')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Data root containing harmony4d/')
    parser.add_argument('--splits', nargs='+', default=['test'],
                        help='Splits to process')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Max sequences per big_seq (for debugging)')
    parser.add_argument('--undistort', action='store_true', default=True,
                        help='Undistort fisheye exo images to pinhole '
                             '(writes images_undistorted/ next to images/)')
    args = parser.parse_args()

    h4d_root = os.path.join(args.data_root, 'harmony4d')

    for split in args.splits:
        split_root = os.path.join(h4d_root, split)
        if not os.path.isdir(split_root):
            print(f'Split dir not found, skipping: {split_root}')
            continue

        big_seqs = sorted(
            d for d in os.listdir(split_root)
            if os.path.isdir(os.path.join(split_root, d)))
        print(f'\n=== {split} === ({len(big_seqs)} big sequences)')

        annots = {}
        total = 0

        for big_seq in big_seqs:
            big_seq_path = os.path.join(split_root, big_seq)
            seqs = sorted(
                d for d in os.listdir(big_seq_path)
                if os.path.isdir(os.path.join(big_seq_path, d)))
            if args.max_seqs is not None:
                seqs = seqs[:args.max_seqs]

            for seq_name in tqdm(seqs, desc=big_seq, ncols=80):
                seq_path = os.path.join(big_seq_path, seq_name)
                n = process_sequence(
                    seq_path, split, big_seq, seq_name, annots,
                    undistort=args.undistort)
                total += n

            print(f'  {big_seq}: cumulative {total} annotated frames')

        out_path = os.path.join(h4d_root, f'harmony4d_smpl_{split}.npz')
        np.savez(out_path, annots=annots)
        print(f'Saved {total} image annotations to {out_path}')

    if FAIL_LIST:
        print(f'\nFailed {len(FAIL_LIST)} entries:')
        for p in FAIL_LIST:
            print(f'  {p}')


if __name__ == '__main__':
    main()
