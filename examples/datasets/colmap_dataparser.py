import sys
sys.path.append('/run/determined/workdir/home/gsplat/examples')

from pathlib import Path
from typing import List, Literal

import cv2
import os
import numpy as np
import datasets.colmap_parsing_utils as colmap_utils
from datasets.colmap_utils import parse_colmap_camera_params, auto_orient_and_center_poses
from trajectory_evaluation import plot_trajectories3D, align_umeyama, compute_absolute_error_translation

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class ColmapParser:
    """an adapted version of nerfstudio ColmapDataParser"""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        scale_factor: float = 1.0,
        orientation_method: Literal["pca", "up", "vertical", "none"] = "up",
        center_method: Literal["poses", "focus", "none"] = "poses",
        auto_scale_poses: bool = True,
        test_every: int = 8,
    ):
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        self.data_dir = Path(data_dir)
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.scale_factor = scale_factor
        
        colmap_dir = data_dir / "sparse/0/"
        if not os.path.exists(colmap_dir):
            colmap_dir = data_dir / "sparse"
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        if (colmap_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(colmap_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(colmap_dir / "images.txt")
        elif (colmap_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(colmap_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(colmap_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {colmap_dir}")

        # read in GT poses if exists
        if (colmap_dir / "images_gt.bin").exists():
            im_id_to_image_gt = colmap_utils.read_images_binary(colmap_dir / "images_gt.bin")
        elif (colmap_dir / "images_gt.txt").exists():
            im_id_to_image_gt = colmap_utils.read_images_text(colmap_dir / "images_gt.txt")
        else:
            im_id_to_image_gt = None

        cameras = {}
        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())

        # Extract extrinsic matrices in world-to-camera format.
        # imdata = manager.images
        w2c_mats = []
        w2c_mats_gt = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for im_id in ordered_im_id:
            im = im_id_to_image[im_id]
            rot = colmap_utils.qvec2rotmat(im.qvec)
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            if im_id_to_image_gt is not None:
                im_gt = im_id_to_image_gt[im_id+1] # NOTE: +1 hacks for now
                rot_gt = colmap_utils.qvec2rotmat(im_gt.qvec)
                trans_gt = im_gt.tvec.reshape(3, 1)
                w2c_gt = np.concatenate([np.concatenate([rot_gt, trans_gt], 1), bottom], axis=0)
                w2c_mats_gt.append(w2c_gt)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            # fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            fx, fy, cx, cy = cam['fl_x'], cam['fl_y'], cam['cx'], cam['cy']
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam["model"]
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            # imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            imsize_dict[camera_id] = (cam['w'] // factor, cam['h'] // factor)
        print(
            f"[Parser] {len(im_id_to_image)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(im_id_to_image) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ in ["PINHOLE", "SIMPLE_PINHOLE", 0, 1]):
            print(f"Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [im_id_to_image[k].name for k in ordered_im_id]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # construct masks path
        mask_dir = os.path.join(data_dir, "masks")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # get masks path
        if os.path.exists(mask_dir):
            mask_paths = [os.path.join(mask_dir, f) for f in image_names]
            self.mask_paths = mask_paths
            print('Got masks path!')

        camtoworlds, transform_matrix = auto_orient_and_center_poses(
            camtoworlds, 
            method=orientation_method,
            center_method=center_method
        )

        scale_factor = 1.0 # NOTE: 
        if auto_scale_poses:
            scale_factor /= float(np.max(np.abs(camtoworlds[:, :3, 3]))) 
        scale_factor *= self.scale_factor
        camtoworlds[:, :3, 3] *= scale_factor
        N = camtoworlds.shape[0]
        bottoms = np.repeat(bottom[np.newaxis, :], N, axis=0)
        camtoworlds = np.concatenate((camtoworlds, bottoms), axis=1)

        if im_id_to_image_gt is not None:
            w2c_mats_gt = np.stack(w2c_mats_gt, axis=0)
            camtoworlds_gt = np.linalg.inv(w2c_mats_gt)
            camtoworlds_gt = camtoworlds_gt[inds]
            camtoworlds_gt, _ = auto_orient_and_center_poses(
            camtoworlds_gt, 
            method=orientation_method,
            center_method=center_method
            )
            scale_factor_gt = 1.0
            if auto_scale_poses:
                scale_factor_gt /= float(np.max(np.abs(camtoworlds_gt[:, :3, 3]))) 
            scale_factor_gt *= self.scale_factor
            camtoworlds_gt[:, :3, 3] *= scale_factor_gt
            camtoworlds_gt = np.concatenate((camtoworlds_gt, bottoms), axis=1)
            self.camtoworlds_gt = camtoworlds_gt

        # load in 3D points
        if (colmap_dir / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_dir / "points3D.bin")
        elif (colmap_dir / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_dir / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_dir}")
        points = np.array([p.xyz for p in colmap_points.values()], dtype=np.float32)
        points = (
            np.concatenate(
                (
                    points,
                    np.ones_like(points[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points *= scale_factor

        points_rgb = np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8)
        points_err = np.array([p.error for p in colmap_points.values()], dtype=np.float32)

        
        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.transform = transform_matrix  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


if __name__ == "__main__":
    parser = ColmapParser(data_dir=Path('/run/determined/workdir/home/gsplat/examples/data/sci_nerf/colmap/hotdog'))
    
    c2ws = parser.camtoworlds
    c2ws_gt = parser.camtoworlds_gt

    traj1 = c2ws_gt[:, :3, -1]
    traj2 = c2ws[:, :3, -1]

    # plot them before alignment
    plot_trajectories3D(traj1, traj2, filename="trajs_before_align.png")

    # align w/ SIM3
    s, R, t = align_umeyama(traj1, traj2)

    # apply transformation on the data (traj2)
    traj2_aligned = (s * (R @ traj2.T)).T + t

    # plot them post alignment
    plot_trajectories3D(traj1, traj2_aligned, filename="trajs_after_align.png")

    e_trans, _ = compute_absolute_error_translation(traj2_aligned, traj1)
    print(f"ATE for translation vector is {e_trans}")