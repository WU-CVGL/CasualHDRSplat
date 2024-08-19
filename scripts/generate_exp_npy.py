import os
import json
import numpy as np

if __name__=='__main__':
    # Load the existing N*17 npy file
    root = '/run/user/1000/gvfs/sftp:host=login/datasets/HDR-Bad-Gaussian/'
    scenes = ['factory', 'outdoorpool', 'cozyroom']
    for scene in scenes:
        scene_path = os.path.join(root,scene)
        poses_bounds_path = os.path.join(scene_path,'poses_bounds.npy')
        poses_bounds = np.load(poses_bounds_path)
        exposure_json_path = os.path.join(scene_path, 'exposure.json')
        k_times = float([f for f in os.listdir(scene_path) if f.startswith('k_times=')][0].split('=')[1])

        # Load the JSON file with exposure times
        with open(exposure_json_path, 'r') as f:
            exposure_times = json.load(f)

        # Extract the exposure times and match them with corresponding frames
        exposure_values = []
        for i in range(len(poses_bounds)):
            frame_name = f"frame_{str(i + 1001).zfill(5)}.png"
            exposure = exposure_times.get(frame_name, None)
            print(exposure)
            if exposure is not None:
                exposure *=k_times
                exposure_values.append(exposure)
            else:
                print('None!')
                exposure_values.append(0.0)  # Default value if exposure time is missing

        # Convert exposure_values to a numpy array and reshape to match poses_bounds
        exposure_values = np.array(exposure_values).reshape(-1, 1)

        # Append the exposure times as the last column
        poses_bounds_exps = np.hstack((poses_bounds, exposure_values))
        poses_bounds_exps_path = os.path.join(scene_path,'poses_bounds_exps.npy')
        # Save the new array with exposure times
        np.save(poses_bounds_exps_path, poses_bounds_exps)