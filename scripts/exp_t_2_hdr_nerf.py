import os
import json
if __name__ == '__main__':
    root = '/run/user/1000/gvfs/sftp:host=login/datasets/HDR-Bad-Gaussian/'
    scenes = ['factory', 'outdoorpool', 'cozyroom']

    for scene in scenes:
        scene_path = os.path.join(root,scene)
        exposure_gt_path = os.path.join(scene_path,'exposure_times_gt.txt')
        exposure_json_path = os.path.join(scene_path,'exposure.json')

        with open(exposure_gt_path, 'r') as file:
            lines = file.readlines()

        # Process the lines to create a dictionary
        exposure_dict = {}
        for line in lines:
            filename, exposure_time = line.strip().split()
            # Convert filename to the required format
            frame_number = int(float(filename))
            filename_formatted = f"frame_{frame_number:05d}.png"
            exposure_dict[filename_formatted] = float(exposure_time)

        # Save the dictionary as a JSON file
        with open(exposure_json_path, 'w') as json_file:
            json.dump(exposure_dict, json_file, indent=4)