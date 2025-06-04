import json
import numpy as np
import os
import sys

from config import DATA_DIR

def extract_keypoints_sequences(json_path, num_keypoints=18):
    
    with open(json_path, 'r') as f:
        frames = json.load(f)
    if len(frames) == 0:
        raise ValueError("No frames found in the JSON file.")
    
    base_frame_poses = frames[0]['poses']
    i=1
    while i < len(frames) and not base_frame_poses:
        base_frame_poses = frames[i]['poses']
        i += 1

    if not base_frame_poses:
        print(f"No poses found in any frame of {json_path}")
        return None, None

    target_pose = max(base_frame_poses, key=lambda x: x['confidence'])
    target_id = target_pose['id']

    sequence = []
    for frame in frames:
        matched_pose = None
        for pose in frame['poses']:
            if pose['id'] == target_id:
                matched_pose = pose
                break
        if matched_pose is None:
            sequence.append(np.full((num_keypoints, 2), -1, dtype=np.float32))
        else:
            keypoints = matched_pose['keypoints']
            coords = np.array([[kp['x'], kp['y']] if kp['x'] != -1 or kp['y'] != -1 else [-1, -1] for kp in keypoints], dtype=np.float32)
            sequence.append(coords)
    
    return target_id, np.stack(sequence) # (T, K, 2) 


def main():
    json_dir = os.path.join(DATA_DIR, 'keypoints')
    output_dir = os.path.join(DATA_DIR, 'sequences')
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        target_id, sequence = extract_keypoints_sequences(json_path)

        if sequence is None:
            print(f"Skipping {json_file} due to no valid poses.")
            continue
        
        output_path = os.path.join(output_dir, json_file.replace('.json', '.npz'))
        np.savez_compressed(output_path, keypoints=sequence, id=target_id)
        print(f"Saved sequence to {output_path} (id={target_id}, length={sequence.shape[0]})")


if __name__ == '__main__':
    main()