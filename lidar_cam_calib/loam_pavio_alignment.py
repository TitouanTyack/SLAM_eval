import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation 

from isae_odometry import IsaeEvalOdom
import kitti_odometry as ko


def load_poses_from_csv(file_name):
    """Load poses from txt (KITTI format + timestamp)
    Each line in the file should follow one of the following structures
        (1) timestamp
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
        timestamp (dict)
    """
    df = pd.read_csv(file_name)
    timestamps = {}
    poses = {}
    
    for idx, row in df.iterrows():
        timestamps[idx] = row['timestamp'] * 1e9
        q = np.array([row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']])
        r = Rotation.from_quat(q)
        R = r.as_matrix()
        P = np.eye(4)
        P[0:3, 0:3] = R
        P[0:3, 3] = np.array([row['trans_x'], row['trans_y'], row['trans_z']])
        poses[idx] = P

    return poses, timestamps


isae_eval = IsaeEvalOdom()


filename = "lidar_cam_calib/fisheye3_pavio.txt"
filename_loam = "lidar_cam_calib/fisheye3_loam.csv"

poses_pavio, ts_pavio = isae_eval.load_poses_from_txt_ts(filename)
poses_loam, ts_loam = load_poses_from_csv(filename_loam)

# sync both trajectories
df_pavio = pd.DataFrame({
    "timestamp" : ts_pavio
})
df_loam = pd.DataFrame({
    "timestamp" : ts_loam
})

poses_loam_sync = {}
poses_pavios_sync = {}
counter = 0
counter_loam = 0 

for ts in df_loam['timestamp']:
    counter_loam += 1
    idx = df_pavio['timestamp'].sub(float(ts)).abs().idxmin()
    if (df_pavio['timestamp'].sub(float(ts)).abs()[idx] * 1e-9 > 0.1):
        continue

    poses_pavios_sync[counter] = poses_pavio[idx]
    poses_loam_sync[counter] = poses_loam[counter_loam]
    
    counter += 1

poses_pavio = poses_pavios_sync
poses_loam = poses_loam_sync

# format lists
xyz_loam = []
xyz_pavio = []
for cnt in poses_pavio:
    xyz_loam.append([poses_loam[cnt][0, 3], poses_loam[cnt][1, 3], poses_loam[cnt][2, 3]])
    xyz_pavio.append([poses_pavio[cnt][0, 3], poses_pavio[cnt][1, 3], poses_pavio[cnt][2, 3]])
xyz_loam = np.asarray(xyz_loam).transpose(1, 0)
xyz_pavio = np.asarray(xyz_pavio).transpose(1, 0)

# Umeyama alignement
r,t,c = ko.umeyama_alignment(xyz_pavio, xyz_loam)

# Display
print("Rotation")
print(r)
print("Translation")
print(t)

# r = Rotation.from_matrix(r)
# print(r.as_euler('xyz')*180/np.pi)
