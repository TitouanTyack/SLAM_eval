import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from scipy.spatial.transform import Rotation
import os
from glob import glob
import pickle


def load_poses_from_txt_ov(file_name, length=-1):
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
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    timestamps = {}

    for cnt, line in enumerate(s):
        

        # Stop the loading if there is a length parameter
        if (cnt == 0):
            continue

        P = np.eye(4)
        line_split = [i for i in line.split(" ") if i != '']
        line_split = [float(i) for i in line_split[0:9]]

        t = np.array([line_split[5], line_split[6], line_split[7]])
        q = np.array([line_split[1], line_split[2], line_split[3], line_split[4]])
        r = Rotation.from_quat(q)
        P[0:3, 0:3] = r.as_matrix()
        P[0:3, 3] = t

        timestamps[cnt-1] = line_split[0] * 1e9
        poses[cnt-1] = P

    return poses, timestamps

def load_poses_from_txt_ORBSLAM(file_name, length=-1):
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
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    timestamps = {}

    for cnt, line in enumerate(s):

        # Stop the loading if there is a length parameter
        if (length != -1):
            if (cnt > length):
                break

        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 14
        t = np.array([line_split[1], line_split[2], line_split[3]])
        q = np.array([line_split[4], line_split[5], line_split[6], line_split[7]])
        r = Rotation.from_quat(q)
        P[0:3, 0:3] = r.as_matrix()
        P[0:3, 3] = t

        timestamps[cnt] = line_split[0]
        poses[cnt] = P

    return poses, timestamps

def load_poses_from_txt_ts(file_name, length=-1):
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
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    timestamps = {}

    for cnt, line in enumerate(s):

        # Stop the loading if there is a length parameter
        if (length != -1):
            if (cnt > length):
                break

        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 14
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + 1 + withIdx]

        else:
            timestamps[cnt+1] = line_split[0]
            poses[cnt+1] = P

    return poses, timestamps


def load_poses_from_csv_isae(file_name):
    """Load poses from csv ISAE format
    Each line in the file should follow one of the following structures
        (1) timestamp
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
        timestamp (dict)
    """
    poses = {}
    timestamp = {}

    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            P = np.eye(4)
            P[0, 3] = row[' T_wf(03)']
            P[1, 3] = row[' T_wf(13)']
            P[2, 3] = row[' T_wf(23)']

            P[0:3, 0:3] = np.array([[row[' T_wf(00)'], row[' T_wf(01)'], row[' T_wf(02)']],
                                    [row[' T_wf(10)'], row[' T_wf(11)'],
                                     row[' T_wf(12)']],
                                    [row[' T_wf(20)'], row[' T_wf(21)'], row[' T_wf(22)']]])
            
            # For "foire à la saucisse"  
            #P[0:3, 0:3] = np.identity(3)

            poses[counter] = P
            timestamp[counter] = int(row['timestamp (ns)'])
            counter += 1

    return poses, timestamp


def load_poses_from_csv_EUROC(file_name):
    """Load poses from csv (EUROC format)
    The poses are returned in the left camera frame

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
        timestamp (dict)
    """
    poses = {}
    timestamp = {}

    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            P = np.eye(4)
            P[0, 3] = row[' p_RS_R_x [m]']
            P[1, 3] = row[' p_RS_R_y [m]']
            P[2, 3] = row[' p_RS_R_z [m]']

            r = Rotation.from_quat([row[' q_RS_x []'],
                                    row[' q_RS_y []'],
                                    row[' q_RS_z []'],
                                    row[' q_RS_w []']])

            P[0:3, 0:3] = r.as_matrix()
            poses[counter] = P
            timestamp[counter] = int(row['#timestamp'])
            counter += 1
    return poses, timestamp

def load_poses_from_csv_KIMERA(file_name):
    """Load poses from csv (KIMERA format)
    The poses are returned in the left camera frame

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
        timestamp (dict)
    """
    poses = {}
    timestamp = {}

    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = 0
        for row in reader:
            P = np.eye(4)
            P[0, 3] = row['x']
            P[1, 3] = row['y']
            P[2, 3] = row['z']

            r = Rotation.from_quat([row['qx'],
                                    row['qy'],
                                    row['qz'],
                                    row['qw']])

            P[0:3, 0:3] = r.as_matrix()
            
            # For "foire à la saucisse"  
            # P[0:3, 0:3] = np.identity(3)
            
            poses[counter] = P
            timestamp[counter] = int(row['#timestamp'])
            counter += 1
    return poses, timestamp

def load_poses_from_csv_OIVIO(file_name):
        """Load poses from csv (OVIO format)
        The poses are returned in the left camera frame

        Args:
            file_name (str): txt file path
        Returns:
            poses (dict): {idx: 4x4 array}
            timestamp (dict)
        """
        poses = {}
        timestamp = {}

        with open(file_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            counter = 0
            for row in reader:
                P = np.eye(4)
                P[0, 3] = row['p_WS_R_x [m]']
                P[1, 3] = row['p_WS_R_y [m]']
                P[2, 3] = row['p_WS_R_z [m]']

                r = Rotation.from_quat([0, 0, 0, 1])

                P[0:3, 0:3] = r.as_matrix()
                poses[counter] = P
                timestamp[counter] = int(row['#timestamp [ns]'])
                counter += 1
        return poses, timestamp
