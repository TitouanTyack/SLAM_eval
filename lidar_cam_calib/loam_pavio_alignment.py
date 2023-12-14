import argparse
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation 
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy import optimize

from isae_odometry import IsaeEvalOdom
import kitti_odometry as ko

class LidarCamCalibration:
    def __init__(self, dT_c_traj, dT_l_traj):
        self.x_arr = []
        self.cost_arr = []
        self. dT_c_traj =  dT_c_traj
        self.dT_l_traj = dT_l_traj
        self.N = len(dT_l_traj)
        self.N_res = 3*self.N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        l_nu_c = x[:6]  # currently estimated transfo as se3 6D vector representation
        l_T_c = pin.exp6(l_nu_c)  # currently estimated transfo as SE3 Lie group

        res = np.zeros(self.N_res)
        for i in range(self.N):
            lt_T_ltp1 = pin.SE3(self.dT_l_traj[i])
            ct_T_ctp1 = pin.SE3(self.dT_c_traj[i])
            
            err_se3   = l_T_c * ct_T_ctp1 * l_T_c.inverse() * lt_T_ltp1.inverse()
            res[3*i:3*i+3]   = err_se3.translation

        self.cost_arr.append(np.linalg.norm(res))

        return res

class RotationCalibration:
    def __init__(self, dT_c_traj, dT_l_traj):
        self.x_arr = []
        self.cost_arr = []
        self. dT_c_traj =  dT_c_traj
        self.dT_l_traj = dT_l_traj
        self.N = len(dT_l_traj)
        self.N_res = self.N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        l_nu_c = x[:3]  # currently estimated transfo as se3 6D vector representation
        l_R_c = pin.exp3(l_nu_c)  # currently estimated transfo as SE3 Lie group

        res = np.zeros(self.N_res)
        for i in range(self.N):
            lt_T_ltp1 = pin.SE3(self.dT_l_traj[i])
            ct_T_ctp1 = pin.SE3(self.dT_c_traj[i])
            
            err_so3   =  lt_T_ltp1.rotation * l_R_c -  l_R_c * ct_T_ctp1.rotation
            res[i]   = np.sqrt(np.trace(err_so3 * np.transpose(err_so3)))

        self.cost_arr.append(np.linalg.norm(res))

        return res

class TranslationCalibration:
    def __init__(self, dT_c_traj, dT_l_traj, l_R_c):
        self.x_arr = []
        self.cost_arr = []
        self. dT_c_traj =  dT_c_traj
        self.dT_l_traj = dT_l_traj
        self.l_R_c = l_R_c
        self.N = len(dT_l_traj)
        self.N_res = 3*self.N  # size of the trajectory x 6 degrees of freedom of the se3 lie algebra

    def f(self, x):
        self.x_arr.append(x)
        l_t_c = x[:3]  # currently estimated transfo as se3 6D vector representation
        l_T_c = pin.SE3(np.eye(4,4))
        l_T_c.translation = l_t_c
        l_T_c.rotation = l_R_c

        res = np.zeros(self.N_res)
        for i in range(self.N):
            lt_T_ltp1 = pin.SE3(self.dT_l_traj[i])
            ct_T_ctp1 = pin.SE3(self.dT_c_traj[i])
            
            err_se3   = l_T_c * ct_T_ctp1 * l_T_c.inverse() * lt_T_ltp1.inverse()
            res[3*i:3*i+3]   = err_se3.translation

        self.cost_arr.append(np.linalg.norm(res))

        return res


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


filename = "lidar_cam_calib/titou_pavio.txt"
filename_loam = "lidar_cam_calib/titou_loam.csv"

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
poses_pavio_sync = {}
dT_loam_sync = {}
dT_pavio_sync = {}
counter = 0
counter_loam = 0 

for ts in df_loam['timestamp']:
    counter_loam += 1
    idx = df_pavio['timestamp'].sub(float(ts)).abs().idxmin()
    if (df_pavio['timestamp'].sub(float(ts)).abs()[idx] * 1e-9 > 0.01 or counter_loam > len(poses_loam)-1):
        continue
    
    if (counter > 0):
        dT_loam_sync[counter-1] = np.linalg.inv(poses_loam_sync[counter-1]) * poses_loam[counter_loam]
        dT_pavio_sync[counter-1] = np.linalg.inv(poses_pavio_sync[counter-1]) * poses_pavio[idx]
    
    poses_pavio_sync[counter] = poses_pavio[idx]
    poses_loam_sync[counter] = poses_loam[counter_loam]
    
    counter += 1

# optimize calibration
cost = LidarCamCalibration(dT_pavio_sync, dT_loam_sync)
x0 = np.zeros(6)
r = optimize.least_squares(cost.f, x0, jac='2-point', method='lm', verbose=2)
l_T_c = pin.exp6(r.x)

print('Transfo lidar cam')
print(l_T_c)

# examine the problem jacobian at the solution
J = r.jac
H = J.T @ J
u, s, vh = np.linalg.svd(H, full_matrices=True)

plt.figure('cost evolution')
plt.plot(np.arange(len(cost.cost_arr)), np.log(cost.cost_arr))
plt.xlabel('Iterations')
plt.ylabel('Residuals norm')

plt.figure('Hessian singular values')
plt.bar(np.arange(len(s)), np.log(s))
plt.xlabel('degrees of freedom')
plt.ylabel('log(s)')

plt.show()


# # format lists
# xyz_loam = []
# xyz_pavio = []
# for cnt in poses_pavio:
#     xyz_loam.append([poses_loam[cnt][0, 3], poses_loam[cnt][1, 3], poses_loam[cnt][2, 3]])
#     xyz_pavio.append([poses_pavio[cnt][0, 3], poses_pavio[cnt][1, 3], poses_pavio[cnt][2, 3]])
# xyz_loam = np.asarray(xyz_loam).transpose(1, 0)
# xyz_pavio = np.asarray(xyz_pavio).transpose(1, 0)

# # Umeyama alignement
# r,t,c = ko.umeyama_alignment(xyz_pavio, xyz_loam)

# # Display
# print("Rotation")
# print(r)
# print("Translation")
# print(t)

r = Rotation.from_matrix(l_T_c.rotation)
print(r.as_euler('xyz')*180/np.pi)
