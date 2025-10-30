import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from scipy.spatial.transform import Rotation
import os
from glob import glob
import pickle
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
from kitti_odometry import EvalOdom, umeyama_alignment, gt_alignment, full_alignment
from data_parser import *

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "DejaVu Sans"
})

class IsaeEvalOdom(EvalOdom):
    """Evaluate isae odometry result
    Usage example:
        vo_eval = IsaeEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def compute_scale_error(self, gt, pred):
        """Compute scale error
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            scale errors
        """
        scale_errors = []

        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2

            # print(i, " : ", self.scale_error(gt_rel, pred_rel))

            scale_errors.append(self.scale_error(gt_rel, pred_rel))
        return scale_errors

    def compute_scale_ratio(self, gt, pred):
        """Compute scale ration
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            scale ratios
        """
        scale_ratios = []
        length = 0

        for i in list(pred.keys())[:-1]:
            gt1 = gt[i]
            gt2 = gt[i+1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i+1]
            pred_rel = np.linalg.inv(pred1) @ pred2

            length += np.linalg.norm(gt_rel[0:3, 3])

            if (np.linalg.norm(gt_rel[0:3, 3]) < 0.05):
                continue

            scale_ratios.append(self.scale_ratio(gt_rel, pred_rel))

        print("Length : ", length)

        return scale_ratios

    def eval(self, gt_dir, result_dir,
             alignment=None,
             seqs=None,
             filename=None,
             length=-1):
        """Evaulate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            alignment (str): if not None, optimize poses by
                - scale: optimize scale factor for trajectory alignment and evaluation
                - scale_7dof: optimize 7dof for alignment and use scale for trajectory evaluation
                - 7dof: optimize 7dof for alignment and evaluation
                - 6dof: optimize 6dof for alignment and evaluation
            seqs (list/None):
                - None: Evalute all available seqs in result_dir
                - list: list of sequence indexs to be evaluated
        """
        seq_list = ["alpine", "C1", "C3", "C4", "C5", "demo_coax", "demo_mars",
                    "nonoverlapping_test", "nonoverlapping_cave", "chariot1", "chariot2", "chariot3", "chariot4",
                    "traj_3", "traj_4", "sar1", "tour_butte2", "grand_tour", "loops", "building3", "raspi1"]

        # Initialization
        self.gt_dir = gt_dir
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create evaluation list
        if seqs is None:
            available_seqs = [os.path.basename(x).rsplit(
                ".", 1)[0] for x in glob(result_dir+'/*.csv')]
            self.eval_seqs = [i for i in available_seqs if i in seq_list]
        else:
            self.eval_seqs = seqs

        # evaluation
        for i in self.eval_seqs:

            self.cur_seq = i
            # Read pose txt
            self.cur_seq = '{}'.format(i)
            gt_file_name = '{}.csv'.format(i)

            if (filename is None):
                result_file_name = '{}.txt'.format(i)
            else:
                result_file_name = filename

            poses_result, timestamp_result = load_poses_from_txt_ov(
                result_dir + "/" + result_file_name)
            poses_gt, timestamp_gt = load_poses_from_csv_isae(
                self.gt_dir + "/" + gt_file_name)
            self.result_file_name = result_dir + result_file_name

            df_result = pd.DataFrame({
                "timestamp": timestamp_result
            })

            df_gt = pd.DataFrame({
                "timestamp": timestamp_gt
            })

            df_gt["timestamp"] += 0

            poses_gt_sync = {}
            poses_result_new = {}
            counter = 0
            idx_res = 0
            for ts in df_result['timestamp']:
                pose_res = poses_result[idx_res]
                idx_res += 1
                idx = df_gt['timestamp'].sub(float(ts)).abs().idxmin()

                if (df_gt['timestamp'].sub(float(ts)).abs().min() > 311111111110810880):
                    continue

                poses_result_new[counter] = pose_res
                poses_gt_sync[counter] = poses_gt[idx]

                counter += 1
            poses_gt = poses_gt_sync
            poses_result = poses_result_new

            # Pose alignment to first frame
            idx_0 = sorted(list(poses_result.keys()))[0]
            pred_0 = poses_result[idx_0]
            gt_0 = poses_gt[idx_0]
            for cnt in poses_result:
                poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

            if alignment == "scale":
                poses_result = self.scale_optimization(poses_gt, poses_result)
            elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
                # get XYZ
                xyz_gt = []
                xyz_result = []
                for cnt in poses_result:
                    xyz_gt.append(
                        [poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                    xyz_result.append(
                        [poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
                xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                xyz_result = np.asarray(xyz_result).transpose(1, 0)

                r, t, scale = umeyama_alignment(
                    xyz_result, xyz_gt, alignment != "6dof")

                align_transformation = np.eye(4)
                align_transformation[:3:, :3] = r
                align_transformation[:3, 3] = t

                for cnt in poses_result:
                    poses_result[cnt][:3, 3] *= scale
                    if alignment == "7dof" or alignment == "6dof":
                        poses_result[cnt] = align_transformation @ poses_result[cnt]

            poses_gt = full_alignment(poses_result, poses_gt)
            print(" scale = " + str(scale))

            df_result = pd.DataFrame({
                "pose": poses_result
            })
            df_gt = pd.DataFrame({
                "pose": poses_gt
            })

            # save df to pickle
            df_result.to_pickle("results.pkl")
            df_gt.to_pickle("gt.pkl")

            # Compute ATE
            ate = self.compute_ATE(poses_gt, poses_result)
            seq_ate.append(ate)
            print(i)
            print("ATE (m): ", ate)

            # Compute RPE
            rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
            # avg_scale_err = np.mean(np.asarray(self.compute_scale_error(poses_gt, poses_result)))
            seq_rpe_trans.append(rpe_trans)
            seq_rpe_rot.append(rpe_rot)
            print("RPE (m): ", rpe_trans)
            print("RPE (deg): ", rpe_rot * 180 / np.pi)
            # print("Scale : ", avg_scale_err)

            # Plotting
            self.plot_path_dir = result_dir + "/plot_path"
            self.plot_trajectory(poses_gt, poses_result, self.cur_seq)
            self.compute_scale_ratio(poses_gt, poses_result)

        return [ate, rpe_trans]

    def eval_cov(self, gt_dir, result_dir,
                 alignment=None,
                 seqs=None,
                 filename=None,
                 length=-1):
        """Evaulate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            alignment (str): if not None, optimize poses by
                - scale: optimize scale factor for trajectory alignment and evaluation
                - scale_7dof: optimize 7dof for alignment and use scale for trajectory evaluation
                - 7dof: optimize 7dof for alignment and evaluation
                - 6dof: optimize 6dof for alignment and evaluation
            seqs (list/None):
                - None: Evalute all available seqs in result_dir
                - list: list of sequence indexs to be evaluated
        """
        seq_list = ["alpine", "C1", "C3", "C4", "C5", "demo_coax", "demo_mars",
                    "nonoverlapping_test", "nonoverlapping_cave", "chariot1", "chariot2", "chariot3", "chariot4",
                    "traj_3", "traj_4", "sar1", "tour_butte2", "grand_tour", "loops", "building3", "raspi1"]

        # Initialization
        self.gt_dir = gt_dir
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create evaluation list
        if seqs is None:
            available_seqs = [os.path.basename(x).rsplit(
                ".", 1)[0] for x in glob(result_dir+'/*.csv')]
            self.eval_seqs = [i for i in available_seqs if i in seq_list]
        else:
            self.eval_seqs = seqs

        # evaluation
        for i in self.eval_seqs:

            self.cur_seq = i
            # Read pose txt
            self.cur_seq = '{}'.format(i)
            gt_file_name = '{}.csv'.format(i)
            cov_file_name = '{}_cov.csv'.format(i)

            if (filename is None):
                result_file_name = '{}.csv'.format(i)
            else:
                result_file_name = filename

            poses_result, timestamp_result = load_poses_from_csv_isae(
                result_dir + "/" + result_file_name)
            poses_gt, timestamp_gt = load_poses_from_csv_EUROC(
                self.gt_dir + "/" + gt_file_name)
            self.result_file_name = result_dir + result_file_name

            df_result = pd.DataFrame({
                "timestamp": timestamp_result
            })

            df_gt = pd.DataFrame({
                "timestamp": timestamp_gt
            })

            poses_gt_sync = []
            poses_result_new = []
            ts_gt_sync = []
            ts_result_sync = []
            idx_res = 0
            for ts in df_result['timestamp']:
                pose_res = poses_result[idx_res]
                idx_res += 1
                idx = df_gt['timestamp'].sub(float(ts)).abs().idxmin()

                if (df_gt['timestamp'].sub(float(ts)).abs().min() > 311111111110810880):
                    continue

                poses_result_new.append(pose_res)
                ts_result_sync.append(ts)
                poses_gt_sync.append(poses_gt[idx])
                ts_gt_sync.append(df_gt['timestamp'][idx])

            poses_gt = poses_gt_sync
            poses_result = poses_result_new

            # Pose alignment to first frame
            pred_0 = poses_result[0]
            gt_0 = poses_gt[0]
            for idx, pose in enumerate(poses_result):
                poses_result[idx] = np.linalg.inv(pred_0) @ pose
                poses_gt[idx] = np.linalg.inv(gt_0) @ poses_gt[idx]

            if alignment == "scale":
                poses_result = self.scale_optimization(poses_gt, poses_result)
            elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
                # get XYZ
                xyz_gt = []
                xyz_result = []
                for idx in range(len(poses_result)):
                    xyz_gt.append(
                        [poses_gt[idx][0, 3], poses_gt[idx][1, 3], poses_gt[idx][2, 3]])
                    xyz_result.append(
                        [poses_result[idx][0, 3], poses_result[idx][1, 3], poses_result[idx][2, 3]])
                xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                xyz_result = np.asarray(xyz_result).transpose(1, 0)

                r, t, scale = umeyama_alignment(
                    xyz_result, xyz_gt, alignment != "6dof")

                align_transformation = np.eye(4)
                align_transformation[:3:, :3] = r
                align_transformation[:3, 3] = t

                for cnt in range(len(poses_result)):
                    poses_result[cnt][:3, 3] *= scale
                    if alignment == "7dof" or alignment == "6dof":
                        poses_result[cnt] = align_transformation @ poses_result[cnt]

            # Compute dictionnaries for built-in functions
            results_dict = {i: poses_result[i]
                            for i in range(len(poses_result))}
            gt_dict = {i: poses_gt[i] for i in range(len(poses_gt))}
            gt_dict = full_alignment(results_dict, gt_dict)

            df_result = pd.DataFrame({
                "timestamp": ts_result_sync,
                "pose": poses_result
            })
            df_gt = pd.DataFrame({
                "timestamp": ts_gt_sync,
                "pose": poses_gt
            })

            # Read covariance dataframe
            df_cov = pd.read_csv(result_dir + "/" + cov_file_name)

            # Find corespondences
            for idx, line in df_cov.iterrows():
                ts = line['timestamp (ns)']
                ts_prev = line[' timestamp previous (ns)']
                if ((df_result['timestamp'].sub(float(ts)).abs().min() == 0) & (df_result['timestamp'].sub(float(ts)).abs().min() == 0)):
                    idx = df_result['timestamp'].sub(float(ts)).abs().idxmin()
                    T_w_f = df_result['pose'][idx]
                    T_w_f_gt = df_gt['pose'][idx]
                    idx = df_result['timestamp'].sub(
                        float(ts_prev)).abs().idxmin()
                    T_w_fp = df_result['pose'][idx]
                    T_w_fp_gt = df_gt['pose'][idx]
                    T_f_fp = np.linalg.inv(T_w_f) @ T_w_fp
                    T_f_fp_gt = np.linalg.inv(T_w_f_gt) @ T_w_fp_gt
                    err = T_f_fp @ np.linalg.inv(T_f_fp_gt)
                    cov_gt = np.multiply(err[0:3, 3],  err[0:3, 3])
                    print("Cov gt : ")
                    print(cov_gt)
                    cov_est = np.array(
                        [line[' cov(33)'], line[' cov(44)'], line[' cov(55)']])
                    print("Cov est : ")
                    print(cov_est)

            # Compute ATE
            ate = self.compute_ATE(gt_dict, results_dict)
            seq_ate.append(ate)
            print(i)
            print("ATE (m): ", ate)

            # Compute RPE
            rpe_trans, rpe_rot = self.compute_RPE(gt_dict, results_dict)
            # avg_scale_err = np.mean(np.asarray(self.compute_scale_error(poses_gt, poses_result)))
            seq_rpe_trans.append(rpe_trans)
            seq_rpe_rot.append(rpe_rot)
            print("RPE (m): ", rpe_trans)
            print("RPE (deg): ", rpe_rot * 180 / np.pi)
            # print("Scale : ", avg_scale_err)

            # Plotting
            fig = plt.figure()
            ax = plt.gca()
            ax.set_aspect('equal')

            pos_xy = []
            for idx, line in df_cov.iterrows():
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = df_result['pose'][idx]
                pos_xy.append([pose[1, 3],  pose[0, 3]])
                
                # Add covariance ellipses
                if (idx % 1 == 0):
                    
                    if (df_cov[' cov(33)'][idx] == 1):
                       continue
                    cov_vec = np.array([df_cov[' cov(33)'][idx], df_cov[' cov(44)'][idx], df_cov[' cov(55)'][idx]])
                    R_gt = Rotation.from_matrix(pose[:3:, :3])
                    theta = R_gt.as_euler('xyz', True)[2]
                    ellipse = Ellipse(xy=(pose[1, 3],  pose[0, 3]), width=cov_vec[1]*0.25, height=cov_vec[0]*0.25, angle=theta,
                        edgecolor='r', fc='None', lw=2)
                    ax.add_patch(ellipse)
            pos_xy = np.asarray(pos_xy)
            plt.plot(pos_xy[:, 0],  pos_xy[:, 1], color='black')
            
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.savefig("propagation.pdf",bbox_inches = 'tight')
            plt.show()
            plt.close(fig)
        
        return [ate, rpe_trans]

    def plot_trajectory(self, poses_gt, poses_result, seq):
        """Plot trajectory for both GT and prediction
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xy = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xy.append([pose[0, 3],  pose[1, 3]])
            pos_xy = np.asarray(pos_xy)
            plt.plot(pos_xy[:, 0],  pos_xy[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{}".format(seq)
        fig_pdf = self.plot_path_dir + "/" + png_title + ".pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def plot_scale_error(self, poses_gt, poses_result, seq):

        fig = plt.figure()
        ax = plt.gca()
        fontsize_ = 12

        scale_ratio = self.compute_scale_ratio(poses_gt, poses_result)

        # Write in a csv
        # scale_df = pd.DataFrame(scale_ratio)
        # scale_df.to_csv('scale.csv')

        plt.plot(scale_ratio, "*", color='red')
        plt.xlabel('Keyframes', fontsize=fontsize_)
        plt.ylabel('Scale ratio', fontsize=fontsize_)
        ax.set_ylim([0, 2])
        plt.show()
