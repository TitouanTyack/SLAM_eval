import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from scipy.spatial.transform import Rotation
import os
from glob import glob

from kitti_odometry import EvalOdom, umeyama_alignment

class EurocEvalOdom(EvalOdom):
    """Evaluate EUROC odometry result
    Usage example:
        vo_eval = EurocEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def load_poses_from_txt_ts(self, file_name):
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
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i!=""]
            withIdx = len(line_split) == 14
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row*4 + col + 1 + withIdx] 
            if withIdx:
                timestamps[cnt] = line_split[0]
                poses[cnt] = P
            else:
                timestamps[cnt] = line_split[0]
                poses[cnt] = P

        return poses, timestamps

    def load_poses_from_csv(self, file_name):
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

    def eval(self, gt_dir, result_dir, 
                alignment=None,
                seqs=None):
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
        seq_list = ["V1", "MH1", "V3", "V2"]

        # Initialization
        self.gt_dir = gt_dir
        seq_ate = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create evaluation list
        if seqs is None:
            available_seqs = [os.path.basename(x).rsplit( ".", 1 )[ 0 ] for x in glob(result_dir+'/*.txt')]
            self.eval_seqs = [i for i in available_seqs if i in seq_list]
        else:
            self.eval_seqs = seqs

        # evaluation
        for i in self.eval_seqs:
            self.cur_seq = i
            # Read pose txt
            self.cur_seq = '{}'.format(i)
            gt_file_name = '{}.csv'.format(i)
            result_file_name = '{}.txt'.format(i)

            poses_result, timestamp_result = self.load_poses_from_txt_ts(result_dir + "/" + result_file_name)
            poses_gt, timestamp_gt = self.load_poses_from_csv(self.gt_dir + "/" + gt_file_name)
            self.result_file_name = result_dir + result_file_name

            # # fit timestamp_gt and timestamp_result
            # timestamp_result = [timestamp_gt[0] + i * 5e7 for i in range(len(poses_result))]
            
            df_result = pd.DataFrame({
                "timestamp" : timestamp_result
            })
            df_gt = pd.DataFrame({
                "timestamp" : timestamp_gt
            })

            poses_gt_sync = {}
            poses_results_sync = {}
            counter = 0
            counter_gt = 0 
            # for ts in df_result['timestamp']:

            #     idx = df_gt['timestamp'].sub(float(ts)).abs().idxmin()

            #     if (df_gt['timestamp'].sub(float(ts)).abs()[idx] * 1e-9 > 0.05):
            #         continue

            #     poses_results_sync[counter] = poses_result[counter]
            #     poses_gt_sync[counter] = poses_gt[idx]

            #     # We ignore rotation on EUROC
            #     # poses_results_sync[counter][0:3, 0:3] = np.eye(3)
                
            #     counter += 1

            for ts in df_gt['timestamp']:
                counter_gt += 1
                idx = df_result['timestamp'].sub(float(ts)).abs().idxmin()
                if (df_result['timestamp'].sub(float(ts)).abs()[idx] * 1e-9 > 0.1):
                    continue

                poses_results_sync[counter] = poses_result[idx]
                poses_gt_sync[counter] = poses_gt[counter_gt]

                # We ignore rotation on EUROC
                poses_results_sync[counter][0:3, 0:3] = np.eye(3)
                
                counter += 1

            poses_result = poses_results_sync
            poses_gt = poses_gt_sync


            # Pose alignment to first frame
            # idx_0 = sorted(list(poses_result.keys()))[0]
            # pred_0 = poses_result[idx_0]
            # gt_0 = poses_gt[idx_0]
            # for cnt in poses_result:
            #     poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
            #     poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

            if alignment == "scale":
                poses_result = self.scale_optimization(poses_gt, poses_result)
            elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
                # get XYZ
                xyz_gt = []
                xyz_result = []
                for cnt in poses_result:
                    xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                    xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
                xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                xyz_result = np.asarray(xyz_result).transpose(1, 0)

                r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment!="6dof")

                align_transformation = np.eye(4)
                align_transformation[:3:, :3] = r
                align_transformation[:3, 3] = t
                print(align_transformation)
                
                for cnt in poses_result:
                    poses_result[cnt][:3, 3] *= scale
                    if alignment=="7dof" or alignment=="6dof":
                        poses_result[cnt] = align_transformation @ poses_result[cnt]

            # Compute ATE
            print(i)
            ate = self.compute_ATE(poses_gt, poses_result)
            seq_ate.append(ate)
            print("ATE (m): ", ate)

            # Compute RPE
            rpe_trans, rpe_rot = self.compute_RPE(poses_gt, poses_result)
            seq_rpe_trans.append(rpe_trans)
            seq_rpe_rot.append(rpe_rot)
            print("RPE (m): ", rpe_trans)
            print("RPE (deg): ", rpe_rot * 180 /np.pi)

            # Plotting
            self.plot_path_dir = result_dir + "/plot_path"
            self.plot_trajectory(poses_gt, poses_result, self.cur_seq)
    
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
            pos_xz = []
            frame_idx_list = sorted(poses_dict["Ours"].keys())
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3],  pose[1, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0],  pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{}".format(seq)
        fig_pdf = self.plot_path_dir + "/" + png_title + ".pdf"
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
