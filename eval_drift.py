import argparse
import pandas as pd
import numpy as np

from isae_odometry import IsaeEvalOdom

parser = argparse.ArgumentParser(description='Drift Evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory")
parser.add_argument('--seqs', 
                    nargs="+",
                    type=str, 
                    help="sequences to be evaluated",
                    default=None)
args = parser.parse_args()

gt_dir = "dataset/isae_drift/"
result_dir = args.result

continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
if continue_flag == "y":
    
    for seq in args.seqs:
        
        print(seq)
        gt_file_name = gt_dir + '{}.csv'.format(seq)
        
        # Load drift
        df_gt = pd.read_csv(gt_file_name)
        T_l0_lf = np.eye(4)
        for idx, row in df_gt.iterrows():
            T_l0_lf[0, 3] = row[' T_l0lf(03)']
            T_l0_lf[1, 3] = row[' T_l0lf(13)']
            T_l0_lf[2, 3] = row[' T_l0lf(23)']

            T_l0_lf[0:3, 0:3] = np.array([[row['T_l0lf(00)'], row[' T_l0lf(01)'], row[' T_l0lf(02)']],
                                    [row[' T_l0lf(10)'], row[' T_l0lf(11)'],
                                     row[' T_l0lf(12)']],
                                    [row[' T_l0lf(20)'], row[' T_l0lf(21)'], row[' T_l0lf(22)']]])
            
        # Load calib
        calib_file_name = gt_dir + 'calib.csv'
        df_calib = pd.read_csv(calib_file_name)
        T_l_f = np.eye(4)
        for idx, row in df_calib.iterrows():
            T_l_f[0, 3] = row[' T_lf(03)']
            T_l_f[1, 3] = row[' T_lf(13)']
            T_l_f[2, 3] = row[' T_lf(23)']

            T_l_f[0:3, 0:3] = np.array([[row['T_lf(00)'], row[' T_lf(01)'], row[' T_lf(02)']],
                                    [row[' T_lf(10)'], row[' T_lf(11)'],
                                     row[' T_lf(12)']],
                                    [row[' T_lf(20)'], row[' T_lf(21)'], row[' T_lf(22)']]])
        
        # Load results
        result_file_name = result_dir + '{}.csv'.format(seq)
        df_results = pd.read_csv(result_file_name)
        poses = []
        for idx, row in df_results.iterrows():
            P = np.eye(4)
            P[0, 3] = row[' T_wf(03)']
            P[1, 3] = row[' T_wf(13)']
            P[2, 3] = row[' T_wf(23)']

            P[0:3, 0:3] = np.array([[row[' T_wf(00)'], row[' T_wf(01)'], row[' T_wf(02)']],
                                    [row[' T_wf(10)'], row[' T_wf(11)'],
                                     row[' T_wf(12)']],
                                    [row[' T_wf(20)'], row[' T_wf(21)'], row[' T_wf(22)']]])
            poses.append(P)
        
        # Compute deltas
        T_w_f0 = poses[0]
        T_w_ff = poses[-1]
        T_f0_ff = np.linalg.inv(T_w_f0) @ T_w_ff
        gt = np.linalg.inv(T_l_f) @ T_l0_lf @ T_l_f
        
        # Compute drift
        err = T_f0_ff @ np.linalg.inv(gt)
        err_t = np.linalg.norm(err[0:3, 3])
        err_ang = np.arccos((np.trace(err[0:3, 0:3]) - 1) / 2) * (180 / np.pi)
        print("Translationnal error : " + str(err_t) + " m")
        print("Rotationnal error : " + str(err_ang) + " deg")
        
        

        
else:
    print("Double check the path!")