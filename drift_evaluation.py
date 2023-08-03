import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oivio_odometry import OivioEvalOdom
from isae_odometry import IsaeEvalOdom

parser = argparse.ArgumentParser(description='EUROC evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory")
parser.add_argument('--align', type=str,
                    choices=['scale', 'scale_7dof', '7dof', '6dof'],
                    default=None,
                    help="alignment type")
parser.add_argument('--seqs',
                    nargs="+",
                    type=str,
                    help="sequences to be evaluated",
                    default=None)
parser.add_argument('--len',
                    nargs="+",
                    type=int,
                    help="length of the drift figure",
                    default=-1)
args = parser.parse_args()

eval_tool = IsaeEvalOdom()
gt_dir = "dataset/gazebo/"
result_dir = args.result

ate_list_marg = []
n_kf = []

for i in range(10, args.len[0], 5):
    [ate, rpe] = eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        filename="C4_marg.txt",
        length=i
    )
    ate_list_marg.append(ate)
    n_kf.append(i)

ate_list_nomarg = []

for i in range(10, args.len[0], 5):

    [ate, rpe] = eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        filename="C4_nothing.txt",
        length=i
    )
    ate_list_nomarg.append(ate)

ate_list_sparsif = []

for i in range(10, args.len[0], 5):

    [ate, rpe] = eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        filename="C4_spars.txt",
        length=i
    )
    ate_list_sparsif.append(ate)


# plt.boxplot([ate_list_nomarg, ate_list_marg, ate_list_sparsif], showfliers=False, labels = ['Discard', 'Marginalization', 'Sparsification'])
# plt.ylabel("ATE (m)")
# plt.show()

dict_ate = {'labels' : ['Discard', 'Marginalization', 'Sparsification'],
 'ate' : [ate_list_nomarg[-1], ate_list_marg[-1], ate_list_sparsif[-1]],
 'mean' : [np.mean(ate_list_nomarg), np.mean(ate_list_marg), np.mean(ate_list_sparsif)],
 'std' : [np.std(ate_list_nomarg), np.std(ate_list_marg), np.std(ate_list_sparsif)]}
df_ate = pd.DataFrame(dict_ate)
print(df_ate)

plt.errorbar(df_ate['labels'], df_ate['mean'], yerr=df_ate['std'], fmt='o', color='Black', elinewidth=3,capthick=3,errorevery=1, alpha=1, ms=4, capsize = 5)
plt.bar(df_ate['labels'], df_ate['ate'],tick_label = df_ate['labels'])##Bar plot
plt.ylabel('Average Performance') ##Label on Y axis
plt.show()


