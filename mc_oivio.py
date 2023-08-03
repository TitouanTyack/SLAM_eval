import argparse
import os
import numpy as np

from oivio_odometry import OivioEvalOdom

parser = argparse.ArgumentParser(description='Oivio evaluation')
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
args = parser.parse_args()

eval_tool = OivioEvalOdom()
gt_dir = "dataset/OIVIO/"
result_dir = args.result

for seq in args.seqs:
    folder_path = result_dir + "/" + seq

    ates = []
    rpes = []
    for filename in os.listdir(folder_path):
        print(filename)
        [ate, rpe] = eval_tool.eval(
            gt_dir,
            result_dir,
            alignment=args.align,
            seqs=args.seqs,
            filename=seq + "/" + filename)
        ates.append(ate)
        rpes.append(rpe)

    print("----------------------")
    print(seq)
    print("Avg ate (m) : " + str(np.average(ates)))
    print("Avg rpe (m) : " + str(np.average(rpe)))
    print("Std ate (m) : " + str(np.std(ates)))
    print("Std rpe (m) : " + str(np.std(rpe)))
    print("----------------------")
