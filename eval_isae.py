import argparse

from isae_odometry import IsaeEvalOdom

parser = argparse.ArgumentParser(description='ISAE evaluation')
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

eval_tool = IsaeEvalOdom()
gt_dir = "dataset/isae"
result_dir = args.result

eval_tool.eval(
    gt_dir,
    result_dir,
    alignment=args.align,
    seqs=args.seqs,
    )

