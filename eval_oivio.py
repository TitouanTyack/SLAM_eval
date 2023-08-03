import argparse

from oivio_odometry import OivioEvalOdom

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
args = parser.parse_args()

eval_tool = OivioEvalOdom()
gt_dir = "dataset/OIVIO/"
result_dir = args.result

continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
if continue_flag == "y":
    eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        )
else:
    print("Double check the path!")