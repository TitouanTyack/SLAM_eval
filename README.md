## SLAM_eval

# Install

python -m venv myenv
source myenv/bin/activate

python -m pip install pin
pip install matplotlib
pip install pandas
pip install scipy

# Run

python ./eval_isae.py --result path_to_odom_result --seqs traj_name --align 6dof
