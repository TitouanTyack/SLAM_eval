import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation 

df = pd.read_csv('nonoverlapping_test.txt', sep = ' ')

dict_euroc = {'#timestamp' : [],
        'p_RS_R_x [m]' : [],
        'p_RS_R_y [m]' : [],
        'p_RS_R_z [m]' : [], 
        'q_RS_x []' : [],
        'q_RS_y []' : [],
        'q_RS_z []' : [],
        'q_RS_w []' : []}

for idx, line in df.iterrows():
    R = np.array([[line['R00'], line['R01'], line['R02']],
        [line['R10'], line['R11'], line['R12']],
        [line['R20'], line['R21'], line['R22']]])
    r = Rotation.from_matrix(R)
    q = r.as_quat()
    dict_euroc['#timestamp'].append((int)(line['timestamp']))
    dict_euroc['p_RS_R_x [m]'].append(line['x'])
    dict_euroc['p_RS_R_y [m]'].append(line['y'])
    dict_euroc['p_RS_R_z [m]'].append(line['z'])
    dict_euroc['q_RS_x []'].append(q[0])
    dict_euroc['q_RS_y []'].append(q[1])
    dict_euroc['q_RS_z []'].append(q[2])
    dict_euroc['q_RS_w []'].append(q[3])


df_euroc = pd.DataFrame.from_dict(dict_euroc)
df_euroc.to_csv('groundtruth.csv', index=False)
    
