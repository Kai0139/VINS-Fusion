from scipy.spatial.transform import Rotation as Rot
import numpy as np
from math import *

def transformation_matrix(q, t):
    """q = [qw, qx, qy, qz]
       t = [tx, ty, tz]
    """
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R0 = Rot.from_quat([qx, qy, qz, qw])

    eul = R0.as_euler('xyz')
    eul_deg = [degrees(i) for i in eul]
    # print("euler angle: {}".format(eul_deg))

    tx = t[0]
    ty = t[1]
    tz = t[2]

    tm = np.array([
        [0, 0, 0, tx],
        [0, 0, 0, ty],
        [0, 0, 0, tz],
        [0, 0, 0,  1],
    ])

    rotm = np.concatenate((R0.as_matrix(), np.array([[0, 0, 0]])), axis=0)
    rotm = np.concatenate((rotm, np.array([[0, 0, 0, 0]]).T), axis=1)
    # print(rotm)
    tf = rotm + tm
    return tf

T_l2r = np.array([
    [  0.9998,   -0.0036,    0.0210,    0.6150],
    [  0.0036,    1.0000,   -0.0010,   -0.0033],
    [ -0.0210,    0.0011,    0.9998,   -0.0339],
    [       0,         0,         0,    1.0000]
])

T_i2l = np.array([
    [       0,       0, 1.0000,  0.3100],
    [ -1.0000,       0,      0, -0.2000],
    [       0, -1.0000,      0, -0.2400],
    [       0,       0,      0,  1.0000]
])

print(T_l2r @ T_i2l)