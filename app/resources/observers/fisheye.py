"""
@file   fisheye.py
@author Xinyu Cai, Shanghai AI Lab
@brief  
"""

import math
import torch
import numpy as np

def fish_eye_distort(src: torch.Tensor):
    # K = cam.intr.mat_3x3().cpu().numpy()
    h,w = src.shape[0:2]
    fx = 318.44998905930794 / 1280 * w
    fy = 317.8314899911656 / 960 * h
    cx = 636.2089399611955 / 1280 * w
    cy = 481.71423781914115 / 960 * h
    K=np.array([[fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]])
    D=np.array([0.18198802503702904, -0.04198598106075817, 0.010013633995507613, -0.0025294664427881705])
    IK = np.linalg.inv(K)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    import math
    dstImg = np.zeros(src.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            _x = i*IK[0][1] + IK[0][2] + j*IK[0][0]
            _y = i*IK[1][1] + IK[1][2] + j*IK[1][0]
            _w = i*IK[2][1] + IK[2][2] + j*IK[2][0]
            a = _x / _w
            b = _y / _w
            r = math.sqrt(a*a + b*b)
            theta = math.atan(r)
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta4 * theta4
            theta_d = theta * (1 + D[0]*theta2 + D[1]*theta4 + D[2]*theta6 + D[3]*theta8)
            scale = 1.0 if r == 0 else theta_d / r
            u = int(fx * a * scale + cx)
            v = int(fy * b * scale + cy)
            dstImg[v][u] = src[i][j]
    return dstImg