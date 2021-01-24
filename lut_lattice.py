import argparse
import gzip
import json
import pickle
import random
from itertools import product
from pathlib import Path
from tqdm import tqdm

import colour
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.spatial import Delaunay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data_path', type=Path)
    parser.add_argument('model_file_path', type=Path)
    parser.add_argument('lut_path', type=Path)
    parser.add_argument('--color_space', type=str, default='lab')
    parser.add_argument('--lut_size', type=int, default=33)
    args = parser.parse_args()

    print('load training data')

    with gzip.open(args.training_data_path, 'rb') as f:
        raw, jpeg = pickle.load(f)
    raw = np.array(raw, dtype=np.float32)
    jpeg = np.array(jpeg, dtype=np.float32)

    raw /= 255
    jpeg /= 255

    if args.color_space == 'lab':
        raw = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(raw))
        jpeg = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(jpeg))

    print('load xgboost models')

    with open(args.model_file_path, 'rb') as f:
        models = pickle.load(f)

    print('lattice regression')

    print('define A')

    lattice_size = args.lut_size
    lattice_points = np.linspace(0.0, 1.0, lattice_size)
    A = np.array([p for p in product(lattice_points,
                                    lattice_points,
                                    lattice_points)],
                dtype=np.float32)
    if args.color_space == 'lab':
        A = colour.sRGB_to_XYZ(A)
        A = colour.XYZ_to_Lab(A).astype(np.float32)

    print('compute W')

    tri = Delaunay(A)
    simplex_indices = tri.find_simplex(raw)
    weights = np.einsum('...ij,...j->...i',
                        tri.transform[simplex_indices, :-1],
                        raw - tri.transform[simplex_indices, -1])
    rests = 1.0 - weights.sum(axis=1)
    rests = rests.reshape((rests.size, 1))
    weights = np.hstack([weights, rests])

    M = lattice_size ** 3
    N = raw.shape[0]
    W = sparse.lil_matrix((M, N))

    W[tri.simplices[simplex_indices], [[i] * weights.shape[1] for i in range(N)]] = weights
        
    W = W.tocsr()

    print('compute L')

    E = sparse.lil_matrix((M, M), dtype=np.uint8)

    used_simplices = set()
    for i in tqdm(range(simplex_indices.size)):
        if simplex_indices[i] in used_simplices:
            continue
        
        simplex = tri.simplices[simplex_indices[i]]
        for j, s1 in enumerate(simplex):
            for s2 in simplex[(j + 1):]:
                E[s1, s2] = 1
                E[s2, s1] = 1
        used_simplices.add(simplex_indices[i])
    E = E.tocsr()

    ones_E = E.sum(axis=0).tolist()[0]
    ones_E_diag = sparse.diags(ones_E)
    L = (2 * (ones_E_diag - E) / E.sum())

    print('compute f~(A)')

    Y = jpeg.T
    fA = np.vstack([model.predict(A) for model in models])

    alpha = 1.0
    gamma = 1.0

    print('compute left hand')

    left_hand1 = (W @ W.T) / N
    left_hand2 = alpha * L
    left_hand3 = gamma / M * sparse.identity(M, format='csr')
    left_hand = left_hand1 + left_hand2 + left_hand3

    left_hand = left_hand.tocsc()

    print('compute right hand')

    right_hand1 = (Y @ W.T) / N
    right_hand2 = gamma / M * fA
    right_hand = right_hand1 + right_hand2

    right_hand = right_hand.astype(np.float32)

    print('compute inverse of left hand')

    left_hand_LU = sparse.linalg.splu(left_hand, diag_pivot_thresh=0)

    step = (lattice_size) ** 2
    e = sparse.identity(M, dtype=np.float32, format='lil')

    left_hand_inv = []
    for i in tqdm(range(0, M, step)):
        e_begin = i
        e_end = e_begin + step
        left_hand_inv.append(left_hand_LU.solve(e[:, e_begin:e_end].todense()).astype(np.float32))
        
    left_hand_inv = np.hstack(left_hand_inv)

    print('compute B')

    B = (right_hand @ left_hand_inv).T

    if args.color_space == 'lab':
        B = colour.Lab_to_XYZ(B)
        B = colour.XYZ_to_sRGB(B)

    B = np.clip(B, 0.0, 1.0)

    print('save lut')

    B = B.reshape((lattice_size, lattice_size, lattice_size, 3))
    lut = colour.LUT3D(B, size=lattice_size)
    colour.write_LUT(lut, args.lut_path)