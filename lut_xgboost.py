import argparse
import gzip
import pickle
from itertools import product
from pathlib import Path
from tqdm import tqdm

import colour
import numpy as np
import xgboost as xgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data_path', type=Path)
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

    learning_rate = 0.1
    max_depth = 20
    reg_lambda = 0.5
    n_estimators = 100

    print('train models')

    models = []
    for i in tqdm(range(3)):
        if i == 0:
            c = '(1, 0, 0)'
        elif i == 1:
            c = '(0, 1, 0)'
        else:
            c = '(0, 0, 1)'

        model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=max_depth,
                                reg_lambda=reg_lambda, n_estimators=n_estimators,
                                monotone_constraints=c, objective='reg:squarederror',
                                n_jobs=8)
        model.fit(raw, jpeg[:, i].copy())
        models.append(model)

    print('predict lut values')

    xi = np.linspace(0.0, 1.0, args.lut_size)
    xi = np.array([pi for pi in product(xi, xi, xi)])
    if args.color_space == 'lab':
        xi = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(xi))

    pred = [model.predict(xi) for model in models]
    pred = np.vstack(pred).transpose()
    if args.color_space == 'lab':
        pred = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(pred))
    pred= np.clip(pred, 0.0, 1.0)

    pred = pred.reshape((args.lut_size, args.lut_size, args.lut_size, 3))

    print('save lut')

    lut = colour.LUT3D(pred, size=args.lut_size)
    colour.write_LUT(lut, args.ut_path)

    print('save models')

    with open(f'{args.lut_path}.model.pkl', 'wb') as f:
        pickle.dump(models, f)