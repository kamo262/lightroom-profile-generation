import argparse
import gzip
import pickle
from collections import defaultdict
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image


def generate_color_pair_data(raw_file_path, args):
    dst_path = args.training_data_path / f'{raw_file_path.stem}.pkl.gz'
    if dst_path.exists():
        return

    jpeg_file_path = args.jpeg_path / raw_file_path.name
    raw_img = np.asarray(Image.open(raw_file_path))
    jpeg_img = np.asarray(Image.open(jpeg_file_path))
    if raw_img.size != jpeg_img.size:
        print(raw_file_path, raw_img.shape, jpeg_img.shape)
        return

    diff_thresh = 20
    diff = np.abs(raw_img.astype(np.float32) - jpeg_img).sum() / raw_img.size
    if diff > diff_thresh:
        print(f'Difference of raw and jpeg of {raw_file_path} is significant. '
              'Skip this file.')
        return

    center = [raw_img.shape[0] // 2, raw_img.shape[1] // 2]
    crop_size = [int(c * args.used_area) for c in center]
    coordinates = [
        center[0] - crop_size[0], center[1] - crop_size[1],
        center[0] + crop_size[0], center[1] + crop_size[1]
    ]
    raw_img = raw_img[coordinates[0]:coordinates[2],
                      coordinates[1]:coordinates[3], :]

    jpeg_img = jpeg_img[coordinates[0]:coordinates[2],
                        coordinates[1]:coordinates[3], :]

    raw_img = raw_img[::args.stride, ::args.stride, :]
    raw_img = raw_img.reshape(raw_img.shape[0] * raw_img.shape[1], 3)

    jpeg_img = jpeg_img[::args.stride, ::args.stride, :]
    jpeg_img = jpeg_img.reshape(jpeg_img.shape[0] * jpeg_img.shape[1], 3)

    with gzip.open(dst_path, 'wb') as f:
        pickle.dump([raw_img, jpeg_img], f)


def quantize(v, q):
    return v // q * q


def summarize(file_path, color_pairs, nums):
    if 'training_data' in file_path.stem:
        return

    with gzip.open(file_path, 'rb') as f:
        raw_img, jpeg_img = pickle.load(f)

    if raw_img.shape[0] != jpeg_img.shape[0]:
        print(f'sizes of raw and jpeg of {file_path} are different')
        return

    raw_img = quantize(raw_img, args.quantize)
    jpeg_img = quantize(jpeg_img, args.quantize).astype(np.uint64)

    for i in range(raw_img.shape[0]):
        c = tuple(raw_img[i, :])
        color_pairs[c] += jpeg_img[i, :]
        nums[c] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', type=Path)
    parser.add_argument('jpeg_path', type=Path)
    parser.add_argument('training_data_path', type=Path)
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--used_area', type=float, default=0.5)
    parser.add_argument('--quantize', type=int, default=2)
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()

    args.training_data_path.mkdir(exist_ok=True)

    print('generate color pair data for each image')
    raw_file_paths = [p for p in sorted(args.raw_path.glob(f'*{args.ext}'))]
    Parallel(n_jobs=-1, verbose=10)([
        delayed(generate_color_pair_data)(raw_file_path, args)
        for raw_file_path in raw_file_paths
    ])

    print('summarize color pair data')
    file_paths = sorted(args.training_data_path.glob('*.pkl.gz'))
    color_pairs = defaultdict(lambda: np.zeros(3, dtype=np.uint64))
    nums = defaultdict(int)
    for file_path in tqdm(file_paths):
        summarize(file_path, color_pairs, nums)

    raw_colors = []
    jpeg_colors = []
    for k in tqdm(color_pairs.keys()):
        raw_colors.append(list(k))
        jpeg_colors.append((color_pairs[k] / nums[k]).astype(np.int32).tolist())

    print('save training data')
    with gzip.open(args.training_data_path / f'training_data.pkl.gz', 'wb') as f:
        pickle.dump([raw_colors, jpeg_colors], f)