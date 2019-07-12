import os
import shutil
import argparse
import subprocess
import numpy as np

from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--pre_path',    required=True)
parser.add_argument('--gt_path',     required=True)
parser.add_argument('--save_path',   required=True)
parser.add_argument('--seed_path',   default=None)
parser.add_argument('--D', type=int, default=9000)
parser.add_argument('--no_eval', default=False, action='store_true')
args = parser.parse_args()


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def evaluation(file_name):
    gt_path = os.path.join(args.gt_path, file_name)
    pre_path = os.path.join(args.pre_path, file_name.replace('off', 'xyz'))
    assert os.path.exists(gt_path)
    assert os.path.exists(pre_path)
    if os.path.exists(os.path.join(args.save_path, file_name.replace('.off', '_density.xyz'))) and \
        os.path.exists(os.path.join(args.save_path, file_name.replace('.off', '_point2mesh_distance.xyz'))) and \
            os.path.exists(os.path.join(args.save_path, file_name.replace('.off', '_sampling_seed.txt'))):
        return

    bash = './evaluation {} {}'.format(gt_path, pre_path)
    if args.seed_path:
        seed_path = os.path.join(args.seed_path, file_name.replace('.off', '_sampling_seed.txt'))
        if not os.path.exists(seed_path):
            return
        bash = '{} {} {}'.format(bash, 'F', seed_path)
    else:
        bash = '{} {} {}'.format(bash, 'D', args.D)

    p = subprocess.Popen(bash.split())
    p.wait()
    if args.seed_path is None:
        if not os.path.exists(os.path.join(args.pre_path,  file_name.replace('.off', '_sampling_seed.txt'))):
            return
        shutil.move(os.path.join(args.pre_path,  file_name.replace('.off', '_sampling_seed.txt')),
                    os.path.join(args.save_path, file_name.replace('.off', '_sampling_seed.txt')))
    shutil.move(os.path.join(args.pre_path,  file_name.replace('.off', '_density.xyz')),
                os.path.join(args.save_path, file_name.replace('.off', '_density.xyz')))
    shutil.move(os.path.join(args.pre_path,  file_name.replace('.off', '_point2mesh_distance.xyz')),
                os.path.join(args.save_path, file_name.replace('.off', '_point2mesh_distance.xyz')))


files = os.listdir(args.gt_path)
if not args.no_eval:
    Pool(8).map(evaluation, files)

distances = []
for file in files:
    path = os.path.join(args.save_path, file.replace('.off', '_point2mesh_distance.xyz'))
    if os.path.exists(path):
        distances.append(np.loadtxt(path))
distances = np.vstack(distances)
print('Mean: {}, Std: {}'.format(np.mean(distances[:, -1]), np.std(distances[:, -1])))

density = []
for file in files:
    path = os.path.join(args.save_path, file.replace('.off', '_density.xyz'))
    if os.path.exists(path):
        density.append(np.loadtxt(path))
density = np.vstack(density)
nuc = np.sqrt(np.mean((density - np.mean(density, axis=0, keepdims=True))**2, axis=0))
for p, n in zip([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5], nuc):
    print('{}%: {}'.format(p, n))
