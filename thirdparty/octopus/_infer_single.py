import os
import sys
import argparse
import tensorflow as tf
import keras.backend as K

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus

import pickle as pkl
import cv2
import numpy as np

def main(weights, root_dir, name, out_dir, opt_pose_steps, opt_shape_steps):

    num = 8
    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=num)
    model.load(weights)

    base_path = root_dir 
    file_paths = sorted(os.listdir(base_path + "frames_mat"))

    if not os.path.exists(base_path + "params"):
        os.mkdir(base_path + "params")

    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    # name = file_paths[idx]
    if not os.path.exists(base_path + "params/" + name):
        os.mkdir(base_path + "params/" + name)
    if not os.path.exists("./results/" + name):
        os.mkdir("./results/" + name)

    # if os.path.exists(base_path + "/vertices/" + name + "/frame_data.pkl"):
    #     continue
    if os.path.exists(base_path + "params/" + name + "/param.pkl"):
        sys.exit(0)
    frames_files = sorted(glob(os.path.join(base_path + 'frames_mat/' + name, '*.png')))
    # segm_files = sorted(glob(os.path.join(base_path + '/segs/' + name, '*.png')))
    pose_files = sorted(glob(os.path.join(base_path + '2d_joints/' + name + '/json', '*.json')))
    mask_files = sorted(glob(os.path.join(base_path + 'mask_mat/' + name, '*.png')))

    if len(mask_files) != len(pose_files) or len(mask_files) == len(pose_files) == 0:
        exit(name + ' Inconsistent input.')

    # segmentations = [read_segmentation(f) for f in segm_files]

    print('{} Begin.'.format(name))

    data_pack = {}
    data_pack['poses'] = np.zeros([int(len(frames_files) / 8) * 8,72])
    data_pack['betas'] = np.zeros([int(len(frames_files) / 8) * 8,10])
    data_pack['trans'] = np.zeros([int(len(frames_files) / 8) * 8,3])

    for jIdx in range(len(frames_files) / 8):
        model.load(weights)

        print(jIdx)
        begin = jIdx
        end = len(frames_files) - len(frames_files)%8
        step = (len(frames_files) / 8)
        pose_file = [pose_files[iIdx] for iIdx in range(begin, end, step) ]
        mask_file = [mask_files[iIdx] for iIdx in range(begin, end, step) ]
        frame_file = [frames_files[iIdx] for iIdx in range(begin, end, step) ]
        # print(pose_file)
        seq_idx = [iIdx for iIdx in range(begin, end, step)]

        masks = [cv2.resize(cv2.imread(f), (1080, 1080)) / 255.0 for f in mask_file]

        joints_2d, face_2d = [], []
        for f in pose_file:
            j, f = openpose_from_file(f, resolution=(1024, 1024))

            assert(len(j) == 25)
            assert(len(f) == 70)

            joints_2d.append(j)
            face_2d.append(f)

        if opt_pose_steps:
            print('Optimizing for pose...')
            model.opt_pose(None, masks ,joints_2d, opt_steps=opt_pose_steps)

        if opt_shape_steps:
            print('Optimizing for shape...')
            model.opt_shape(None, masks, joints_2d, face_2d, opt_steps=opt_shape_steps)

        print('Estimating shape...')
        pred = model.predict(masks, joints_2d)

        # if jIdx == 0:
        for l in range(num):
            data_pack['poses'][seq_idx[l]] = pred['poses'][l]
            data_pack['betas'][seq_idx[l]] = pred['betas'] # [l] !!!
            data_pack['trans'][seq_idx[l]] = pred['trans'][l]

        for num_idx in range(num):
            input_img = cv2.imread(frame_file[num_idx])
            project = (np.repeat(pred['rendered'][num_idx], 3, axis=2) * 255).astype('uint8')
            input_img = cv2.resize(input_img, (project.shape[0], project.shape[1]))
            project_img = cv2.addWeighted(project, 0.3, input_img, 0.7, 0)
            cv2.imwrite('./results/' + name + '/project_{}.png'.format(seq_idx[num_idx]), project_img)

            # write_mesh('./results/' + name + '/{}.obj'.format(seq_idx[num_idx]), pred['vertices'][num_idx], pred['faces'])

    with open(base_path + "params/" + name + '/param.pkl', 'wb') as file:
        pkl.dump(data_pack, file)



    print('{} Done.'.format(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root_dir',
        default='',
        type=str,
        help="Sample name")

    parser.add_argument(
        '--name',
        default='',
        type=str,
        help="Sample name")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=30, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=100, type=int,
        help="Optimization steps")

    parser.add_argument(
        '--out_dir', '-od',
        default='out',
        help='Output directory')

    parser.add_argument(
        '--weights', '-w',
        default='weights/octopus_weights.hdf5',
        # default='weights/pipline_model_v13.h5',

        help='Model weights file (*.hdf5)')

    args = parser.parse_args()
    main(args.weights, args.root_dir, args.name, args.out_dir, args.opt_steps_pose, args.opt_steps_shape)
