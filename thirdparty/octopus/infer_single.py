import os
import argparse
import tensorflow as tf
import keras.backend as K

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh
from model.octopus import Octopus
import cv2
import numpy as np
import pickle as pkl


def main(weights, name, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps):

    num = 8
    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=num)
    model.load(weights)

    base_path = "data/people_snapshot"
    file_paths = sorted(os.listdir(base_path + "/frames"))
    if not os.path.exists(base_path + "/vertices"):
        os.mkdir(base_path + "/vertices")

    for idx in range(len(file_paths)):
        file_path = file_paths[idx]
        if not os.path.exists('results/' + file_path):
            os.mkdir('results/' + file_path)
        if os.path.exists(base_path + "/vertices/" + file_path + "/frame_data.pkl"):
            continue
        frames_files = sorted(glob(os.path.join(base_path + '/frames/' + file_path, '*.png')))
        # segm_files = sorted(glob(os.path.join(base_path + '/segs/' + file_path, '*.png')))
        pose_files = sorted(glob(os.path.join(base_path + '/2d_joints/' + file_path + '/json', '*.json')))
        mask_files = sorted(glob(os.path.join(base_path + '/mask_mat/' + file_path, '*.png')))

        # if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
        #     exit('Inconsistent input.')

        # segmentations = [read_segmentation(f) for f in segm_files]
        masks = [cv2.imread(f) / 255.0 for f in mask_files]

        joints_2d, face_2d = [], []
        for f in pose_files:
            j, f = openpose_from_file(f)

            assert(len(j) == 25)
            assert(len(f) == 70)

            joints_2d.append(j)
            face_2d.append(f)

        if opt_pose_steps:
            print('Optimizing for pose...')
            model.opt_pose(None, masks, joints_2d, opt_steps=opt_pose_steps)

        if opt_shape_steps:
            print('Optimizing for shape...')
            model.opt_shape(None, masks, joints_2d, face_2d, opt_steps=opt_shape_steps)

        print('Estimating shape...')
        pred = model.predict(masks, joints_2d)


        for c in range(8):
            input_img = cv2.imread(frames_files[c])
            project = (np.repeat(pred['rendered'][c], 3, axis=2) * 255).astype('uint8')
            project_img = cv2.addWeighted(project, 0.3, input_img, 0.7, 0)
            cv2.imwrite('results/' + file_path + '/project_{}.png'.format(c), project_img)

            write_mesh('results/' + file_path + "/{}.obj".format(c), pred['vertices'][c], pred['faces'])

        if not os.path.exists(base_path + "/vertices/" + file_path):
            os.mkdir(base_path + "/vertices/" + file_path)
        pkl_data = {}
        pkl_data['vertices'] = pred['vertices']
        with open(base_path + "/vertices/" + file_path + '/frame_data.pkl', 'wb') as file:
            pkl.dump(pkl_data, file)

        print('{} Done.'.format(file_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--name',
        default='',
        type=str,
        help="Sample name")

    parser.add_argument(
        '--segm_dir',
        default='',
        type=str,
        help="Segmentation images directory")

    parser.add_argument(
        '--pose_dir',
        default='',
        type=str,
        help="2D pose keypoints directory")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=5, type=int,
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
        # default='weights/octopus_weights.hdf5',
        default='weights/pipline_model_v13.h5',
        help='Model weights file (*.hdf5)')

    args = parser.parse_args()
    main(args.weights, args.name, args.segm_dir, args.pose_dir, args.out_dir, args.opt_steps_pose, args.opt_steps_shape)
