import os
import scipy.misc as misc
import glob
import numpy as np


def mask_image_generator(image_dir):
    folder_list = os.listdir(image_dir)

    for folder in folder_list:
        mask_path = os.path.join(image_dir, folder, os.listdir(os.path.join(image_dir, folder))[-1])
        mask_name_list = glob.glob(os.path.join(mask_path, '*.png'))
        first_image_flag = True
        for mask_name in mask_name_list:
            mask = misc.imread(mask_name)
            if first_image_flag:
                final_mask = np.zeros(mask.shape)
            first_image_flag = False
            final_mask = np.concatenate([np.expand_dims(final_mask, 2),np.expand_dims(mask, 2)], axis=2)
            final_mask = np.max(final_mask, axis=2)
            print final_mask.shape
        final_file_name = os.path.join(mask_path, 'mask.png')
        misc.imsave(final_file_name, final_mask)


image_dir = './data/stage1_train'

