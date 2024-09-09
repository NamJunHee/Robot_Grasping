import random

import cv2
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from ..dataset_processing.gripper_model import Gripper


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, binary_input = True , gripper=None):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.binary_input = binary_input
        self.grasp_files = []
        self.gripper = gripper

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        # if len(s.shape) == 2:
        #     return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        # else:
        #     return torch.from_numpy(s.astype(np.float32))
        return torch.from_numpy(np.array(s)).float()
        # return torch.from_numpy(np.array(s)).long()


    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def normalize(self, val, max_v, min_v):
        return (val - min_v) / (max_v - min_v)

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        # 1 point version 
        # normalize
        # norm_theta = self.normalize(bbs[0][0],360,-1)
        # norm_width = self.normalize(bbs[0][1],60,9)
        # norm_q = self.normalize(bbs[0][2],20,50)

        # # print(125, norm_theta,norm_width,norm_q)

        # # theta = self.numpy_to_torch(bbs[0][0])
        # # width = self.numpy_to_torch(bbs[0][1])
        # theta = self.numpy_to_torch(norm_theta)
        # width = self.numpy_to_torch(norm_width)
        # q = self.numpy_to_torch(norm_q)
        
        # print(75, type(pos_img), type(ang_img), type(width_img))
        # fig_1 = plt.figure(figsize=(10, 5))
        # # plt.gcf().set_size_inches(3, 2)
        # rows = 1
        # cols = 3

        # ax1 = fig_1.add_subplot(rows,cols,1)
        # # ax1.imshow(input_data[1])
        # ax1.imshow(pos_img)
        # ax1.set_title('pos_img')

        # ax2 = fig_1.add_subplot(rows,cols,2)
        # ax2.imshow(ang_img)
        # ax2.set_title('ang_img')

        # ax3 = fig_1.add_subplot(rows,cols,3)
        # ax3.imshow(width_img)
        # ax3.set_title('width_img')

        # plt.show()


        # stop()
        # width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)
        if self.binary_input == False:

            if self.include_depth and self.include_rgb:
                x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         rgb_img),
                        0
                    )
                )
            elif self.include_depth:
                x = self.numpy_to_torch(depth_img)
            elif self.include_rgb:
                x = self.numpy_to_torch(rgb_img)

        else:
            # numpyarray , (64,64)
            # print(type(depth_img), depth_img.shape)
            grip = Gripper([32,32], 0, self.gripper[0], self.gripper[1], self.gripper[2], (64,64))

            # grip.rotate(0)
            # grip.change_width(45)

            # contours = grip.get_contours()


            depth_state = np.clip((depth_img - depth_img.mean()) * 5 + 0.5, 0, 1)
            depth_state = np.uint8(depth_state * 255)
            grasp_base = np.zeros([64,64], dtype=np.uint8)
            # center, angle, width, rel, base, max_val
            contours_base = grip.get_contours(base=True)
            cv2.drawContours(grasp_base, contours_base, -1, (255), -1)
            depth_copy = depth_state.copy()
            grasp_base_not = cv2.bitwise_not(grasp_base)

            # print(33,depth_copy)
            # print(44,grasp_base_not)
            depth_mask = cv2.bitwise_or(depth_copy, grasp_base_not)
            highest = int(depth_mask.min())

            if highest == 0:
                highest = 1
            
            gripper_depth = int((depth_mask.max()-depth_mask.min())*0.2)

            state =  np.where(highest + gripper_depth >= depth_state[:,:], 1, 0)

            grasp_crop = np.zeros([64,64], dtype=np.uint8)
            contours = grip.get_contours()
            cv2.drawContours(grasp_crop, contours, -1, 2, -1)
            
            # grasp_state = np.clip(grasp_crop, 0, 255)

            grip_state = np.bitwise_or(grasp_crop,state)

            # fig_1 = plt.figure(figsize=(10, 5))
            # rows = 1
            # cols = 3

            # ax1 = fig_1.add_subplot(rows,cols,1)
            # ax1.imshow(state)
            # ax1.set_title('state')
            # ax2 = fig_1.add_subplot(rows,cols,2)
            # ax2.imshow(grasp_crop)
            # ax2.set_title('grasp_crop')
            # ax3 = fig_1.add_subplot(rows,cols,3)
            # ax3.imshow(grip_state)
            # ax3.set_title('grip_state')

            # plt.show()

            # only state
            # x = self.numpy_to_torch(np.expand_dims(state, 0))
            
            # state + gripper + sum
            
            x = self.numpy_to_torch(
                    np.concatenate((np.expand_dims(state, 0),np.expand_dims(grasp_crop, 0),np.expand_dims(grip_state, 0)),0))

            # print(2, x.shape)
            # stop()
            


        # multi output version
        output = self.numpy_to_torch(bbs)
        # print('output', output.size())
        

        return x, output, idx, rot, zoom_factor
        # return x, (theta,width,q), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)