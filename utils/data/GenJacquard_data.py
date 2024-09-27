import glob
import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_processing import grasp, image

from .grasp_data import GraspDatasetBase
from utils.dataset_processing.gripper_model import Gripper


def masked_uniform_sampling(mask, num_samples=10):
    indices = np.flatnonzero(mask)
    np.random.shuffle(indices)
    output = np.zeros(mask.shape[0]*mask.shape[1], dtype=bool)
    output[indices[:num_samples]] = True
    return output.reshape(mask.shape)

class GenJacquardDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """

    def __init__(self, file_path, img_size, ds_rotate=0, min_width=12, max_width=60, training=True, n_train_samples=25,
                 **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GenJacquardDataset, self).__init__(**kwargs)
        
        self.img_size = img_size
        self.grasp_files = glob.glob(os.path.join(file_path, '*', '*_label.txt'))
        # for test
        # self.grasp_files = [f for f in self.grasp_files if 1 <= int(os.path.basename(os.path.dirname(f))) <= 300]
        self.grasp_files.sort()
        

        self.not_none_ids = []
        for idx, file in enumerate(self.grasp_files):
            with open(file) as f:
                line = f.readline().strip()
                if (str.lower(line) != 'none') and (len(line) > 4):
                    self.not_none_ids.append(idx)
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('label.txt', 'depth.npy') for f in self.grasp_files]
        self.rgb_files = [f.replace('depth.npy', 'RGB.png') for f in self.depth_files]
        self.training = training
        self.n_train_samples = n_train_samples

        self.min_w, self.max_w = (min_width - 8)/39., (max_width - 8)/39.
        self.min_a, self.max_a = 1/361., 1.
        self.step_w, self.step_a = (2/39., 5/361.) # if self.training else (2/39., 5/361.)
        self.arr_w = np.arange(self.min_w + self.step_w / 2, self.max_w, self.step_w)
        self.arr_a = np.arange(self.min_a + self.step_a / 2, self.max_a, self.step_a)
        self.grid_a, self.grid_w = np.meshgrid(self.arr_a, self.arr_w) # create a grid search space for width and angle

        # create a gripper mask for each pair (width, angle)
        self.gripper_masks = np.zeros((len(self.arr_w), len(self.arr_a), self.img_size, self.img_size))
        for idw in range(len(self.arr_w)):
            for ida in range(len(self.arr_a)):
                self.gripper_masks[idw, ida] = self.get_gripper_mask(width=self.arr_w[idw] * 39 + 8, angle=self.arr_a[ida] * 361 - 1)

    def get_gripper_mask(self, width, angle):
        gripper_mask = np.zeros((self.img_size, self.img_size))
        grip = Gripper([32, 32], 0, self.gripper[0], self.gripper[1], self.gripper[2], (self.img_size, self.img_size))
        grip.change_width(width)
        grip.rotate(angle)
        contours = grip.get_contours()
        cv2.drawContours(gripper_mask, contours, -1, 1, -1)
        return gripper_mask

    def get_gtbb(self, idx, rot=0, zoom=1.0, normalize=True):
        gtbbs = grasp.GraspRectangles.load_from_cropdata_file(self.grasp_files[idx], normalize=normalize)
        # c = self.output_size // 2
        # gtbbs.rotate(rot, (c, c))
        # gtbbs.zoom(zoom, (c, c))
        gtbbs[:, 2] /= max(gtbbs[:, 2])
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        # depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img = np.load(self.depth_files[idx],allow_pickle=True)
        # depth_img.rotate(rot)
        # depth_img.normalise()
        # depth_img.zoom(zoom)
        # depth_img.resize((self.output_size, self.output_size))
        return depth_img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        # rgb_img.rotate(rot)
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])

    def __len__(self):
        return len(self.not_none_ids) #len(self.grasp_files)

    def __getitem__(self, idx):
        idx = self.not_none_ids[idx]
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        returned_data = {}

        # Load the depth image
        # if self.include_depth:
        depth_img = self.get_depth(idx, rot, zoom_factor)
        depth_norm = np.copy(depth_img)
        mask = depth_norm > 0
        min_d, max_d = np.min(depth_norm[mask]), np.max(depth_norm[mask])
        depth_norm[mask] = (depth_norm[mask] - min_d) / (max_d - min_d)

        # Load the RGB image
        # if self.include_rgb:
        rgb_img = self.get_rgb(idx, rot, zoom_factor)

        rgbd_img = np.concatenate((rgb_img, np.expand_dims(depth_norm, 0)), axis=0)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor, normalize=True)

        grip = Gripper([self.img_size/2, self.img_size/2], 0, self.gripper[0], self.gripper[1], self.gripper[2], (self.img_size, self.img_size))
        depth_state = np.clip((depth_img - depth_img.mean()) * 5 + 0.5, 0, 1)
        depth_state = np.uint8(depth_state * 255)
        grasp_base = np.zeros([self.img_size, self.img_size], dtype=np.uint8)
        # center, angle, width, rel, base, max_val
        contours_base = grip.get_contours(base=True)
        cv2.drawContours(grasp_base, contours_base, -1, (255), -1)
        depth_copy = depth_state.copy()
        grasp_base_not = cv2.bitwise_not(grasp_base)

        depth_mask = cv2.bitwise_or(depth_copy, grasp_base_not)
        highest = int(depth_mask.min())

        if highest == 0:
            highest = 1

        gripper_depth = int((depth_mask.max() - depth_mask.min()) * 0.2)

        state = np.where(highest + gripper_depth >= depth_state[:, :], 1, 0)

        scene_inputs = np.concatenate((np.expand_dims(depth_img, 0).astype(np.float32), rgbd_img), 0) 
        returned_data["scene_inputs"] = scene_inputs.astype(np.float32)

        if self.training:
            nw, na = len(self.arr_w), len(self.arr_a)
            pos_mask = np.zeros((nw, na), dtype=np.bool)

            gt_a, gt_w, gt_q = bbs[:, 0], bbs[:, 1], bbs[:, 2]
            mask_gt_a = (gt_a >= self.min_a) & (gt_a <= self.max_a)
            mask_gt_w = (gt_w >= self.min_w) & (gt_w <= self.max_w)
            mask_gt = mask_gt_a & mask_gt_w
            gt_a, gt_w, gt_q = gt_a[mask_gt], gt_w[mask_gt], gt_q[mask_gt]

            n_pos_samples = len(gt_a)
            if n_pos_samples > 0:
                gt_id_a = ((gt_a - self.min_a) / self.step_a).astype(np.uint8)
                gt_id_w = ((gt_w - self.min_w) / self.step_w).astype(np.uint8)
                _, unq_index = np.unique(np.stack((gt_id_a, gt_id_w), axis=1), return_index=True, axis=0)
                n_pos_samples = len(unq_index)

                gt_id_a, gt_id_w = gt_id_a[unq_index], gt_id_w[unq_index]
                pos_mask[gt_id_w, gt_id_a] = True

                pos_gripper_inputs = self.gripper_masks[pos_mask, :, :]
                pos_labels =  np.ones(pos_gripper_inputs.shape[0])
                pos_weights = gt_q[unq_index] / len(unq_index)

            n_neg_samples = self.n_train_samples - n_pos_samples
            sampled_neg_mask = masked_uniform_sampling(~pos_mask, num_samples=n_neg_samples)
            neg_gripper_inputs = self.gripper_masks[sampled_neg_mask, :, :]
            neg_labels = np.zeros(neg_gripper_inputs.shape[0])
            neg_weights = np.ones_like(neg_labels) / len(neg_labels)

            if len(gt_a) > 0:
                gripper_inputs = np.concatenate((pos_gripper_inputs, neg_gripper_inputs), 0)
                labels = np.concatenate((pos_labels, neg_labels), 0)
                weights = np.concatenate((pos_weights, neg_weights), 0)
            else:
                gripper_inputs, labels, weights = neg_gripper_inputs, neg_labels, neg_weights
            returned_data.update({"labels": labels.astype(np.float32),  # [N_samples,]
                                  "weights": weights.astype(np.float32)})  # [N_samples, ]
        else:
            gripper_inputs = self.gripper_masks  # [nw, na, self.img_size, self.img_size]

        # print(gripper_inputs.shape, labels.shape, weights.shape)
        returned_data.update({"gripper_inputs": gripper_inputs.astype(np.float32),
                              "gt_grasps": bbs.astype(np.float32),
                              "grid_w": self.grid_w.astype(np.float32),
                              "grid_a": self.grid_a.astype(np.float32),
                              "object_mask": np.stack((state, depth_img), axis=0).astype(np.float32)})

        return returned_data