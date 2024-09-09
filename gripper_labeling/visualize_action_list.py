import os
import cv2
import numpy as np
import yaml
from itertools import product
from gripper_model import Gripper


def save_gripper_masks(action_list, gripper, output_dir, img_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gripper 초기 위치와 이미지 크기 설정
    center = [img_size/2, img_size/2]  # 이미지 중심 좌표
    max_val = (img_size, img_size)  # 이미지 크기

    for idx, action in enumerate(action_list):
        grip = Gripper(center, 0, gripper[0], gripper[1], gripper[2], max_val)
        grip.rotate(action[2])
        grip.change_width(action[3])
        contours = grip.get_contours()

        # 빈 이미지 생성
        gripper_mask_img = np.zeros(max_val, dtype=np.uint8)

        # 그리퍼 마스크를 이미지에 그리기
        cv2.drawContours(gripper_mask_img, contours, -1, 255, -1)

        # 파일명 설정 및 저장
        mask_filename = os.path.join(output_dir, f"gripper_mask_{idx}_theta{action[2]}_width{action[3]}.png")
        cv2.imwrite(mask_filename, gripper_mask_img)
        print(f"Saved {mask_filename}")


def main():
    
    
    # Gripper 정보 설정
    with open('../configs/gripper_info.yml', 'r') as file:
        gripper_data = yaml.safe_load(file)
    
    gripper_info = gripper_data['2f_64']
    gripper = (gripper_info['width_base'], 
            gripper_info['gripper_rect'], 
            gripper_info['base'])
    w_resolution = 2
    theta_resolution = 1

    
    # Action list 생성
    x_action = [0]
    y_action = [0]
    w_action = list(range(9, 47, w_resolution))
    theta_action = list(range(0, 360, theta_resolution))

    action_list = list(product(x_action, y_action, theta_action, w_action))

    # 저장할 디렉토리 설정
    output_dir = f"./gripper_masks/{gripper_info['name']}_w-{w_resolution}_theta-{theta_resolution}"
    
    # Check if ori_data_root exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Directory {output_dir} created.')

    # Gripper 마스크 생성 및 저장
    save_gripper_masks(action_list, gripper, output_dir, gripper_info['img_size'])

if __name__ == "__main__":
    main()