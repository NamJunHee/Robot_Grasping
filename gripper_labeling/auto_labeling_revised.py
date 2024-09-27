'''
Code to label by gripper
- Dataset : Jacquard
- Process : 
1. define gripper mask
    ex) gripper = (45, [[15, 0, 4, 6, 0], [-15, 0, 4, 6, 0]], [0, 0, 50, 40, 0]) 
        ( gripper control parameter, [[gripper pos1], [gripper pos2]..], depth parameter)


'''


import multiprocessing as mp
import os, random
from pathlib import Path
import numpy as np
import cv2
from gripper_model import Gripper
from PIL import Image
import random
from itertools import product, starmap

# from utils.data import get_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import torch
# import torch.optim as optim
# import torch.nn.functional as F

import glob
# import json
import time
import datetime
from random import randrange

# import tifffile
# from imageio import imread
import shutil
import yaml



# https://stackoverflow.com/a/25558333
def proc_init(l, c, o ,a, g, gi):
    global filesys_lock
    global crop_data_root
    global ori_data_root
    global action_list
    global gripper
    global gripper_info

    filesys_lock = l
    crop_data_root = c
    ori_data_root = o
    action_list = a
    gripper = g
    gripper_info = gi
    
        
def load_from_jacquard_file_base(fname, scale=1.0):
    """
    Load grasp rectangles from a Jacquard dataset file.
    :param fname: Path to file.
    :param scale: Scale to apply (e.g. if resizing images)
    :return: [((x,y), theta, w, h),..]
    """
    grs = []
    with open(fname) as f:
        for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                grs.append(((x, y), theta, w, h))
                # break
    return grs
    
def unique(lst, key):
    return list(
            {key(x):x for x in lst}.values()
    )

def subimage(image, center, theta, width, height):
    '''
    Function to Image crop
    :param image: target image
    :param center: center pos (x,y)
    :param theta: theta
    :param width: crop image width
    :param height: crop image height
    :return: croped image
    '''
    shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
    image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

    x = int( center[0] - width/2  )
    y = int( center[1] - height/2 )

    image = image[ y:y+height, x:x+width ]

    return image


def step(input_data, action, gripper, deep):
    '''
    Simulation Step
    :param input_data: [crop_img , crop_depth, depth_state]
    :param action: action of action_list
    :param gripper: gripper mask
    :param deep: parameter for depth
    :return: reward_sum , g_pass , r_pass , done , center_weight
        g_pass : bool, grasp success or noe
        r_pass : bool, No collision occurred : True
    '''
    
    min_w = 31
    reward = 0
    deep_rate = deep
    
    # Gripper
    grip = Gripper([gripper_info['img_size']/2, gripper_info['img_size']/2], 0, gripper[0], gripper[1], gripper[2], (gripper_info['img_size'], gripper_info['img_size']))
    
    # Action
    grip.rotate(action[2])
    grip.change_width(action[3])
    
    contours = grip.get_contours()
    
    depth_state = input_data[2]
    grasp_base = np.zeros([gripper_info['img_size'],gripper_info['img_size']], dtype=np.uint8)
    
    contours_base = grip.get_contours(base=True)
    cv2.drawContours(grasp_base, contours_base, -1, (255), -1)
    depth_copy = depth_state.copy()
    grasp_base_not = cv2.bitwise_not(grasp_base)
    depth_mask = cv2.bitwise_or(depth_copy, grasp_base_not)
    highest = int(depth_mask.min())

    if highest == 0:
        highest = 1
        
    grasp_crop = np.zeros([gripper_info['img_size'], gripper_info['img_size']], dtype=np.uint8)
    
    cv2.drawContours(grasp_crop, contours, -1, (highest), -1)
    #cv2.imwrite('./contour_image.png', grasp_crop)
    grasp_state_bin = np.zeros([gripper_info['img_size'], gripper_info['img_size']], dtype=np.uint8)
    cv2.drawContours(grasp_state_bin, contours, -1, 2, -1)
    
    
    d = int(action[3])/8
    
    line_points=[-2*d,-d,0,d,2*d]
    ref_line1 = []
    ref_line2 = []
    ref_center = []
    
    cos_th = np.cos(action[2]* np.pi/180)
    sin_th = np.sin(action[2]* np.pi/180)
    
    cx1 = (int(gripper_info['img_size']/2-10*cos_th),int(gripper_info['img_size']/2+ 10*sin_th))
    cx2 = (int(gripper_info['img_size']/2+10*cos_th),int(gripper_info['img_size']/2- 10*sin_th))
    
    gripper_depth = int((depth_mask.max()-depth_mask.min())*deep_rate)
    
    refine_state =  np.int8(highest + gripper_depth >= depth_state)
    
    for n in line_points:
        cos_d = np.cos((action[2]-90)* np.pi/180)
        sin_d = np.sin((action[2]-90)* np.pi/180)
        ref_cnt = (int(gripper_info['img_size']/2-n*cos_d),int(gripper_info['img_size']/2+ n*sin_d))
        ref_pnt1 = (int(cx1[0]-n*cos_d),int(cx1[1]+ n*sin_d))
        ref_pnt2 = (int(cx2[0]-n*cos_d),int(cx2[1]+ n*sin_d))

        ref_center.append(ref_cnt)
        ref_line1.append(ref_pnt1)
        ref_line2.append(ref_pnt2)
    
    check_1 = refine_state[np.transpose(ref_line1)[0],np.transpose(ref_line1)[1]]
    check_2 = refine_state[np.transpose(ref_line2)[0],np.transpose(ref_line2)[1]]
    check_c = refine_state[np.transpose(ref_center)[0],np.transpose(ref_center)[1]]
    
    
    a1= (check_1 == 1).sum()
    a2= (check_2 == 1).sum()
    ac= (check_c == 1).sum()

    check_through = [a1,ac,a2]
    
    observation = np.bitwise_or(grasp_state_bin.astype(int), refine_state.astype(int))

    open_g_any = bool(np.any(np.logical_and(grasp_crop > 0, highest + gripper_depth >= depth_state)))
    
    width = grip.base_width

    close_g = np.zeros([gripper_info['img_size'],gripper_info['img_size']], dtype=np.uint8)
    close_crop = np.zeros([gripper_info['img_size'],gripper_info['img_size']], dtype=np.uint8)
    close_state = np.zeros([gripper_info['img_size'],gripper_info['img_size']], dtype=np.uint8)

    while grip.base_width >= min_w - 25:
        grip.change_width_dw(-5)
        contours_c = grip.get_contours()
        cv2.drawContours(close_crop, contours_c, -1, highest+ gripper_depth, -1)


    close_state = np.clip(close_crop, 0, 255)

    close_g = np.int8((close_state >= depth_state) & (close_state != 0))

    grip.change_width(width)

    r,c = np.nonzero(close_g)
    
    # Map Center
    center_r = gripper_info['img_size']/2
    center_c = gripper_info['img_size']/2

    
    # 그리퍼 한 중심에 어느정도 물체에 해당하는 픽셀이 존재해야 함을 반영하기위한 중앙 감지 사각형 영역 
    center_weight =0 
    # 잡힘 상태 
    if np.size(r) != 0 :
        # 잡힘 
        center_r = r.mean()
        center_c = c.mean()
        reward += 10
        g_pass = True

    # 그리퍼랑 겹침 정도  close_g 값의 평균값이 존재 하지 않음 = 그리퍼 중앙엔 아무 물체에 해당하는 픽샐이 없을 경우 
    else:
        reward -= 10
        done = False
        g_pass = False

    
    grip_center = (int(center_r),int(center_c))

    # 2. 진입시  충돌 x 
    if not open_g_any:
        colide_reward = 30
        r_pass = True

    else:
        colide_reward = -30
        done = False
        r_pass = False
            
    # if a1 != 0 and a2 != 0 and ac != 0:   
    if check_through.count(0) == 0:
        reward += 10
        t_pass = True
    else:   
        reward -= 10
        t_pass = False  
        done = False

    grip_pixel = np.sum(close_g == 1)

    # 4. 잡는점 중심점 편심 정도 
    grip_centroid_dist = np.linalg.norm(np.array((gripper_info['img_size']/2,gripper_info['img_size']/2)) - grip_center)

    # 5. 그리퍼 중심 편심 정도
    para_weight = [1 , 0.01 , -0.5 , -0.5]

    reward_sum = reward + para_weight[0]*colide_reward + para_weight[1]*grip_pixel + para_weight[2]*\
                    grip_centroid_dist + action[3]*para_weight[3]


    # 잡힘 완료 기준 에 대한 여려 변형들 
    # if (outside_val < 0.0001) and (open_collision < 0.0001) and (close_max > 0.01):
    # if g_pass==True and r_pass == True and reward > 20 :# big size gripper
    if g_pass==True and r_pass == True and t_pass == True:  #small size gripper  
        done = True

        # rows = 1
        # cols = 1
        # fontsize = 20

        # fig_1 = plt.figure(figsize=(10, 8), dpi = 100)

        # ax1 = fig_1.add_subplot(rows,cols,1)
        # # ax1.imshow(input_data[1])
        # ax1.imshow(observation )
        # ax1.set_title(action[2],fontsize=fontsize)

        # plt.show()
        # plt.close(fig_1)

            
    return reward_sum , g_pass , r_pass , done ,center_weight


def calculate_min_square_side(w, h, theta_deg):

    # Convert theta from degrees to radians
    theta = np.deg2rad(theta_deg)
    
    # Calculate the width and height of the rotated rectangle's bounding box
    w_rotated = abs(w * np.cos(theta)) + abs(h * np.sin(theta))
    h_rotated = abs(w * np.sin(theta)) + abs(h * np.cos(theta))
    
    # The side of the minimum square that can cover the rotated rectangle
    return max(w_rotated, h_rotated)


def draw_grasp_boxes(image, grasp_boxes, save_path):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for ((x, y), theta, w, h) in grasp_boxes:
        # Convert theta from degrees to radians for calculation
        theta_rad = np.deg2rad(theta)

        # Calculate the box corners
        dx = w / 2.0
        dy = h / 2.0

        corners = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])

        # Rotate and translate corners
        corners = np.dot(corners, R)
        corners[:, 0] += x
        corners[:, 1] += y

        # Draw the box
        rect = patches.Polygon(corners, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')  # Hide axis

    # Save the image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) 


def save_original_images(idx, img, ori_data_path, depth_filename, grasps):
    ori_img_save = ori_data_path+'/'+str(idx)+'_RGB.png'
    ori_depth_save = ori_data_path+'/'+str(idx)+'_depth.tiff'        
    
    if (str(idx)+'_RGB') not in os.listdir(ori_data_path):
        cv2.imwrite(ori_img_save,img)
        shutil.copyfile(depth_filename,ori_depth_save)
        
    ori_img_with_boxes_save = ori_data_path+'/'+str(idx)+'_RGB_with_grasp_boxes.png'
    draw_grasp_boxes(img, grasps, ori_img_with_boxes_save)
        
        
def task(
        rgb_filename: str, 
        depth_filename: str,
        mask_filename : str,
        grasp: tuple, 
        idx: int, 
        num: int
) -> str:
    '''
    Task for each process.
    Returns status message.
    '''
    
    ## Read rgb, depth, mask
    img = cv2.imread(rgb_filename)
    depth = cv2.imread(depth_filename,-1)
    mask = cv2.imread(mask_filename)
    
    ## Masking 
    mask = np.where((mask[:,:,0] == 255),1 , 0)
    
    
    depth_img_max = depth.max()
    # apply mask to depth (make background : 0)
    depth = depth*mask
    depth = np.where( 0 == depth[:,:], depth_img_max, depth[:,:])
    
    zoom_factor = 1.0
    
    ## Crop image
    #crop_size = int(np.array([grasp[2],grasp[3]]).max()) *2
    # print("w, h, theta : ", grasp[2], grasp[3], grasp[1])
    crop_size = int(calculate_min_square_side(grasp[2], grasp[3], grasp[1])*1.5)
    # print("crop size : ", crop_size)
    crop_img = subimage(img, center=grasp[0], theta=0, width= crop_size, height=crop_size)
    crop_depth = subimage(depth, center=grasp[0], theta=0, width= crop_size, height=crop_size)
    
    # cropped image is outside the area -> pass
    if crop_depth.shape[0] != crop_size or crop_depth.shape[1] != crop_size:
        return f'edge point  at current_label {idx},{num}'
    
    
    crop_img = cv2.resize(crop_img,(gripper_info['img_size'],gripper_info['img_size']),interpolation=cv2.INTER_CUBIC)
    crop_depth = cv2.resize(crop_depth,(gripper_info['img_size'],gripper_info['img_size']),interpolation=cv2.INTER_CUBIC)
    
    depth_state = np.clip((crop_depth - crop_depth.mean()) * 5 + 0.5, 0, 1)
    depth_state = np.uint8(depth_state * 255)
    
    
    input_data = [crop_img , crop_depth, depth_state]
    action_dict = {}
    
    # deep change
    a_len = 0
    a = []
    for deep in [0.1,0.2,0.3,0.4]:
        for action_3dim in action_list:
            action_dict[action_3dim] = step(input_data,action_3dim,gripper, deep)
        action_dict_1 = {
            key: value
            for key, value
            in action_dict.items()
            if not (value[0] < 0 or value[3] == False) # value[0] : reward_sum, value[3] = done
        }
        a_sorted=sorted(action_dict_1.items(), key=lambda item: item[1], reverse=True)
        if len(a_sorted) > a_len:
            a_len = len(a_sorted)
            if len(a_sorted) < 10:
                a = a_sorted
            else:
                a = a_sorted[0:10]
              
    # Set data path
    crop_data_path = crop_data_root+'/'+str(idx)
    ori_data_path = ori_data_root+'/'+str(idx)
    
    # # Save ori image
    # ori_img_save = ori_data_path+'/'+str(idx)+'_RGB.png'
    # ori_depth_save = ori_data_path+'/'+str(idx)+'_depth.tiff'        
    
    # if (str(idx)+'_RGB') not in os.listdir(ori_data_path):
    #     cv2.imwrite(ori_img_save,img)
    #     shutil.copyfile(depth_filename,ori_depth_save)
        
    
    # ori_label_save = ori_data_path+'/'+str(idx)+'_label.txt'
    # with open(ori_label_save, 'a') as f:
    #     if not a:
    #         pass
    #     else:
    #         for row in a:
    #             #(x,y,theta,w,h, 224_width,q)
    #             f.write(str(grasp[0][0])+';'+str(grasp[0][1])+';'+str(row[0][2])+';'+str(grasp[2])+';'+str(grasp[3])+';'+str(row[0][3])+';'+str(row[1][0])+'\n')


    # Save crop image
    img_save = crop_data_path+'/'+str(num)+'_crop'+'_RGB.png'
    img_frame = input_data[0]
    cv2.imwrite(img_save, img_frame)

    # float16 변환 버전 
    # depth_result = input_data[1].astype(np.uint8)

    # uint8 변환 버번 

    min_val = input_data[1].min()
    max_val = input_data[1].max()

    depth_result_normalized = (input_data[1] - min_val)/ (max_val - min_val) 

    depth_result = (depth_result_normalized *255).astype(np.uint8)


    depth_save = crop_data_path+'/'+str(num)+'_crop'+'_depth.npy'
    np.save(depth_save,depth_result)
    label_save = crop_data_path+'/'+str(num)+'_crop'+'_label.txt'
    with open(label_save, 'w') as f:
        if not a:
            f.write('None')
            result = f'None! at current_label {idx},{num}'
        else:
            for row in a:
                f.write(str(grasp[0][0])+';'+str(grasp[0][1])+';'+str(row[0][2])+';'+str(row[0][3])+';'+str(row[1][0])+'\n')
            result = f'len a!! {len(a):>2}  at current_label {idx},{num}'

    return result

def main():
    
    ##== define gripper info
    with open('../configs/gripper_info.yml', 'r') as file:
        gripper_data = yaml.safe_load(file)
    
    gripper_info = gripper_data['3f_224']
    gripper = (gripper_info['width_base'], 
            gripper_info['gripper_rect'], 
            gripper_info['base'])
    
    w_resolution = 5
    theta_resolution = 5
    
    
    ##== Make action_list (action candidate)
    x_action = [0]
    y_action = [0]
    w_action = list(range(31,171,w_resolution))
    theta_action = list(range(0,360,theta_resolution))

    action_list=list(product(x_action,y_action,theta_action, w_action))
    action_list.append((0, 0, 0, 0))

    print('total action_list:', len(action_list))
    
    
    ##== Read labeled index
    labeled_idx = 0
    labeled_idx_path = f"./labeled_idx/labeled_idx_{gripper_info['name']}_w-{w_resolution}_theta-{theta_resolution}.txt"
    if os.path.exists(labeled_idx_path):
        # if already file exist -> read txt file
        with open(labeled_idx_path, 'r', encoding='utf-8') as file:
            content = file.read()
            try:
                labeled_idx = int(content.strip())
                print(f"labeled index: {labeled_idx}")
            except ValueError:
                print("error exist")
    else:
        # else -> create txt file 
        with open(labeled_idx_path, 'w', encoding='utf-8') as file:
            init_idx = 0  
            file.write(str(init_idx))
        print(f"labeled_idx_{gripper_info['name']}_w-{w_resolution}_theta-{theta_resolution}.txt file creation complete")
        
        
    ##== Read Target Dataset
    dataset_path = '/home/nmail/workplace/gripper_dataset/Jacquard_dataset/sum'
    
    grasp_files = glob.glob(os.path.join(dataset_path, '*', '*_grasps.txt'))
    grasp_files.sort()
    
    rgb_files = [f.replace('grasps.txt', 'RGB.png') for f in grasp_files]
    depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in grasp_files]
    mask_files = [f.replace('grasps.txt', 'mask.png') for f in grasp_files]
    print('length of grasp_files:',len(grasp_files))
    
    
    ##== Set Labeled Data path
    ori_data_root = f"/home/nmail/workplace/gripper_dataset/dataset_{gripper_info['name']}_w-{w_resolution}_theta-{theta_resolution}/ori_dataset"
    crop_data_root = f"/home/nmail/workplace/gripper_dataset/dataset_{gripper_info['name']}_w-{w_resolution}_theta-{theta_resolution}/crop_dataset"
    
    # Check if ori_data_root exists, if not, create it
    if not os.path.exists(ori_data_root):
        os.makedirs(ori_data_root)
        print(f'Directory {ori_data_root} created.')

    # Check if crop_data_root exists, if not, create it
    if not os.path.exists(crop_data_root):
        os.makedirs(crop_data_root)
        print(f'Directory {crop_data_root} created.')
    
    ##== Set workers
    
    workers = len(os.sched_getaffinity(0)) // 2 - 1
    print('-'*40)
    print(f"starting with {workers} workers...")
    
    l = mp.Lock()
    ##== Labeling Loop
    with mp.Pool(workers, initializer=proc_init, initargs=(l, crop_data_root, ori_data_root, action_list, gripper, gripper_info)) as pool:
        for idx in range(labeled_idx, len(grasp_files)):
            print(f'----- Current time: {datetime.datetime.now()} -----', flush=True)
            print(f'index [{idx}]', flush=True)
            
            if str(idx) not in os.listdir(crop_data_root):
                os.mkdir(crop_data_root+'/'+str(idx))

            if str(idx) not in os.listdir(ori_data_root):
                os.mkdir(ori_data_root+'/'+str(idx))
                                    
            
            ## grasps bbox of idx_th object
            grasps = load_from_jacquard_file_base(grasp_files[idx])
            # remove overlapping rows ( unique (x,y) pos )
            grasps = unique(grasps, lambda x: tuple(x[0]))
            print("len(grasps) : ", len(grasps))
            
            img = cv2.imread(rgb_files[idx])
            save_original_images(idx, img, ori_data_root+'/'+str(idx), depth_files[idx], grasps)
            
            
            task_arguments = [
                (rgb_files[idx], depth_files[idx], mask_files[idx] ,grasp, idx, num)
                for num, grasp
                in enumerate(grasps)
            ]
            
            gathered_result = pool.starmap(task, task_arguments, 4)
            
            print('\n'.join(gathered_result), flush=True)
            
            # Rename files
            dir = str(idx)
            
            files = sorted(set([ int(p.name.split('_')[0]) for p in (Path(crop_data_root) / dir).glob('*')]))
            for new_x, x in enumerate(files):
                for thing in ('crop_label.txt', 'crop_RGB.png', 'crop_depth.npy'):
                    (Path(crop_data_root) / dir / f'{x}_{thing}').rename(Path(crop_data_root) / dir / f'{new_x}_{thing}')

            with open(labeled_idx_path, 'w', encoding='utf-8') as file:
                file.write(str(idx + 1))


    
    
    
if __name__ == '__main__':
    main()