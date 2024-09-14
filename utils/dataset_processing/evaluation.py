import warnings
import os
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from .grasp import GraspRectangles, detect_grasps, detect_grasps_parallel, Grasp

from .gripper_model import Gripper
import matplotlib
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

def normalize(val, max_v, min_v):
    return (val - min_v) / (max_v - min_v)

def round_to_m5(x, multiple=5):
    rounded = round(x)

    if rounded % multiple !=0:
        if (rounded+ multiple/2) % multiple ==0:
            rounded += multiple/2
        else:
            rounded -= multiple/2

    return rounded

def round_to_odd(x, even=False):
    rounded = round(x)

    if even:
        if rounded % 2 !=0:
            rounded += 1
    else:
        if rounded % 2 ==0:
            rounded += 1

    return rounded


def plot_output(fig, rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img:
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.pause(0.1)
    fig.canvas.draw()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):

        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False

def parallel_match(grasp_angle, img_size, grasp_width, ground_truth_bbs, no_grasps=1, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """

    grasp_angle = grasp_angle*180
    # print('a pred/ a gt', grasp_angle , ground_truth_bbs[0])
    # print('w pred/ w gt', grasp_width, ground_truth_bbs[1])
    # print('ground_truth_bbs', ground_truth_bbs)
    # print('gt angle', np.deg2rad(ground_truth_bbs[0]))
    grasp_point = [img_size//2,img_size//2]

    gt_bbs = []
    gt_bb = Grasp(grasp_point, ground_truth_bbs[0])
    gt_bb.length = ground_truth_bbs[1]
    gt_bb.width = gt_bb.length / 2

    gt_bbs.append(gt_bb.as_gr)
    
    g = detect_grasps_parallel(grasp_angle, grasp_width, no_grasps=no_grasps)

    if g.max_iou(gt_bbs) > threshold:
        # print('pass')
        return True
    else:
        # print('fail')
        return False


def calculate_val_match(theta_pred, width_pred, q_pred, ground_truth_bbs, grasp_width=None, threshold=0.1):
    
    # print('ground_truth_bbs',ground_truth_bbs[0])

    # if not isinstance(ground_truth_bbs, GraspRectangles):
    #     print('not isinstance')
    #     # gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    # else:

    gt_bbs = ground_truth_bbs[0]
    # print('gt_bbs',gt_bbs)

    # print('theta_pred',type(theta_pred), theta_pred)
    # print('width_pred',type(width_pred), width_pred)
    # print('q_pred',type(q_pred), q_pred)

    # print('theta_gt',type(gt_bbs[0]), gt_bbs[0])
    # print('width_gt',type(gt_bbs[1]), gt_bbs[1])
    # print('q_gt',type(gt_bbs[2]), gt_bbs[2])    

    norm_theta = normalize(gt_bbs[0],360,-1)
    norm_width = normalize(gt_bbs[1],60,10)
    norm_q = normalize(gt_bbs[2],20,50)

    # print(22, norm_theta)

    d_th = abs(norm_theta - theta_pred)  
    d_wi = abs(norm_width - width_pred) 
    d_q = abs(norm_q - q_pred)  

    # print('theta_gt',theta_pred, norm_theta)
    # print('width_gt',width_pred, norm_width)
    # print('q_gt',q_pred, norm_q)   

    # # print(d_th/gt_bbs[0],d_wi/gt_bbs[1],d_q/gt_bbs[2])
    # print(abs(d_th/norm_theta),abs(d_wi/norm_width), abs(d_q/norm_q))
    # print(abs(d_th/norm_theta)+abs(d_wi/norm_width)+ abs(d_q/norm_q))

    # if abs(d_th/norm_theta)+abs(d_wi/norm_width)+ abs(d_q/norm_q) < threshold:
    
    # if abs(d_th)+abs(d_wi)+ abs(d_q) < threshold:
    if abs(d_th)+abs(d_wi) < threshold:
    
        return True
    else:
        return False

    
    # gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    # for g in gs:
    #     if g.max_iou(gt_bbs) > threshold:
            # return True
    # else:
    #     return False


def check_grasp_d(data_input, img_size, action, gripper,cnt,epoch):

    gripper_length= 0.05
    ref_contours = []
    grasp_contours = []
    grs = []
    min_w = 16
    reward = 0
    done = False


    # center, angle, width, rel, base, max_val
    #self.gripper_obj = Gripper([self.max_x // 2, self.max_y // 2], 0, self.gripper[0], self.gripper[1], self.gripper[2], (self.max_x, self.max_y))
    grip = Gripper([img_size//2,img_size//2], 0, gripper[0], gripper[1], gripper[2], (img_size,img_size))

    # 그림만 넣는경우 
    data_input = data_input.squeeze().cpu().detach().numpy()

    # print(33, data_input.shape)
    
    # 여러 그림 3채널로 넣는경우 
    data_input = np.transpose(data_input,(1,2,0))
    # print(44, data_input.shape)
    data_cp = data_input.copy()
    data_input = data_input[:,:,0]
    #print('datainput 3',data_cp[:,:,3])
    #print('datainput 4',data_cp[:,:,4])

    # depth_input = np.squeeze(data_input[:,:,0], axis=2)
    # print(55, depth_input.shape)
    depth_state = np.clip((data_input - data_input.mean()) * 5 + 0.5, 0, 1)
    depth_state = np.uint8(depth_state * 255)


    action_cp=[]
    # print('action1', action)
    action_cp = [action[0], action[1]]

    action[0] = action[0]*361 - 1 #np.clip((float(action[0])*361) - 1, -1, 360)
    action[1] = action[1]*39 + 8 #np.clip((float(action[1])*39) + 8, 8, 47)

    #action[0] = round_to_m5(np.clip( (float(action[0])*361)-1,0,360),multiple=5)
    #action[1] = round_to_odd(np.clip( (float(action[1])*39) +8, 9,47),even=False)
    
    # print('action2', action)

    
    #if action[0] == 0 and action[1]==9:
     #   print(77, action_cp[0],action_cp[1])


    #action
    grip.rotate(action[0])
    grip.change_width(action[1])


    # need to generate depth image with the grasp position
    grasp_base = np.zeros([img_size,img_size], dtype=np.uint8)
    # center, angle, width, rel, base, max_val
    contours_base = grip.get_contours(base=True)
    cv2.drawContours(grasp_base, contours_base, -1, (255), -1)
    depth_copy = depth_state.copy()
    grasp_base_not = cv2.bitwise_not(grasp_base)
    #print('de cp', depth_copy.shape,type(depth_copy))
    #print(11, depth_copy)
    #print('gbn', grasp_base_not.shape,type(grasp_base_not))
    #print(22, grasp_base_not)
    depth_mask = cv2.bitwise_or(depth_copy, grasp_base_not)
    highest = int(depth_mask.min())

    if highest == 0:
        highest = 1


    contours = grip.get_contours()
   

    grasp_state_bin = np.zeros([img_size, img_size], dtype=np.uint8)
    
    cv2.drawContours(grasp_state_bin, contours, -1, (highest), -1)

    d = int(action[1])/8
    
    line_points=[-2*d,-d,0,d,2*d]
    ref_line1 = []
    ref_line2 = []
    ref_center = []
    # test finding center line

    cos_th = np.cos(action[0]* np.pi/180)
    sin_th = np.sin(action[0]* np.pi/180)


    cx1 = (int((img_size//2)-10*cos_th),int((img_size//2)+ 10*sin_th))
    cx2 = (int((img_size//2)+10*cos_th),int((img_size//2)- 10*sin_th))

    gripper_depth = int((depth_mask.max()-depth_mask.min())*0.2)
    
    refine_state = np.int8(highest + gripper_depth >= depth_state)
    
    for n in line_points:
        cos_d = np.cos((action[0]-90)* np.pi/180)
        sin_d = np.sin((action[0]-90)* np.pi/180)
        ref_cnt = (int((img_size//2)-n*cos_d),int((img_size//2)+ n*sin_d))
        ref_pnt1 = (int(cx1[0]-n*cos_d),int(cx1[1]+ n*sin_d))
        ref_pnt2 = (int(cx2[0]-n*cos_d),int(cx2[1]+ n*sin_d))
    
        ref_center.append(ref_cnt)
        ref_line1.append(ref_pnt1)
        ref_line2.append(ref_pnt2)

    check_1 = refine_state[np.transpose(ref_line1)[0],np.transpose(ref_line1)[1]]
    check_2 = refine_state[np.transpose(ref_line2)[0],np.transpose(ref_line2)[1]]
    check_c = refine_state[np.transpose(ref_center)[0],np.transpose(ref_center)[1]]

    a1= (check_1 ==1).sum()
    a2= (check_2 ==1).sum()
    ac= (check_c ==1).sum()

    check_through = [a1,ac,a2]

    # observation = np.bitwise_or(grasp_state_bin.astype(int), refine_state.astype(int))


    # observation = np.bitwise_or(grasp_state_bin, refine_state)


    # open_g = np.where((observation[:,:] == 3), 1, 0)
    # open_g = np.where((grasp_state[:,:] >= depth_state[:,:]) & (grasp_state[:,:] != 0), 1, 0)
    # open_collision = np.mean(open_g)
    
    open_g_any = bool(np.any(np.logical_and(grasp_state_bin > 0, highest + gripper_depth >= depth_state)))


    width = grip.base_width
    
   
    close_g = np.zeros([img_size,img_size], dtype=np.uint8)
    close_crop = np.zeros([img_size,img_size], dtype=np.uint8)
    close_state = np.zeros([img_size,img_size], dtype=np.uint8)

    while grip.base_width >= min_w - 10:

        grip.change_width_dw(-2)
        # close_crop = np.zeros([img_size,img_size], dtype=np.uint8)
        contours_c = grip.get_contours()
        cv2.drawContours(close_crop, contours_c, -1, highest+ gripper_depth, -1)

    close_state = np.clip(close_crop, 0, 255)
    
    close_g = np.int8((close_state >= depth_state) & (close_state != 0))

    grip.change_width(width)

    r,c = np.nonzero(close_g)

    #초기 값 맵 중심 
    center_r = (img_size//2)
    center_c = (img_size//2)
    
    grip_center = (int(center_c),int(center_r))

    center_check_range = 5
    center_weight = np.sum(close_g[(img_size//2)-center_check_range:(img_size//2)+center_check_range,(img_size//2)-center_check_range:(img_size//2)+center_check_range] == 1)
    
    # grip
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
        # print('r pass true')
        r_pass = True
    else:
        colide_reward = -30
        done = False 
        r_pass = False
        # return next_state, reward, True

    # 모두 관통 여부 
    #if check_through.count(0) <= 1 :
    if check_through.count(0) == 0 :
        reward += 10
        t_pass = True
    else:   
        reward -= 10
        t_pass = False  
        done = False



    grip_pixel = np.sum(close_g == 1)

    # 3. 잡는점 중심점 편심 정도 
    grip_centroid_dist = np.linalg.norm(np.array(((img_size//2),(img_size//2))) - grip_center)

    # 4. 그리퍼 중심 편심 정도
    # movement_dist = np.linalg.norm(np.array((32,32)) - (self.d_x, self.d_y))

    para_weight = [1 , 0.1 , -0.5 , -0.3]

    # reward = para_weight[0]*colide_reward + para_weight[1]*grip_pixel + para_weight[2]* grip_centroid_dist

    reward = para_weight[0]*colide_reward + para_weight[1]*grip_pixel + para_weight[2]*\
            grip_centroid_dist + action[1]*para_weight[3]

    # if (outside_val < 0.0001) and (open_collision < 0.0001) and (close_max > 0.01):
    # if g_pass==True and r_pass == True and reward > 20 :# big size gripper
    if g_pass==True and r_pass == True and t_pass == True:  #small size gripper
        
        done = True


    return  done, g_pass, r_pass, reward 



def visualize_grasp_result(data_input, action, depth_state, close_g, img_size, save_path="grasp_result.png"):
    """
    주어진 데이터를 바탕으로 그리퍼의 상태를 시각화하고 이미지를 저장합니다.
    
    :param data_input: 입력 이미지 데이터 (예: 씬 데이터)
    :param action: 그리퍼의 회전 각도 및 너비 정보 [각도, 너비]
    :param depth_state: 깊이 상태 정보 (이미지 형태)
    :param close_g: 그리퍼가 닫히는 상태 정보 (바이너리 마스크 형태)
    :param img_size: 입력 이미지 크기 (예: 224, 320 등)
    :param save_path: 저장할 이미지 경로
    """
    # 입력 데이터를 시각화 가능한 형태로 변환 (예: 첫 번째 채널 사용)
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = np.uint8(data_input[:, :, 0] * 255)  # 첫 번째 채널을 사용하여 시각화

    # Matplotlib을 이용해 그리퍼 상태 그리기
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)

    # 그리퍼의 중심 및 그립 상태 그리기
    center = (img_size // 2, img_size // 2)
    plt.scatter(center[0], center[1], s=100, c='red', label='Gripper Center')

    # 그리퍼의 각도 및 너비를 기반으로 선 그리기
    angle = action[0] * np.pi / 180  # 각도를 라디안으로 변환
    width = action[1]

    # 그리퍼의 선 그리기
    x1 = center[0] + int(width * np.cos(angle))
    y1 = center[1] + int(width * np.sin(angle))
    x2 = center[0] - int(width * np.cos(angle))
    y2 = center[1] - int(width * np.sin(angle))
    
    plt.plot([x1, x2], [y1, y2], 'b-', lw=3, label='Gripper')

    # 그리퍼가 닫히는 상태 시각화 (바이너리 마스크)
    close_mask = close_g.astype(np.uint8) * 255
    plt.imshow(close_mask, cmap='Blues', alpha=0.5, label='Close Gripper Mask')

    # 깊이 정보 시각화
    depth_state_visual = np.uint8(depth_state * 255)
    plt.imshow(depth_state_visual, cmap='Reds', alpha=0.3, label='Depth State')

    # 그리퍼 각도 및 너비 정보 출력
    plt.title(f"Gripper Angle: {action[0]}°, Width: {action[1]}")

    plt.legend()
    plt.axis('off')

    # 이미지를 저장
    plt.savefig(save_path)
    plt.close()
    print(f"결과 이미지가 {save_path}에 저장되었습니다.")
    
    
def check_grasp_v2(data_input, img_size, action_input, gripper,cnt,epoch, filter_ratio=0.2):

    # different filter ratio

    gripper_length= 0.05
    ref_contours = []
    grasp_contours = []
    grs = []
    min_w = 16
    reward = 0
    done = False


    grip = Gripper([img_size//2,img_size//2], 0, gripper[0], gripper[1], gripper[2], (img_size,img_size))
    
    # action =[0,45]
    # grip.rotate(action[0])
    # grip.change_width(action[1])

    # center, angle, width, rel, base, max_val
    #self.gripper_obj = Gripper([self.max_x // 2, self.max_y // 2], 0, self.gripper[0], self.gripper[1], self.gripper[2], (self.max_x, self.max_y))
    

    action = action_input
    # print('action 1 ', action)
    # action[0] = int((action[0]*361) - 1)
    # action[1] = int((action[1]*39) + 8)

    
    # action[0] = round_to_m5(np.clip( (float(action[0])*361)-1,0,360),multiple=5)
    # action[1] = round_to_odd(np.clip( (float(action[1])*39) +8, 9,47),even=False)
    action[0] = round_to_m5(((float(action[0])*361)-1),multiple=5)
    action[1] = round_to_odd(((float(action[1])*39) +8),even=False)
    # print('action after', action)
    #action
    grip.rotate(action[0])
    grip.change_width(action[1])

    contours = grip.get_contours()


    # 그림만 넣는경우 
    data_input = data_input #.squeeze().cpu().detach().numpy()


    # print(33, data_input.shape)
    
    # 여러 그림 3채널로 넣는경우 
    data_input = np.transpose(data_input,(1,2,0))
    # print(44, data_input.shape)
    data_cp = data_input.copy()
    depth_img = data_input[:,:,4]

    #print('datainput 3',data_cp[:,:,3])
    #print('datainput 4',data_cp[:,:,4])




    # depth_input = np.squeeze(data_input[:,:,0], axis=2)
    depth_state = np.clip((depth_img - depth_img.mean()) * 5 + 0.5, 0, 1)
    depth_state = np.uint8(depth_state * 255)
    # print('depth_state', depth_state.shape)

    # need to generate depth image with the grasp position
    grasp_base = np.zeros([img_size,img_size], dtype=np.uint8)
    # center, angle, width, rel, base, max_val
    contours_base = grip.get_contours(base=True)
    cv2.drawContours(grasp_base, contours_base, -1, (255), -1)
    depth_copy = depth_state.copy()
    grasp_base_not = cv2.bitwise_not(grasp_base)
    # print(depth_copy.shape, grasp_base_not.shape)
    # print(type(depth_copy), type(grasp_base_not))
    depth_mask = cv2.bitwise_or(depth_copy, grasp_base_not)
    highest = int(depth_mask.min())

    if highest == 0:
        highest = 1


    # refine_state = np.where(highest + gripper_depth >= depth_state[:, :], 1, 0)
    
    # action_cp=[]
    # print('action1', action_input)
    # print('action 1_1', action)
    # action_cp = [action[0], action[1]]


   

    grasp_state_bin = np.zeros([img_size, img_size], dtype=np.uint8)
    
    cv2.drawContours(grasp_state_bin, contours, -1, (highest), -1)

    d = int(action[1])/8
    
    line_points=[-2*d,-d,0,d,2*d]
    ref_line1 = []
    ref_line2 = []
    ref_center = []
    # test finding center line

    cos_th = np.cos(action[0]* np.pi/180)
    sin_th = np.sin(action[0]* np.pi/180)


    cx1 = (int((img_size//2)-10*cos_th),int((img_size//2)+ 10*sin_th))
    cx2 = (int((img_size//2)+10*cos_th),int((img_size//2)- 10*sin_th))

    
    gripper_depth = int((depth_mask.max() - depth_mask.min()) * filter_ratio)
    refine_state = np.int8(highest + gripper_depth >= depth_state)
    
    for n in line_points:
        cos_d = np.cos((action[0]-90)* np.pi/180)
        sin_d = np.sin((action[0]-90)* np.pi/180)
        ref_cnt = (int((img_size//2)-n*cos_d),int((img_size//2)+ n*sin_d))
        ref_pnt1 = (int(cx1[0]-n*cos_d),int(cx1[1]+ n*sin_d))
        ref_pnt2 = (int(cx2[0]-n*cos_d),int(cx2[1]+ n*sin_d))
    
        ref_center.append(ref_cnt)
        ref_line1.append(ref_pnt1)
        ref_line2.append(ref_pnt2)

    check_1 = refine_state[np.transpose(ref_line1)[0],np.transpose(ref_line1)[1]]
    check_2 = refine_state[np.transpose(ref_line2)[0],np.transpose(ref_line2)[1]]
    check_c = refine_state[np.transpose(ref_center)[0],np.transpose(ref_center)[1]]

    a1= (check_1 ==1).sum()
    a2= (check_2 ==1).sum()
    ac= (check_c ==1).sum()

    check_through = [a1,ac,a2]

    # observation = np.bitwise_or(grasp_state_bin.astype(int), refine_state.astype(int))


    # observation = np.bitwise_or(grasp_state_bin, refine_state)


    # open_g = np.where((observation[:,:] == 3), 1, 0)
    # open_g = np.where((grasp_state[:,:] >= depth_state[:,:]) & (grasp_state[:,:] != 0), 1, 0)
    # open_collision = np.mean(open_g)

    open_g_any = bool(np.any(np.logical_and(grasp_state_bin > 0, highest + gripper_depth >= depth_state)))


    width = grip.base_width
    
   
    close_g = np.zeros([img_size,img_size], dtype=np.uint8)
    close_crop = np.zeros([img_size,img_size], dtype=np.uint8)
    close_state = np.zeros([img_size,img_size], dtype=np.uint8)

    while grip.base_width >= min_w - 10:

        grip.change_width_dw(-2)
        # close_crop = np.zeros([img_size,img_size], dtype=np.uint8)
        contours_c = grip.get_contours()
        cv2.drawContours(close_crop, contours_c, -1, highest+ gripper_depth, -1)

    close_state = np.clip(close_crop, 0, 255)
    
    close_g = np.int8((close_state >= depth_state) & (close_state != 0))

    grip.change_width(width)

    r,c = np.nonzero(close_g)

    #초기 값 맵 중심 
    center_r = img_size//2
    center_c = img_size//2
    
    grip_center = (int(center_c),int(center_r))

    center_check_range = 5
    center_weight = np.sum(close_g[(img_size//2)-center_check_range:(img_size//2)+center_check_range,(img_size//2)-center_check_range:(img_size//2)+center_check_range] == 1)
    
    # grip
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
        # print('r pass true')
        r_pass = True
    else:
        colide_reward = -30
        done = False 
        r_pass = False
        # return next_state, reward, True

    # 모두 관통 여부 
    #if check_through.count(0) <= 1 :
    if check_through.count(0) == 0 :
        reward += 10
        t_pass = True
    else:   
        reward -= 10
        t_pass = False  
        done = False



    grip_pixel = np.sum(close_g == 1)

    # 3. 잡는점 중심점 편심 정도 
    grip_centroid_dist = np.linalg.norm(np.array(((img_size//2),(img_size//2))) - grip_center)

    # 4. 그리퍼 중심 편심 정도
    # movement_dist = np.linalg.norm(np.array((32,32)) - (self.d_x, self.d_y))

    para_weight = [1 , 0.1 , -0.5 , -0.5]

    # reward = para_weight[0]*colide_reward + para_weight[1]*grip_pixel + para_weight[2]* grip_centroid_dist

    reward = para_weight[0]*colide_reward + para_weight[1]*grip_pixel + para_weight[2]*\
            grip_centroid_dist + action[1]*para_weight[3]

    # if (outside_val < 0.0001) and (open_collision < 0.0001) and (close_max > 0.01):
    # if g_pass==True and r_pass == True and reward > 20 :# big size gripper
    if g_pass==True and r_pass == True and t_pass == True:  #small size gripper
        
        done = True


    return  done, g_pass, r_pass, reward 

