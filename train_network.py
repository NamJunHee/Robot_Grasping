import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import get_network
from utils.utils import *
from utils.data import get_dataset
from utils.post_process import post_process_output2, post_process_output3, regress_grasp_pose
from utils.dataset_processing import evaluation




def validate(net, device, val_data, iou_threshold, epoch, gripper, img_size, valid_path):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    
    net.eval()
    
    results = {
        'correct': 0,
        'failed': 0,
        'None': 0,
        'loss': 0,
        'losses': {

        }
    }
    
    ld = len(val_data)
    print('ld', ld)
    
    T_label = []
    s = False

    with torch.no_grad():
        cnt = 0
        cnt_t = 0
        None_cnt = 0
        not_none = 0
        pred_none = 0

        batch_cnt = 0
        d = None
        rl_cnt = 0
        dnn_f_cnt = 0
        dnn_s_cnt = 0
        
        
        for batch in tqdm(val_data):
            gt_grasps, grid_w, grid_a = batch["gt_grasps"], batch["grid_w"], batch["grid_a"]
            gripper_data, scene_data = batch["gripper_inputs"], batch["scene_inputs"]   
            # x = batch["object_mask"]
            gt_grasps = gt_grasps.squeeze(0)
          
            xc = gripper_data.to(device), scene_data[:, 1:].to(device)
            output = regress_grasp_pose(net, xc, grid_w.to(device), grid_a.to(device))         
            output = post_process_output3(output)
            
            if gt_grasps[0][0] == 0:
                # print('None!')
                results['None'] += 1
                s = True

            else:            
                re = []
                re_check = []
                reward_list = []
                not_none += 1
                
                for rate in [0.1,0.2,0.3,0.4]:
                # for rate in [0.2]:
                    # action_cp = gt_grasps[0].cpu().detach().numpy().copy()
                    action_cp = output.copy()
                    scene_data_cp = scene_data.squeeze().cpu().detach().numpy().copy()
                    check, gp, rp, reward = evaluation.check_grasp_v2(scene_data_cp, img_size, action_cp, gripper, cnt, epoch, rate)
                    re.append([check, reward])
                    re_check.append(check)
                    reward_list.append(reward)


                if re_check.count(True) > 0:
                #     # print('re_check true')
                    dnn_s_cnt +=1
                    s = True 
                else:
                    # print('fail', action_cp ,re_check )
                    dnn_f_cnt += 1
            
            if s:
                cnt_t += 1
                T_label.append(cnt)
                results['correct'] += 1
            else:
                results['failed'] += 1

            # --- 시각화 추가 ---
            # check_grasp_v2의 결과를 시각화하고 저장
            # save_path = f"{valid_path}/epoch_{epoch}_batch_{batch_cnt}.png"
            # # print('save_path', save_path)
            # evaluation.visualize_grasp_result(
            #     scene_data_cp, action_cp, depth_state=np.random.rand(img_size, img_size),  # 예시 depth_state
            #     close_g=np.random.rand(img_size, img_size),  # 예시 close_g
            #     img_size=img_size,
            #     save_path=save_path
            # )
            # print(f"Batch {batch_cnt}의 시각화 결과가 {save_path}에 저장되었습니다.")
            # --- 시각화 끝 ---
            
            s = False
            cnt += 1
            batch_cnt += 1
        
        print('None', results['None']) # pred == 라벨 none 맞춘 숫자 
        print('none label', None_cnt) # 라벨된 none 숫자 
        print('pred_none', pred_none) # none으로 예측한 숫자  

        print('not_none', not_none)

    return results

    

def train(epoch, net, device, train_data, optimizer, scheduler):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }
    
    net.train()
    
    batch_idx = 0
    for batch_idx, batch in enumerate(tqdm(train_data, desc=f"Epoch {epoch}")):
        labels, weights = batch["labels"].to(device), batch["weights"].to(device)
        scene_inp, gripper_inp = batch["scene_inputs"][:, 1:].to(device), batch["gripper_inputs"].to(device)
        
        out_pred = net(scene_inp, gripper_inp)
        batch_size = out_pred.shape[0]
        out_pred, labels, weights = out_pred.flatten(), labels.flatten(), weights.flatten()

        # print('out_pred shape', out_pred.shape)
        # print('out_pred', out_pred)
        # print('labels shape', labels.shape)
        # print('labels', labels)


        loss = F.binary_cross_entropy_with_logits(out_pred, labels, weight=weights, reduction='sum') / batch_size

        # print('loss', loss)


        
        losses = {"out_loss": loss}  # Store loss
        
        if batch_idx % 500 == 0:
            logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
        
        results['loss'] += loss.item()
        for ln, l in losses.items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    print("lr: ", optimizer.param_groups[0]['lr'])
        
    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results
    

def run():
    args = parse_args()
    
    #==== get test_name ====#
    test_name = args.test_name
    
    #==== load test_name's configs ====#
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_location + test_name + "_" + current_time
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    weight_path = log_path + "/weights"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
        
    valid_path = log_path + "/valid"
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    
    
    with open(log_path + "/args.txt", "w") as f:
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write("{} : {}\n".format(key, args_dict[key]))
    
    configs = import_default_configs(test_name, log_path)
    configs['weight_path'] = weight_path
    with open(log_path + "/configs.txt", "w") as f:
        for key in configs.keys():
            f.write("{} : {}\n".format(key, configs[key]))
    print(configs)
    
    #==== load gripper configs ====#
    gripper_cfg = import_gripper_configs(configs['gripper_type'])
    gripper = (gripper_cfg['width_base'], gripper_cfg['gripper_rect'], gripper_cfg['base'])
    print(gripper_cfg)
    print(gripper)
    
    #==== set up tensorboard ====#
    tensorb_log_dir = os.path.join(log_path, "tensorboard")
    if not os.path.exists(tensorb_log_dir):
        os.makedirs(tensorb_log_dir)
    tb = tensorboardX.SummaryWriter(tensorb_log_dir)

    #==== Initialize logging ====#
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(log_path, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    
    #==== Get device  ====#
    device = get_device(configs['force_cpu'])
    print("device : ", device)
    
    
    #==== Load Dataset ====#
    logging.info('Loading {} Dataset...'.format(configs['dataset'].title()))
    Dataset = get_dataset(configs['dataset'])
    print(configs['dataset_path'])
    
    dataset = Dataset(configs['dataset_path'],
                      img_size=configs['input_size'],
                      ds_rotate=configs['ds_rotate'],
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=configs['use_depth'],
                      include_rgb=configs['use_rgb'],
                      binary_input=True,
                      gripper=gripper,
                      angle_step=configs['angle_step'],
                      width_step=configs['width_step'],
                      min_width=gripper_cfg['min_width'], max_width=gripper_cfg['max_width'], n_train_samples=configs['n_train_samples'])
    logging.info('Dataset size is {}'.format(dataset.length))
    
    
    #==== Creating data indices for training and validation splits ====#
    indices = list(range(len(dataset)))
    split = int(np.floor(configs['split'] * len(dataset)))
    if configs['ds_shuffle']:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))
    
    
    
    #==== Creating data samplers and loaders ====#
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers'],
        sampler=train_sampler
    )
    
    val_dataset = Dataset(configs['dataset_path'],
                        img_size=configs['input_size'],
                        ds_rotate=configs['ds_rotate'],
                        random_rotate=True,
                        random_zoom=True,
                        include_depth=configs['use_depth'],
                        include_rgb=configs['use_rgb'],
                        binary_input=True,
                        training=False,
                        gripper=gripper,
                        angle_step=configs['angle_step'],
                        width_step=configs['width_step'],
                        min_width=gripper_cfg['min_width'], max_width=gripper_cfg['max_width'])
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        sampler=val_sampler
    )
    logging.info('Done')
    
    
    #==== Load the network ====#
    logging.info('Loading Network...')
    # input_channels = 1 * args.use_depth + 3 * args.use_rgb
    input_channels = 5
    
    network = get_network(configs["Model"])
    model_cfg = import_model_configs(configs["Model"])
    
    net = network(
        input_channels=input_channels,
        # dropout=args.use_dropout,
        # prob=args.dropout_prob,
        # channel_size=args.channel_size
    )
    
    net = net.to(device)
    
    #==== Set up multi-gpus ====#
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))

    #==== Training Setup ====#
    lr_param = 2e-4
    if configs['optim'].lower() == 'adam':
        optimizer = optim.AdamW(net.parameters(), lr=lr_param, betas=(0.9, 0.999), weight_decay=0.01)
        # optimizer = optim.SGD(net.parameters(), lr=lr_param, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                    lr_lambda=lambda epoch: 0.5 if (epoch+1)%20 == 0 and (epoch < 60) else 1.0)
    
    best_iou = 0.0
    result_list = []
    best_filepath = None
    
    
    for epoch in range(configs['epochs']):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        
        train_results = train(epoch, net, device, train_data, optimizer, scheduler)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        print('train loss mean: ', train_results['loss'])
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
        
        if (epoch % configs['eval_freq'] == 0) or (epoch == configs['epochs'] - 1):
            logging.info('Validating...')
            test_results = validate(net, device, val_data, model_cfg['iou_threshold'], epoch, gripper, configs['input_size'], valid_path)
            logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                        test_results['correct'] / (test_results['correct'] + test_results['failed'])))

            # Log validation results to tensorbaord
            tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']),
                        epoch)
            # tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
            result_list.append(round(test_results['correct'] / (test_results['correct'] + test_results['failed']), 3))
            # print('val_loss: ', test_results['loss'])
            print(result_list)
            print('optim:', configs['optim'])
            print('lr_param', lr_param)
            for n, l in test_results['losses'].items():
                tb.add_scalar('val_loss/' + n, l, epoch)

            # Save best performing network
            iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
            ckpt_path = os.path.join(weight_path, 'model_epoch_%02d_iou_%0.3f.ckpt' % (epoch, iou))
            torch.save(net, ckpt_path)

            if iou > best_iou or epoch == 0:
                best_iou = iou

            print('best record:', best_iou)
            
            
            
            

if __name__ == "__main__":
    run()
    
    
    
