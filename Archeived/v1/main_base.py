import argparse

import gymnasium as gym
from environment import DTN
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import os, shutil, copy, time

import torch
import numpy as np
import random

from Policy import PPO, device 

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','fal   se','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument('--device_num', type=int, default=10, help='Number of devices in the environment')
parser.add_argument('--max_channel_num', type=int, default=10, help='Max Number of channels in the environment')
parser.add_argument('--max_step', type=int, default=100, help='Max slots of each processing period of MEC')

parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--model_save', type=str2bool, default=False, help='Save model or Not')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--T_horizon', type=int, default=1000, help='lenth of long trajectory')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=200000, help='which model to load')
parser.add_argument('--Max_train_steps', type=int, default=200000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=50000, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1000, help='Model evaluating interval, in steps.')
parser.add_argument('--eval_period', type=int, default=1000, help='Model evaluating period, in steps.')
parser.add_argument('--eval_turns', type=int, default=3, help='Number of evaluation turns')
parser.add_argument('--env_reset_interval', type=int, default=1000, help='Environment reset interval')

parser.add_argument('--beta', type=float, default=0.2, help='Variance Factor')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--v_lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')


opt = parser.parse_args()
print(opt)


def evaluate_policy(env, model, render):
    with torch.no_grad():
        scores_service = 0.
        scores_edge = 0.

        turns = opt.eval_turns
        for _ in range(turns):
            s = env.reset()
            ep_r_edge = 0.
            ep_r_service = 0.
            step = 0
            total_t = opt.eval_period
            while step < total_t:
                a_s = np.random.randint(0, env.action_s_num-1)
                a_p = np.random.uniform(0, 1, size=env.device_num)
                s_prime, _, _, _, info = env.step(a_s, a_p)
                ep_r_edge += info['edge_aoi']  # Collect the edge AOI as the reward
                ep_r_service += info['service_aoi']  # Collect the service AOI as the reward
                #print(ep_r_edge, ep_r_service)
                #print(info['service_aoi'], info['edge_aoi'])
                s = s_prime
                step += 1
            scores_service += ep_r_service / total_t
            scores_edge += ep_r_edge / total_t
    return scores_service/turns, scores_edge/turns

def main():    
    seed = opt.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = DTN(device_num=opt.device_num, max_step=opt.max_step, max_channel_num=opt.max_channel_num)
    eval_env = copy.deepcopy(env)

    print("bandwidth:", env.link.bandwidth, "buffer size:", env.link.buffer_size)
    for i in range(env.device_num):
        print("Device: ", i, "loss rate:", env.devices[i].get_loss_rate(), " delay:", env.devices[i].get_delay())
   
    state_dim = env.observation_space.shape[0]
    action_s_dim = env.action_s_space.n
    action_p_dim = env.action_p_space.shape[0]
    max_e_steps = env.max_step

    write = opt.write
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs_v1/{}-device-seed{}'.format(env.device_num, opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    T_horizon = opt.T_horizon
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    print('Env-device_num:', env.device_num, ' state_dim:', state_dim, '  action_sample_dim:', action_s_dim, '  Random Seed:', seed, ' max_e_steps:', max_e_steps)
    print('\n')

    kwargs = {
        "state_dim": state_dim,
        "action_s_dim": action_s_dim,
        "action_p_dim": action_p_dim,
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "v_lr": opt.v_lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg": opt.l2_reg,
        "entropy_coef": opt.entropy_coef,  #hard env needs large value
        "adv_normalization": opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO(**kwargs) 
    if Loadmodel: model.load(ModelIdex)

    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:
        if total_steps % opt.env_reset_interval == 0:
            s, done, steps, ep_r = env.reset(),False, 0, 0
        else:
            done, steps, ep_r = False, 0, 0

        while not done:
            traj_lenth += 1
            steps += 1
            a_s, log_s = model.select_action(torch.from_numpy(s).float().to(device), 'sample')
            #s_p = np.append(s, a_s)
            a_p, log_p = model.select_action(torch.from_numpy(s).float().to(device), 'push')
            s_prime, r_s, r_p, done, info = env.step(a_s, a_p)
            # print("Step:", steps, "Reward_s:", r_s, "Reward_p:", r_p, "Done:", done)
            # print("Step:", steps, "a_s:", a_s, "a_p:", a_p, "Done:", done)
            # print("left frame:", env.left_frame, "edge aoi:", env.edge_aoi, "service aoi:", env.service_aoi)
            #model.put_data((s, a_s, r_s, a_p, r_p, s_prime, log_s, log_p, done)) 
            s = s_prime

            '''update if its time'''
            # if traj_lenth % T_horizon == 0:
            #     a_s_loss, a_p_loss, v_loss = model.train()
            #     traj_lenth = 0
            #     if write:
            #         writer.add_scalar('a_s_loss', a_s_loss, global_step=total_steps)
            #         writer.add_scalar('a_p_loss', a_p_loss, global_step=total_steps)
            #         writer.add_scalar('v_loss', v_loss, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                score_service, score_edge = evaluate_policy(eval_env, model, False)
                if write:
                    writer.add_scalar('ep_r_service', score_service, global_step=total_steps)
                    writer.add_scalar('ep_r_edge', score_edge, global_step=total_steps)
                print('seed:', seed, 'steps: {}'.format(int(total_steps)), 'score_service: {:.4f}'.format(score_service), 'score_edge: {:.4f}'.format(score_edge))

            total_steps += 1

            '''save model'''
            if opt.model_save and total_steps % save_interval == 0:
                model.save(total_steps)
            
    
if __name__ == "__main__":
    main()

