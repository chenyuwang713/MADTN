
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

parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--T_horizon', type=int, default=1000, help='lenth of long trajectory')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=200000, help='which model to load')
parser.add_argument('--Max_train_steps', type=int, default=200000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=50000, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=500, help='Model evaluating interval, in steps.')

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



def evaluate_policy(env, model, render):
    scores = 0
    turns = 3

    for j in range(turns):
        action = []
        s, done = env.reset(), False
        ep_r = 0.
        while not done:
            # Take deterministic actions at test time
            start = time.perf_counter()
            a_s, pi_a_s = model.evaluate(torch.from_numpy(s).float().to(device), 'sample')
            s_p = np.append(s, a_s)
            a_p, log_p = model.evaluate(torch.from_numpy(s_p).float().to(device), 'push')
            end = time.perf_counter()
            s_prime, r_s, r_p, done, info = env.step(a_s, a_p)
            ep_r += r_p
            s = s_prime
            #print("left frame:", env.left_frame, "edge aoi:", env.edge_aoi, "service aoi:", env.service_aoi)
        scores += ep_r
    return scores/turns

def main():    
    env = DTN()
    eval_env = copy.deepcopy(env)
   
    state_dim = env.observation_space.shape[0]
    action_p_dim = env.action_p_space.shape[0]
    action_s_dim = env.action_s_space.shape[0]
    max_e_steps = env.max_step

    write = opt.write
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-device-seed{}'.format(env.device_num, opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print('Env-device_num:', env.device_num, ' state_dim:', state_dim, '  action_sample_dim:', action_s_dim, ' action_push_dim:', action_p_dim, '  Random Seed:', seed, ' max_e_steps:', max_e_steps)
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
        s, done, steps, ep_r = env.reset(), False, 0, 0
        while not done:
            traj_lenth += 1
            steps += 1
            a_s, log_s = model.select_action(torch.from_numpy(s).float().to(device), 'sample')
            s_p = np.append(s, a_s)
            a_p, log_p = model.select_action(torch.from_numpy(s_p).float().to(device), 'push')
            s_prime, r_s, r_p, done, _ = env.step(a_s, a_p)
            # print("Step:", steps, "Reward_s:", r_s, "Reward_p:", r_p, "Done:", done)
            # print("Step:", steps, "a_s:", a_s, "a_p:", a_p, "Done:", done)
            # print("left frame:", env.left_frame, "edge aoi:", env.edge_aoi, "service aoi:", env.service_aoi)
            model.put_data((s, a_s, a_p, r_s, r_p, s_prime, log_s, log_p, done)) 
            s = s_prime

            '''update if its time'''
            if traj_lenth % T_horizon == 0:
                a_s_loss, a_p_loss, v_loss = model.train()
                traj_lenth = 0
                if write:
                    writer.add_scalar('a_s_loss', a_s_loss, global_step=total_steps)
                    writer.add_scalar('a_p_loss', a_p_loss, global_step=total_steps)
                    writer.add_scalar('v_loss', v_loss, global_step=total_steps)

            '''record & log'''
            if total_steps % eval_interval == 0:
                score  = evaluate_policy(eval_env, model, False)

                if write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                print('seed:', seed, 'steps: {}'.format(int(total_steps)), 'score:', score)
   
            total_steps += 1

            '''save model'''
            if total_steps % save_interval == 0:
                model.save(total_steps)
            
    
if __name__ == "__main__":
    main()

