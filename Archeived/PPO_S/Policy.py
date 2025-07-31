import copy
import time
import numpy as np
import torch
from torch.distributions import Categorical, Bernoulli
import math

from model import DiscretePolicy, ContinousPolicy, HybridValue, MultiBinaryPolicy

device = torch.device("mps" if torch.mps.is_available() else "cpu")

class PPO(object):
    def __init__(
            self,
            state_dim,
            action_s_dim,
            net_width=200,
            gamma=0.99,
            lambd=0.95,
            a_lr=1e-4,
            v_lr=1e-4,
            clip_rate=0.2,
            K_epochs=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef=1e-3,
            adv_normalization = False,
            entropy_coef_decay = 0.99,
    ):

        self.actor_s = MultiBinaryPolicy(state_dim, action_s_dim, net_width).to(device)      
        self.actor_s_optimizer = torch.optim.Adam(self.actor_s.parameters(), lr=a_lr)

        self.critic = HybridValue(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=v_lr)

        self.data = []
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.optim_batch_size = batch_size
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay
    
    def select_action(self, state, stage):
        '''Stochastic Policy'''
        with torch.no_grad():
            if stage == 'sample':
                pi = self.actor_s.pi(state)
                dist = Bernoulli(probs=pi)
                a = dist.sample()
                log_prob_a = dist.log_prob(a).sum(dim=-1).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), log_prob_a
            elif stage == 'push':
                state = state.reshape(1, -1).to(torch.float).to(device)
                dist = self.actor_p.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), logprob_a

    def evaluate(self, state, stage):
        #'''Deterministic Policy'''
        with torch.no_grad():
            if stage == 'sample':
                pi = self.actor_s.pi(state)
                dist = Bernoulli(probs=pi)
                a = dist.sample()
                return a.cpu().numpy().flatten(), 1.0
            elif stage == 'push':
                state = state.reshape(1, -1).to(torch.float).to(device)
    
                dist = self.actor_p.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                return a.cpu().numpy().flatten(), 0.0
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))


    def train(self):
        s, a_s, r_s, s_prime, old_log_prob_s, terminal = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay

        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)
            adv_s, td_target_s = self.get_adv_td(vs, vs_, r_s, terminal)
        
        """PPO Update"""
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        for _ in range(self.K_epochs):
            # Shuff the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            s, a_s,  td_target_s, adv_s, old_log_prob_s,  = \
                s[perm].clone(), a_s[perm].clone(),  \
                td_target_s[perm].clone(),  \
                adv_s[perm].clone(),  \
                old_log_prob_s[perm].clone()
            
            policy_s_loss,  value_loss = 0., 0.

            # start time
            start = time.perf_counter()
            for i in range(c_optim_iter_num):
                index = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, s.shape[0]))

                '''Actor Update'''

                '''Sample Actor Update'''
                distribution = Bernoulli(self.actor_s.pi(s[index]))
                dist_entropy = distribution.entropy().sum(1,keepdim=True)
                logprob_now = distribution.log_prob(a_s[index])
                ratio_s = torch.exp(logprob_now.sum(1, keepdim=True) - old_log_prob_s[index].sum(1, keepdim=True))
                surr1_s = ratio_s * adv_s[index]
                surr2_s= torch.clamp(ratio_s, 1 - self.clip_rate, 1 + self.clip_rate) * adv_s[index]
                actor_s_loss = - torch.min(surr1_s, surr2_s) - self.entropy_coef * dist_entropy
                policy_s_loss += actor_s_loss.mean()
                self.actor_s_optimizer.zero_grad()
                actor_s_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_s.parameters(), 40)
                self.actor_s_optimizer.step()


                '''Critic Update'''
                c_loss = (self.critic(s[index]) - td_target_s[index]).pow(2).mean()  
                value_loss += c_loss

                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

            end = time.perf_counter()
            policy_s_loss /= c_optim_iter_num
            value_loss /= c_optim_iter_num

            return policy_s_loss, value_loss

                
    def get_adv_td(self, vs, vs_, r, terminal):
        '''dw for TD_target and Adv'''
        # DAE advantage
        deltas = r + self.gamma * vs_  - vs
        deltas = deltas.cpu().flatten(end_dim=0).numpy()
        adv = [0]

        '''done for GAE'''
        for dlt, mask in zip(deltas[::-1], terminal.cpu().flatten().numpy()[::-1]):
            advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
            adv.append(advantage)
        adv.reverse()
        adv = copy.deepcopy(adv[0:-1])
   
        adv = torch.from_numpy(np.array(adv)).float().to(device)
        td_target = adv + vs

        return adv, td_target

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_s_lst, r_s_lst, s_prime_lst, old_log_prob_s_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a_s, r_s, s_prime, old_log_prob_s, done = transition
            s_lst.append(s)
            a_s_lst.append(a_s)
            r_s_lst.append([r_s])
            s_prime_lst.append(s_prime)
            old_log_prob_s_lst.append(old_log_prob_s)
            done_lst.append([done])

        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a_s, r_s,  s_prime, old_log_prob_s, done_mask = \
                torch.from_numpy(np.array(s_lst)).float().to(device), \
                torch.from_numpy(np.array(a_s_lst)).float().to(device), \
                torch.from_numpy(np.array(r_s_lst)).float().to(device), \
                torch.from_numpy(np.array(s_prime)).float().to(device), \
                torch.from_numpy(np.array(old_log_prob_s_lst)).float().to(device), \
                torch.from_numpy(np.array(done_lst)).float().to(device), \
                
        return s, a_s, r_s, s_prime, old_log_prob_s, done_mask
    
    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/hybrid_critic{}.pth".format(episode))
        torch.save(self.actor_s.state_dict(), "./model/actor_s{}.pth".format(episode))




