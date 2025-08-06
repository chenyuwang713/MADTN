import copy
import time
import numpy as np
import torch
from torch.distributions import Categorical, Bernoulli
import math

from model import DiscretePolicy, ContinousPolicy, HybridValue, MultiBinaryPolicy, RewardContinuous

device = torch.device("mps" if torch.mps.is_available() else "cpu")

class PPO_Edge(object):
    def __init__(
            self,
            state_dim,
            action_s_dim,
            action_p_dim,
            net_width=200,
            gamma=0.99,
            lambd=0.95,
            a_lr=1e-3,
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

    # def evaluate(self, state, stage):
    #     #'''Deterministic Policy'''
    #     with torch.no_grad():
    #         if stage == 'sample':
    #             pi = self.actor_s.pi(state)
    #             a =  (pi > 0.5).float()
    #             return a.cpu().numpy().flatten(), 1.0

    def evaluate(self, state, stage):
        #'''Stochastic Policy'''
        with torch.no_grad():
            if stage == 'sample':
                state = state.reshape(1, -1).to(torch.float).to(device)
                pi = self.actor_s.pi(state)
                dist = Bernoulli(probs=pi)
                a = dist.sample()
                log_prob_a = dist.log_prob(a).sum(dim=-1).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), log_prob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))


    def train(self):
        s, a_s, r_s, s_prime, old_log_prob_s,  terminal = self.make_batch()
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
            s, a_s, td_target_s, adv_s, old_log_prob_s = \
                s[perm].clone(), a_s[perm].clone(), \
                td_target_s[perm].clone(), \
                adv_s[perm].clone(), \
                old_log_prob_s[perm].clone()

            policy_s_loss, value_loss = 0.,  0.

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

            return policy_s_loss,  value_loss

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
            s, a_s, r_s, s_prime, old_log_prob_s, done_mask = \
                torch.from_numpy(np.array(s_lst)).float().to(device), \
                torch.from_numpy(np.array(a_s_lst)).float().to(device), \
                torch.from_numpy(np.array(r_s_lst)).float().to(device), \
                torch.from_numpy(np.array(s_prime_lst)).float().to(device), \
                torch.from_numpy(np.array(old_log_prob_s_lst)).float().to(device), \
                torch.from_numpy(np.array(done_lst)).float().to(device), \

        return s, a_s, r_s, s_prime, old_log_prob_s, done_mask

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/hybrid_critic{}.pth".format(episode))
        torch.save(self.actor_s.state_dict(), "./model/actor_s{}.pth".format(episode))


class PPO_Hybrid(object):
    def __init__(
            self,
            state_dim,
            action_s_dim,
            action_p_dim,
            net_width=200,
            gamma=0.99,
            lambd=0.95,
            beta=0.6,
            a_lr=1e-3,
            v_lr=1e-4,
            clip_rate=0.2,
            K_epochs=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef=1e-3,
            adv_normalization = False,
            entropy_coef_decay = 0.99,
    ):
        self.actor_s = DiscretePolicy(state_dim, action_s_dim, net_width).to(device)      
        self.actor_s_optimizer = torch.optim.Adam(self.actor_s.parameters(), lr=a_lr)
        self.actor_p = ContinousPolicy(state_dim + action_s_dim, action_p_dim, net_width).to(device)      
        self.actor_p_optimizer = torch.optim.Adam(self.actor_p.parameters(), lr=a_lr)
        self.critic = HybridValue(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=v_lr)

        self.reward_c = RewardContinuous(state_dim, action_s_dim, action_p_dim, net_width).to(device)
        self.reward_c_optimizer = torch.optim.Adam(self.reward_c.parameters(), lr=v_lr)

        self.data = []
        self.gamma = gamma
        self.lambd = lambd
        self.beta = beta
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
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                dist = self.actor_p.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                log_prob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy().flatten(), log_prob_a
            
    def sample_action(self, state, stage):
        with torch.no_grad():
            if stage == 'sample':
                state = torch.from_numpy(state.reshape(1, -1)).to(device)
                pi = self.actor_s.pi(state)
                dist = Bernoulli(probs=pi)
                a_num = 100
                a = dist.sample((a_num,))
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return [sample.cpu().numpy().flatten() for sample in a], logprob_a
            elif stage == 'push':
                state = torch.from_numpy(state.reshape(1, -1)).to(device)
                dist = self.actor_p.get_dist(state)
                # sample times, can be higher or pretrain reward network in more complicated scenarios
                a_num = 100
                a = dist.sample((a_num,))
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=2)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return [sample.cpu().numpy().flatten() for sample in a], logprob_a
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    # def evaluate(self, state, stage):
    #     #'''Deterministic Policy'''
    #     with torch.no_grad():
    #         if stage == 'sample':
    #             pi = self.actor_s.pi(state, softmax_dim=0)
    #             a = torch.argmax(pi).item()
    #             return a, 1.0
    #         elif stage == 'push':
    #             state = state.reshape(1, -1).to(torch.float).to(device)
    #             a = self.actor_p(state)
    #             return a.cpu().numpy().flatten(), 0.0
    #         else:
    #             raise NotImplementedError('Unknown stage {}'.format(stage))
            
    def evaluate(self, state, stage):
        #'''Stochastic Policy'''
        with torch.no_grad():
            if stage == 'sample':
                pi = self.actor_s.pi(state)
                dist = Bernoulli(probs=pi)
                a = dist.sample()
                return a.cpu().numpy().flatten(), 1.0
            elif stage == 'push':
                state = torch.from_numpy(state.reshape(1, -1)).to(device)
                dist = self.actor_p.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                a = torch.softmax(a, dim=1)
                return a.cpu().numpy().flatten(), 0.0
            else:
                raise NotImplementedError('Unknown stage {}'.format(stage))

    def train(self):
        s, a_s, r_s, a_s_sample, a_p, r_p, a_p_sample, s_prime, old_log_prob_s, old_log_prob_p, terminal = self.make_batch()
        self.entropy_coef *= self.entropy_coef_decay  # exploring decay

        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            _, sample_num_s, a_s_dim = a_s_sample.shape
            _, sample_num_p, a_p_dim = a_p_sample.shape

            pr_s = self.reward_c(s.unsqueeze(1).expand(-1, sample_num_s, -1),
                                      a_s_sample,
                                      a_p.unsqueeze(1).expand(-1, sample_num_s, -1))

            pr_p = self.reward_c(s.unsqueeze(1).expand(-1, sample_num_p, -1),
                                      a_s.unsqueeze(1).expand(-1, sample_num_p, -1),
                                      a_p_sample)
            
            pr_s = pr_s.squeeze(-1)
            pr_p = pr_p.squeeze(-1)

            ex_s = pr_s.mean(dim=1, keepdim=True)
            ex_p = pr_p.mean(dim=1, keepdim=True)

            adv_s, td_target_s = self.get_adv_td(vs, vs_, r_s, ex_s, terminal)
            adv_p, td_target_p = self.get_adv_td(vs, vs_, r_p, ex_p, terminal)

        """PPO Update"""
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.optim_batch_size))
        for _ in range(self.K_epochs):
            # Shuff the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            s, a_s, a_p, td_target_s, td_target_p, adv_s, adv_p, old_log_prob_s, old_log_prob_p = \
                s[perm].clone(), a_s[perm].clone(), a_p[perm].clone(), \
                td_target_s[perm].clone(), td_target_p[perm].clone(), \
                adv_s[perm].clone(), adv_p[perm].clone(), \
                old_log_prob_s[perm].clone(), old_log_prob_p[perm].clone()

            policy_s_loss, policy_p_loss, reward_c_loss, value_loss = 0., 0., 0., 0.

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

                '''Push Actor Update'''
                s_p = torch.cat([s[index], a_s[index]], -1)
                distribution = self.actor_p.get_dist(s_p)
                dist_entropy = distribution.entropy().sum(1,keepdim=True)
                logprob_now = distribution.log_prob(a_p[index])
                ratio_p = torch.exp(logprob_now.sum(1, keepdim=True) - old_log_prob_p[index].sum(1, keepdim=True))
                surr1_p = ratio_p * adv_p[index]
                surr2_p= torch.clamp(ratio_p, 1 - self.clip_rate, 1 + self.clip_rate) * adv_p[index]
                actor_p_loss = - torch.min(surr1_p, surr2_p) - self.entropy_coef * dist_entropy
                policy_p_loss += actor_p_loss.mean()
                self.actor_p_optimizer.zero_grad()
                actor_p_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_p.parameters(), 40)
                self.actor_p_optimizer.step()

                '''Critic Update'''
                c_loss = (self.critic(s[index]) - td_target_s[index]).pow(2).mean() + (self.critic(s[index]) - td_target_p[index]).pow(2).mean()
                value_loss += c_loss

                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

                '''reward update'''
                pre_c = self.reward_c(s[index], a_s[index], a_p[index])
                r_c_loss = (pre_c-r_p[index]).pow(2).mean()
                reward_c_loss += r_c_loss

                # l2 regression
                for name, param in self.reward_c.named_parameters():
                    if 'weight' in name:
                        r_c_loss += param.pow(2).sum() * self.l2_reg

            end = time.perf_counter()
            policy_s_loss /= c_optim_iter_num
            policy_p_loss /= c_optim_iter_num
            value_loss /= c_optim_iter_num

            return policy_s_loss, policy_p_loss, value_loss

    def get_adv_td(self, vs, vs_, r, ex, terminal):
        '''dw for TD_target and Adv'''
        # DAE advantage
        deltas = r + self.gamma * vs_  - vs - self.beta * ex
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
        s_lst, a_s_lst, r_s_lst, a_s_sample_lst, a_p_lst, r_p_lst, a_p_sample_lst, s_prime_lst, old_log_prob_s_lst, old_log_prob_p_lst, done_lst = [], [], [], [], [], [], [], [], [], [], []

        for transition in self.data:
            s, a_s, r_s, a_s_sample, a_p, r_p, a_p_sample, s_prime, old_log_prob_s, old_log_prob_p, done = transition
            s_lst.append(s)
            a_s_lst.append(a_s)
            r_s_lst.append([r_s])
            a_s_sample_lst.append(a_s_sample)
            a_p_lst.append(a_p)
            r_p_lst.append([r_p])
            a_p_sample_lst.append(a_p_sample)
            s_prime_lst.append(s_prime)
            old_log_prob_s_lst.append(old_log_prob_s)
            old_log_prob_p_lst.append(old_log_prob_p)
            done_lst.append([done])

        self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a_s, r_s, a_s_sample, a_p, r_p, a_p_sample, s_prime, old_prob_s, old_log_prob_p, done_mask = \
                torch.from_numpy(np.array(s_lst)).float().to(device), \
                torch.from_numpy(np.array(a_s_lst)).float().to(device), \
                torch.from_numpy(np.array(r_s_lst)).float().to(device), \
                torch.from_numpy(np.array(a_s_sample_lst)).float().to(device), \
                torch.from_numpy(np.array(a_p_lst)).float().to(device), \
                torch.from_numpy(np.array(r_p_lst)).float().to(device), \
                torch.from_numpy(np.array(a_p_sample_lst)).float().to(device), \
                torch.from_numpy(np.array(s_prime_lst)).float().to(device), \
                torch.from_numpy(np.array(old_log_prob_s_lst)).float().to(device), \
                torch.from_numpy(np.array(old_log_prob_p_lst)).float().to(device), \
                torch.from_numpy(np.array(done_lst)).float().to(device), \

        return s, a_s, r_s, a_s_sample, a_p, r_p, a_p_sample, s_prime, old_prob_s, old_log_prob_p, done_mask

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/hybrid_critic{}.pth".format(episode))
        torch.save(self.actor_s.state_dict(), "./model/actor_s{}.pth".format(episode))
        torch.save(self.actor_p.state_dict(), "./model/actor_p{}.pth".format(episode))









