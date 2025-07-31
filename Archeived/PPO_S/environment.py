import gymnasium as gym

import numpy as np

from typing import Optional
from gymnasium import spaces
import random
import heapq,copy

import torch

min_bw, max_bw = (5, 5)
min_buffer, max_buffer = (10, 10)
min_uniform_gen, max_uniform_gen = (1, 5)

min_delay, max_delay = (0, 5)
min_loss_rate, max_loss_rate = (0.05, 0.5) 


class Device():
    def __init__(self, device_id, gen_mode="uniform", max_step=100):
        self.device_id = device_id
        self.gen_mode = gen_mode
        self.sys_time = 0
        self.max_step = max_step

        self.delay_bound = random.randint(min_delay, max_delay)
        self.loss_rate = random.uniform(min_loss_rate, max_loss_rate)
        self.next_gen_time = self.step_next_gen()

    def step(self, event_time):
        if event_time == self.next_gen_time:
            self.sys_time = event_time
            self.next_gen_time = self.next_gen_time + self.step_next_gen()
            
    def step_next_gen(self):
        if self.gen_mode == "uniform":
            return round(random.uniform(min_uniform_gen, max_uniform_gen))
        else:
            raise NotImplementedError("Packet generation mode not implemented: {}".format(self.gen_mode))
        

    def get_sys_time(self):
        return self.sys_time

    def get_id(self):
        return self.device_id
    
    def get_delay(self):
        return random.randint(self.delay_bound, self.delay_bound * 2)
    
    def get_loss_rate(self):
        return self.loss_rate

    def reset(self):
        self.sys_time = 0 
        self.next_gen_time = self.step_next_gen()


class Link():
    def __init__(self, max_step, bandwidth=None, buffer_size=None):
        self.max_step = max_step
        self.bandwidth = bandwidth if bandwidth is not None else random.randint(min_bw, max_bw)
        self.buffer_size = buffer_size if buffer_size is not None else random.randint(min_buffer, max_buffer)

        self.airline = []
        self.buffer = []

    def get_packets_to_process(self, event_time): 
        # move packets from airline to buffer if there is any:
        while len(self.airline) > 0 and len(self.buffer) <= self.buffer_size and self.airline[0][0] <= event_time:
            packet = heapq.heappop(self.airline)
            self.buffer.append(packet)
            
        # remove all delayed packets that cannot log in to the buffer from the airline
        while len(self.airline) > 0 and self.airline[0][0] <= event_time:
            packet = heapq.heappop(self.airline)
        packets_to_process = copy.deepcopy(self.buffer[:self.bandwidth])
        self.buffer = copy.deepcopy(self.buffer[self.bandwidth:])

        return packets_to_process

    def packet_enter_line(self, event_time, device):
        packet_process_time = event_time + device.get_delay()
        packet_system_time = device.get_sys_time()
        packet_device_id = device.get_id()

        if random.random() > device.get_loss_rate():
            packet = (packet_process_time, packet_system_time, random.random(), packet_device_id)
            heapq.heappush(self.airline, packet)


    def reset(self):
        self.airline = []
        self.buffer = []


class Service():
    def __init__(self,  device_num:int):
        self.device_num = device_num
        self.priority = list(range(self.device_num))
        random.shuffle(self.priority) 

    def service_priority_update(self):
        random.shuffle(self.priority) 

    def get_priority(self):
        return np.array(self.priority)
    

class DTN(gym.Env):
    def __init__(self, device_num: int = 10):
        
        self.max_step = 100
        self.left_frame = self.max_step
        self.device_num = device_num

        self.devices = [Device(id, gen_mode='uniform', max_step=self.max_step) for id in range(self.device_num)]
        self.service = Service(device_num=self.device_num)
        self.link = Link(max_step=self.max_step)

        self.cur_time = 0
        self.sys_time = np.array([int(0) for _ in range(self.device_num)])
        self.last_schedule = np.array([int(0) for _ in range(self.device_num)])
        self.edge_aoi = np.array([int(1) for _ in range(self.device_num)], dtype=np.float32)
    
        # State: [ (CURR_FRAME, BUFF_LENGTH) Edge AOI, Last_SENT_PERIORD, WEIGHT_OF DEVICE]
        self.observation_space = spaces.Box(
            low = np.array([0 for _ in range(3 * self.device_num + 2)], dtype=np.float32),
            high = np.array([np.finfo(np.float32).max for _ in range(3 * self.device_num + 2)], dtype=np.float32))
        
        #self.action_s_space = spaces.Discrete(self.device_num)
        self.action_s_space = spaces.MultiBinary(self.device_num)

    def get_cur_time(self):
        return self.cur_time
    
    def step_cur_time(self):
        self.cur_time += 1
        self.left_frame -= 1
    
    def process_packets(self, packets, event_time):
        for packet in packets:
            _, sys_time, _, device_id = packet
            self.sys_time[device_id] = max(self.sys_time[device_id], sys_time)        
        self.edge_aoi = event_time - self.sys_time + 1 

    def step_aoi_info(self):
        self.edge_aoi += 1
        self.last_schedule += 1

    def step(self, action_s):
        s = action_s

        reward_s = 0.0
        self.step_cur_time()
        self.step_aoi_info()

        #print("a_s", action_s)
        #print(self.sys_time)

        ## Collect Data from Devices
        for device in self.devices:
            device.step(self.get_cur_time())
            if int(s[device.get_id()]) == 1:
                self.last_schedule[device.get_id()] = 1
                self.link.packet_enter_line(self.get_cur_time(), device)
        packets_to_process = self.link.get_packets_to_process(self.get_cur_time())
        
        self.process_packets(packets_to_process, self.get_cur_time())
        reward_s = - np.mean(self.edge_aoi * self.service.get_priority() / sum(self.service.get_priority()))

        self.service.service_priority_update()
        #print("timestep: ", self.get_cur_time(), "left_time_frame:", self.left_frame, "action:", action_s, "reward_s:", reward_s)
        #print("edge_aoi:", self.edge_aoi, "last_schedule:", self.last_schedule, "sys_time:", self.sys_time)
        

        done = False
        if self.left_frame <= 0:
            self.left_frame = self.max_step
            done = True
        # print("action_s: ", s, "action_p:", push_devices_id)
        # print("edge aoi:" ,self.edge_aoi, "last schedule:", self.last_schedule, "action_s:", s)
        
        return self.get_obs(), reward_s, done, 1
    
    def get_obs(self):
        self.state = [self.left_frame, len(self.link.buffer)]
        self.state.extend(self.edge_aoi)
        self.state.extend(self.last_schedule)
        self.state.extend(self.service.get_priority())
        self.state = np.array(self.state, dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        pass
        
    def reset(self):
        self.left_frame = self.max_step
        self.cur_time = 0
        for device in self.devices:
            device.reset()
        self.sys_time = np.array([int(0) for _ in range(self.device_num)], dtype=np.float32)
        self.last_schedule = np.array([int(0) for _ in range(self.device_num)], dtype=np.float32)
        self.edge_aoi = np.array([int(1) for _ in range(self.device_num)],dtype=np.float32)
        self.link.reset()

        return self.get_obs() 


def main():
    random.seed(2)
    env = DTN(device_num=10)
    for i in range(502):
        #env.step(np.array([1, 0, 0, 0, 0]), np.array([0.6, 0.3, 0.1, 0.4, 0.5]))
        env.step(np.array([random.randint(0, 1) for _ in range(env.device_num)]))

if __name__ == "__main__":
    main()
    
         

        

        



