#!/usr/bin/env python3
"""
DQN Traffic Signal Control - COMPLETE VERSION
All three headway patterns: Uniform, Random, Platoon
"""

import os
import sys
import traci
import numpy as np
import pandas as pd
import random
from collections import defaultdict, deque
import time
import math
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration"""
    
    # Simulation
    SUMO_CFG = "net.sumocfg"
    SUMO_GUI = False
    DURATION = 3600  # 1 hour per episode
    WARMUP = 300
    TLS_IDS = ['J1', 'J2', 'J3']
    
    # Headway patterns
    HEADWAY_PATTERNS = ['uniform', 'random', 'platoon']
    
    # Signal timing
    MIN_GREEN = 20
    MAX_GREEN = 60
    YELLOW_TIME = 3
    
    # DQN Architecture
    STATE_DIM = 36
    ACTION_DIM = 4
    
    # Training
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    BATCH_SIZE = 64
    MEMORY_SIZE = 50000
    TRAIN_START = 1000
    TRAIN_FREQ = 4
    
    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 10000
    
    # Reward weights
    W_QUEUE = -5.0
    W_DELAY = -3.0
    W_SPEED = 4.0
    W_THROUGHPUT = 0.1
    W_IMBALANCE = -2.0
    
    # Normalization
    MAX_QUEUE_PER_APPROACH = 15
    MAX_DELAY = 200
    FREE_SPEED = 50.0
    
    # Traffic flows (vehicles per hour)
    FLOWS = {
        'uniform': {'eb': 600, 'wb': 600, 'nb': 200, 'sb': 200},
        'random': {'eb': 600, 'wb': 600, 'nb': 200, 'sb': 200},
        'platoon': {'eb': 600, 'wb': 600, 'nb': 200, 'sb': 200}
    }
    
    # Directories
    OUTPUT_DIR = "results"
    MODEL_DIR = "models"

config = Config()

# Create directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

# ============================================================================
# DQN NETWORK
# ============================================================================

class DuelingDQN(nn.Module):
    """Dueling DQN with proper dimensions"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Feature layers
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

# ============================================================================
# REPLAY MEMORY
# ============================================================================

class ReplayMemory:
    """Simple replay memory"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.memory)

# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """DQN Agent"""
    
    def __init__(self, tls_id):
        self.tls_id = tls_id
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.target_net = DuelingDQN(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Memory
        self.memory = ReplayMemory(config.MEMORY_SIZE)
        
        # Training state
        self.steps = 0
        self.epsilon = config.EPSILON_START
        
        # Phase tracking
        self.phase_start_time = 0
        self.last_state = None
        self.last_action = None
        
        # Metrics
        self.episode_reward = 0
    
    def get_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(config.EPSILON_END, 
                          self.epsilon * 0.9995)
        return self.epsilon
    
    def get_state(self):
        """Extract state features"""
        state = []
        
        # Define approach lanes for each intersection
        if self.tls_id == 'J1':
            approaches = [
                ['E4_0', 'E4_1'],  # North
                ['-E5_0', '-E5_1'],  # South
                ['-E1_0', '-E1_1', '-E1_2'],  # East
                ['E0_0', 'E0_1', 'E0_2']  # West
            ]
        elif self.tls_id == 'J2':
            approaches = [
                ['E6_0', 'E6_1'],
                ['-E7_0', '-E7_1'],
                ['-E2_0', '-E2_1', '-E2_2'],
                ['E1_0', 'E1_1', 'E1_2']
            ]
        else:  # J3
            approaches = [
                ['E8_0', 'E8_1'],
                ['-E9_0', '-E9_1'],
                ['-E3_0', '-E3_1', '-E3_2'],
                ['E2_0', 'E2_1', 'E2_2']
            ]
        
        # Get features for each approach (7 features per approach = 28 total)
        for approach in approaches:
            valid_lanes = [l for l in approach if l in traci.lane.getIDList()]
            
            if valid_lanes:
                # 1. Queue length
                queue = sum(traci.lane.getLastStepVehicleNumber(l) for l in valid_lanes)
                
                # 2. Waiting time
                waiting = sum(traci.lane.getWaitingTime(l) for l in valid_lanes)
                
                # 3. Mean speed
                speeds = [traci.lane.getLastStepMeanSpeed(l) for l in valid_lanes]
                speed = np.mean(speeds) if speeds else 0
                
                # 4. Occupancy
                occ = [traci.lane.getLastStepOccupancy(l) for l in valid_lanes]
                occupancy = np.mean(occ) if occ else 0
                
                # 5. Vehicle count
                vehicle_count = len(traci.lane.getLastStepVehicleIDs(l) for l in valid_lanes)
                
                # 6. Approaching vehicles (within 100m)
                approaching = 0
                for lane in valid_lanes:
                    for veh in traci.lane.getLastStepVehicleIDs(lane):
                        try:
                            pos = traci.vehicle.getLanePosition(veh)
                            lane_len = traci.lane.getLength(lane)
                            if pos > lane_len - 100:
                                approaching += 1
                        except:
                            pass
                
                # 7. Stopped vehicles
                stopped = 0
                for lane in valid_lanes:
                    for veh in traci.lane.getLastStepVehicleIDs(lane):
                        try:
                            if traci.vehicle.getSpeed(veh) < 0.1:
                                stopped += 1
                        except:
                            pass
                
                state.extend([queue, waiting, speed, occupancy, approaching, stopped, vehicle_count])
            else:
                state.extend([0, 0, 0, 0, 0, 0, 0])
        
        # Add phase information
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            phase_duration = traci.simulation.getTime() - self.phase_start_time
        except:
            current_phase = 0
            phase_duration = 0
        
        # Phase one-hot (4 features)
        phase_one_hot = [0, 0, 0, 0]
        phase_one_hot[current_phase] = 1
        state.extend(phase_one_hot)
        
        # Phase duration (1 feature)
        state.append(phase_duration)
        
        # Time of day (1 feature)
        sim_time = traci.simulation.getTime()
        time_of_day = (sim_time % 3600) / 3600
        state.append(time_of_day)
        
        # Padding to reach 36
        state.extend([0, 0])
        
        return np.array(state, dtype=np.float32)
    
    def calculate_reward(self):
        """Balanced reward function"""
        
        # Get approach queues
        approach_queues = []
        total_queue = 0
        total_waiting = 0
        
        # Define approaches
        if self.tls_id == 'J1':
            approaches = [
                ['E4_0', 'E4_1'],
                ['-E5_0', '-E5_1'],
                ['-E1_0', '-E1_1', '-E1_2'],
                ['E0_0', 'E0_1', 'E0_2']
            ]
        elif self.tls_id == 'J2':
            approaches = [
                ['E6_0', 'E6_1'],
                ['-E7_0', '-E7_1'],
                ['-E2_0', '-E2_1', '-E2_2'],
                ['E1_0', 'E1_1', 'E1_2']
            ]
        else:
            approaches = [
                ['E8_0', 'E8_1'],
                ['-E9_0', '-E9_1'],
                ['-E3_0', '-E3_1', '-E3_2'],
                ['E2_0', 'E2_1', 'E2_2']
            ]
        
        # Collect data
        for approach in approaches:
            valid_lanes = [l for l in approach if l in traci.lane.getIDList()]
            if valid_lanes:
                queue = sum(traci.lane.getLastStepVehicleNumber(l) for l in valid_lanes)
                waiting = sum(traci.lane.getWaitingTime(l) for l in valid_lanes)
                
                total_queue += queue
                total_waiting += waiting
                approach_queues.append(queue)
        
        # Get current speed
        speeds = []
        for lane in traci.lane.getIDList():
            if not lane.startswith(':'):
                try:
                    speeds.append(traci.lane.getLastStepMeanSpeed(lane))
                except:
                    pass
        current_speed = np.mean(speeds) * 3.6 if speeds else 0
        
        # Normalize
        queue_norm = min(total_queue / (4 * config.MAX_QUEUE_PER_APPROACH), 1.0)
        delay_norm = min(total_waiting / config.MAX_DELAY, 1.0)
        speed_norm = min(current_speed / config.FREE_SPEED, 1.0)
        
        # Imbalance penalty
        imbalance = 0
        if approach_queues and np.mean(approach_queues) > 0:
            cv = np.std(approach_queues) / (np.mean(approach_queues) + 1e-6)
            if cv > 0.7:
                imbalance = config.W_IMBALANCE * (cv - 0.7)
        
        # Short phase penalty
        short_phase = 0
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            phase_duration = traci.simulation.getTime() - self.phase_start_time
            if current_phase in [0, 2] and phase_duration < config.MIN_GREEN:
                short_phase = -1.0
        except:
            pass
        
        # Throughput (very low weight)
        vehicles_passed = 0
        for lane in traci.lane.getIDList():
            if self.tls_id in lane and 'internal' not in lane:
                vehicles_passed += traci.lane.getLastStepVehicleNumber(lane)
        throughput_norm = min(vehicles_passed / 5, 1.0)
        
        # Combine rewards
        reward = (config.W_QUEUE * queue_norm + 
                  config.W_DELAY * delay_norm + 
                  config.W_SPEED * speed_norm + 
                  config.W_THROUGHPUT * throughput_norm +
                  imbalance + short_phase)
        
        self.episode_reward += reward
        return reward
    
    def act(self, state, training=True):
        """Select action"""
        
        # Get current phase and duration
        try:
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            phase_duration = traci.simulation.getTime() - self.phase_start_time
        except:
            current_phase = 0
            phase_duration = config.MIN_GREEN + 1
        
        # Enforce minimum green time
        if phase_duration < config.MIN_GREEN and current_phase in [0, 2]:
            return current_phase
        
        # Exploration
        if training and random.random() < self.get_epsilon():
            valid_actions = [0, 1, 2, 3]
            if phase_duration < config.MIN_GREEN + 3:
                valid_actions = [a for a in valid_actions if a not in [1, 3]]
            return random.choice(valid_actions)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions
            if phase_duration < config.MIN_GREEN + 2:
                q_values[1] = -1e9
                q_values[3] = -1e9
            
            return int(np.argmax(q_values))
    
    def learn(self):
        """Update network"""
        if len(self.memory) < config.BATCH_SIZE:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + config.GAMMA * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_phase_start(self):
        """Update phase start time"""
        self.phase_start_time = traci.simulation.getTime()


# ============================================================================
# DQN CONTROLLER
# ============================================================================

class DQNController:
    """Multi-agent DQN controller"""
    
    def __init__(self):
        self.agents = {tls: DQNAgent(tls) for tls in config.TLS_IDS}
        self.training = True
        self.steps = 0
    
    def control_step(self, step):
        """Execute control step"""
        self.steps += 1
        
        for tls, agent in self.agents.items():
            try:
                # Get state
                state = agent.get_state()
                
                # Select action
                action = agent.act(state, training=self.training)
                
                # Execute action
                current_phase = traci.trafficlight.getPhase(tls)
                if action != current_phase:
                    phase_duration = traci.simulation.getTime() - agent.phase_start_time
                    if phase_duration >= config.MIN_GREEN:
                        traci.trafficlight.setPhase(tls, action)
                        agent.update_phase_start()
                
                # Calculate reward
                reward = agent.calculate_reward()
                
                # Get next state
                next_state = agent.get_state()
                
                # Store transition
                if agent.last_state is not None:
                    done = False
                    agent.remember(agent.last_state, agent.last_action, reward, next_state, done)
                    agent.steps += 1
                
                # Update last state and action
                agent.last_state = state
                agent.last_action = action
                
                # Learn
                if self.training and agent.steps > config.TRAIN_START:
                    if agent.steps % config.TRAIN_FREQ == 0:
                        agent.learn()
            
            except Exception as e:
                # Silently handle errors during control
                pass
        
        # Update target networks
        if self.steps % 1000 == 0:
            for agent in self.agents.values():
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
        for agent in self.agents.values():
            agent.training = training

# ============================================================================
# BASELINE CONTROLLER
# ============================================================================

class BaselineController:
    """Rule-based adaptive control"""
    
    def __init__(self):
        self.phase_start_time = {tls: 0 for tls in config.TLS_IDS}
        self.offsets = {'J1': 0, 'J2': 20, 'J3': 40}
    
    def get_queue(self, tls):
        """Get queue length for intersection"""
        try:
            if tls == 'J1':
                main = sum(traci.lane.getLastStepVehicleNumber(l) 
                          for l in ['-E1_0', '-E1_1', '-E1_2', 'E0_0', 'E0_1', 'E0_2'] 
                          if l in traci.lane.getIDList())
                cross = sum(traci.lane.getLastStepVehicleNumber(l) 
                           for l in ['E4_0', 'E4_1', '-E5_0', '-E5_1']
                           if l in traci.lane.getIDList())
            elif tls == 'J2':
                main = sum(traci.lane.getLastStepVehicleNumber(l) 
                          for l in ['-E2_0', '-E2_1', '-E2_2', 'E1_0', 'E1_1', 'E1_2']
                          if l in traci.lane.getIDList())
                cross = sum(traci.lane.getLastStepVehicleNumber(l) 
                           for l in ['E6_0', 'E6_1', '-E7_0', '-E7_1']
                           if l in traci.lane.getIDList())
            else:
                main = sum(traci.lane.getLastStepVehicleNumber(l) 
                          for l in ['-E3_0', '-E3_1', '-E3_2', 'E2_0', 'E2_1', 'E2_2']
                          if l in traci.lane.getIDList())
                cross = sum(traci.lane.getLastStepVehicleNumber(l) 
                           for l in ['E8_0', 'E8_1', '-E9_0', '-E9_1']
                           if l in traci.lane.getIDList())
            return main, cross
        except:
            return 0, 0
    
    def control_step(self, step):
        """Execute control"""
        for tls in config.TLS_IDS:
            try:
                current_phase = traci.trafficlight.getPhase(tls)
                phase_duration = traci.simulation.getTime() - self.phase_start_time[tls]
                
                if current_phase in [0, 2]:  # Green phases
                    main, cross = self.get_queue(tls)
                    
                    # Decide to extend or terminate
                    if current_phase == 0:  # Main street green
                        should_extend = main > cross * 1.2
                    else:  # Cross street green
                        should_extend = cross > main * 1.2
                    
                    # Coordination
                    cycle_time = (traci.simulation.getTime() - self.offsets[tls]) % 90
                    if 80 < cycle_time < 90:
                        should_extend = False
                    
                    if should_extend and phase_duration < config.MAX_GREEN:
                        new_duration = min(phase_duration + 5, config.MAX_GREEN)
                        traci.trafficlight.setPhaseDuration(tls, new_duration)
                    elif phase_duration >= config.MIN_GREEN:
                        next_phase = (current_phase + 1) % 4
                        traci.trafficlight.setPhase(tls, next_phase)
                        self.phase_start_time[tls] = traci.simulation.getTime()
                
                elif current_phase in [1, 3]:  # Yellow phases
                    if phase_duration >= config.YELLOW_TIME:
                        next_phase = (current_phase + 1) % 4
                        traci.trafficlight.setPhase(tls, next_phase)
                        self.phase_start_time[tls] = traci.simulation.getTime()
            except:
                pass

# ============================================================================
# TRAFFIC GENERATOR - WITH ALL THREE PATTERNS
# ============================================================================

class TrafficGenerator:
    """Generate traffic with different headway patterns"""
    
    def __init__(self):
        self.routes = ['EB', 'WB', 'NB', 'SB']
        self.route_edges = {
            'EB': 'E0 E1 E2 E3',
            'WB': '-E3 -E2 -E1 -E0',
            'NB': 'E4 E1 E6',
            'SB': '-E5 -E1 -E7'
        }
    
    def generate_uniform_headways(self, flow_rate, n_vehicles):
        """Generate uniform (exponential) headways"""
        mean_headway = 3600 / flow_rate
        return np.random.exponential(mean_headway, n_vehicles)
    
    def generate_random_headways(self, flow_rate, n_vehicles):
        """Generate random (normal) headways"""
        headways = np.abs(np.random.normal(2.5, 1.2, n_vehicles))
        return np.clip(headways, 0.5, 10)
    
    def generate_platoon_headways(self, flow_rate, n_vehicles):
        """Generate platooned headways"""
        headways = []
        i = 0
        while i < n_vehicles:
            if random.random() < 0.3:  # 30% chance of platoon
                platoon_size = random.randint(3, 6)
                for _ in range(platoon_size):
                    headways.append(random.uniform(0.8, 1.5))  # Close headway
                i += platoon_size
            else:
                headways.append(random.uniform(3.0, 6.0))  # Gap between platoons
                i += 1
        return np.array(headways[:n_vehicles])
    
    def generate_route_file(self, pattern, flows, seed, filename="route.rou.xml"):
        """Generate route file with specified pattern"""
        
        np.random.seed(seed)
        random.seed(seed)
        
        with open(filename, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')
            f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                   'length="5" maxSpeed="50"/>\n\n')
            
            # Write routes
            for route, edges in self.route_edges.items():
                f.write(f'    <route id="{route}" edges="{edges}"/>\n')
            
            f.write('\n')
            
            # Generate vehicles for each route
            total_vehicles = 0
            
            for route in self.routes:
                if route in ['EB', 'WB']:
                    flow_rate = flows['eb'] if route == 'EB' else flows['wb']
                else:
                    flow_rate = flows['nb'] if route == 'NB' else flows['sb']
                
                n_vehicles = int(flow_rate * config.DURATION / 3600)
                
                # Select headway generator based on pattern
                if pattern == 'uniform':
                    headways = self.generate_uniform_headways(flow_rate, n_vehicles)
                elif pattern == 'random':
                    headways = self.generate_random_headways(flow_rate, n_vehicles)
                else:  # platoon
                    headways = self.generate_platoon_headways(flow_rate, n_vehicles)
                
                # Generate vehicles
                depart_time = random.uniform(0, 10)
                for i, h in enumerate(headways):
                    depart_time += h
                    if depart_time < config.DURATION:
                        f.write(f'    <vehicle id="{route}_{i}" type="car" '
                               f'route="{route}" depart="{depart_time:.2f}"/>\n')
                        total_vehicles += 1
            
            f.write('</routes>\n')
        
        print(f"    Generated {total_vehicles} vehicles with {pattern} pattern")
        return total_vehicles

# ============================================================================
# DATA COLLECTOR
# ============================================================================

class DataCollector:
    """Collect performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.queues = []
        self.speeds = []
        self.vehicle_times = {}
        self.completed = 0
        self.timestamps = []
        self.throughput_history = []
    
    def collect(self, step):
        """Collect metrics"""
        if step % 10 != 0:
            return
        
        current_time = traci.simulation.getTime()
        self.timestamps.append(current_time)
        
        # Queue length
        total_queue = 0
        lanes = 0
        try:
            for lane in traci.lane.getIDList():
                if not lane.startswith(':'):
                    total_queue += traci.lane.getLastStepVehicleNumber(lane)
                    lanes += 1
            if lanes > 0:
                self.queues.append(total_queue / lanes)
        except:
            pass
        
        # Speed
        try:
            vehicles = traci.vehicle.getIDList()
            if vehicles:
                speeds = []
                for v in vehicles:
                    try:
                        speeds.append(traci.vehicle.getSpeed(v))
                    except:
                        pass
                if speeds:
                    self.speeds.append(np.mean(speeds) * 3.6)
                
                # Track new vehicles
                for v in vehicles:
                    if v not in self.vehicle_times:
                        self.vehicle_times[v] = {
                            'enter': current_time,
                            'route_length': 2000,
                            'free_flow_time': 144
                        }
        except:
            pass
        
        # Track completed vehicles
        try:
            current_vehicles = set(traci.vehicle.getIDList())
            for v in list(self.vehicle_times.keys()):
                if v not in current_vehicles and 'exit' not in self.vehicle_times[v]:
                    self.vehicle_times[v]['exit'] = current_time
                    self.completed += 1
        except:
            pass
        
        # Throughput
        if current_time > config.WARMUP:
            throughput = self.completed * 3600 / (current_time - config.WARMUP)
            self.throughput_history.append(throughput)
    
    def get_metrics(self):
        """Calculate final metrics"""
        
        travel_times = []
        delays = []
        
        for v, times in self.vehicle_times.items():
            if 'exit' in times:
                tt = times['exit'] - times['enter']
                travel_times.append(tt)
                delays.append(max(0, tt - times.get('free_flow_time', 144)))
        
        return {
            'avg_queue': np.mean(self.queues) if self.queues else 0,
            'avg_speed': np.mean(self.speeds) if self.speeds else 0,
            'avg_travel_time': np.mean(travel_times) if travel_times else 0,
            'avg_delay': np.mean(delays) if delays else 0,
            'throughput': np.mean(self.throughput_history) if self.throughput_history else 0
        }

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class SimulationRunner:
    """Run simulations for all patterns"""
    
    def __init__(self):
        self.traffic_gen = TrafficGenerator()
        self.results = {
            'baseline': {p: [] for p in config.HEADWAY_PATTERNS},
            'dqn': {p: [] for p in config.HEADWAY_PATTERNS}
        }
    
    def run_episode(self, controller, pattern, flows, episode_num):
        """Run one episode"""
        
        print(f"\n  Episode {episode_num} - Pattern: {pattern}")
        
        # Generate traffic
        seed = 1000 * episode_num + 42
        self.traffic_gen.generate_route_file(pattern, flows, seed)
        
        # Start SUMO
        sumo_cmd = ["sumo-gui" if config.SUMO_GUI else "sumo", 
                   "-c", config.SUMO_CFG, "--start"]
        traci.start(sumo_cmd)
        
        try:
            collector = DataCollector()
            
            step = 0
            while step < config.DURATION:
                traci.simulationStep()
                
                if step >= config.WARMUP:
                    collector.collect(step)
                
                if step % 5 == 0:
                    try:
                        controller.control_step(step)
                    except:
                        pass
                
                step += 1
            
            metrics = collector.get_metrics()
            print(f"    Queue: {metrics['avg_queue']:.3f}, "
                  f"Speed: {metrics['avg_speed']:.1f}, "
                  f"Delay: {metrics['avg_delay']:.1f}")
            
            return metrics
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
        finally:
            try:
                traci.close()
            except:
                pass
    
    def train_dqn(self, pattern, episodes=3):
        """Train DQN on a specific pattern"""
        print(f"\n{'='*60}")
        print(f"Training DQN on {pattern} pattern")
        print(f"{'='*60}")
        
        controller = DQNController()
        controller.set_training(True)
        
        flows = config.FLOWS[pattern]
        
        for ep in range(1, episodes + 1):
            self.run_episode(controller, pattern, flows, ep)
        
        return controller
    
    def evaluate_dqn(self, controller, pattern, episodes=3):
        """Evaluate DQN on a specific pattern"""
        print(f"\n{'='*60}")
        print(f"Evaluating DQN on {pattern} pattern")
        print(f"{'='*60}")
        
        controller.set_training(False)
        results = []
        
        flows = config.FLOWS[pattern]
        
        for ep in range(1, episodes + 1):
            metrics = self.run_episode(controller, pattern, flows, ep + 100)
            if metrics:
                results.append(metrics)
        
        return results
    
    def run_baseline(self, pattern, episodes=3):
        """Run baseline on a specific pattern"""
        print(f"\n{'='*60}")
        print(f"Running Baseline on {pattern} pattern")
        print(f"{'='*60}")
        
        controller = BaselineController()
        results = []
        
        flows = config.FLOWS[pattern]
        
        for ep in range(1, episodes + 1):
            metrics = self.run_episode(controller, pattern, flows, ep + 200)
            if metrics:
                results.append(metrics)
        
        return results
    
    def run_all_patterns(self):
        """Run experiments for all three patterns"""
        
        print("\n" + "="*80)
        print("DQN vs BASELINE COMPARISON - ALL THREE PATTERNS")
        print("="*80)
        
        all_comparisons = []
        
        for pattern in config.HEADWAY_PATTERNS:
            print(f"\n{'#'*60}")
            print(f"PATTERN: {pattern.upper()}")
            print(f"{'#'*60}")
            
            # Train DQN
            dqn_controller = self.train_dqn(pattern, episodes=3)
            
            # Evaluate DQN
            dqn_results = self.evaluate_dqn(dqn_controller, pattern, episodes=3)
            
            # Run baseline
            baseline_results = self.run_baseline(pattern, episodes=3)
            
            # Store results
            self.results['baseline'][pattern] = baseline_results
            self.results['dqn'][pattern] = dqn_results
            
            # Compare for this pattern
            if baseline_results and dqn_results:
                comparison = self.compare_pattern(baseline_results, dqn_results, pattern)
                all_comparisons.append(comparison)
        
        # Save all results
        self.save_all_results(all_comparisons)
    
    def compare_pattern(self, baseline, dqn, pattern):
        """Compare results for a single pattern"""
        
        b_avg = {k: np.mean([r[k] for r in baseline]) for k in baseline[0]}
        d_avg = {k: np.mean([r[k] for r in dqn]) for k in dqn[0]}
        
        print(f"\n📊 RESULTS FOR {pattern.upper()}")
        print("-"*50)
        print(f"{'Metric':<20} {'Baseline':<12} {'DQN':<12} {'Change':<12}")
        print("-"*50)
        
        metrics = [
            ('avg_queue', 'Queue Length', True),
            ('avg_speed', 'Speed (km/h)', False),
            ('avg_travel_time', 'Travel Time', True),
            ('avg_delay', 'Delay', True),
            ('throughput', 'Throughput', False)
        ]
        
        comparison = {
            'pattern': pattern,
            'metrics': {}
        }
        
        for key, name, lower_better in metrics:
            b = b_avg[key]
            d = d_avg[key]
            if lower_better:
                change = ((b - d) / b) * 100 if b > 0 else 0
            else:
                change = ((d - b) / b) * 100 if b > 0 else 0
            
            arrow = "✅" if change > 0 else "❌"
            print(f"{name:<20} {b:<12.3f} {d:<12.3f} {arrow} {change:>+5.1f}%")
            
            comparison['metrics'][key] = {
                'baseline': b,
                'dqn': d,
                'improvement': change
            }
        
        return comparison
    
    def save_all_results(self, comparisons):
        """Save all results to Excel"""
        
        # Create summary dataframe
        summary_data = []
        
        for comp in comparisons:
            pattern = comp['pattern']
            for metric, values in comp['metrics'].items():
                metric_name = {
                    'avg_queue': 'Queue Length',
                    'avg_speed': 'Speed',
                    'avg_travel_time': 'Travel Time',
                    'avg_delay': 'Delay',
                    'throughput': 'Throughput'
                }.get(metric, metric)
                
                summary_data.append({
                    'Pattern': pattern,
                    'Metric': metric_name,
                    'Baseline': round(values['baseline'], 3),
                    'DQN': round(values['dqn'], 3),
                    'Improvement %': round(values['improvement'], 1)
                })
        
        # Create detailed data
        detailed_data = []
        
        for pattern in config.HEADWAY_PATTERNS:
            for i, (b, d) in enumerate(zip(self.results['baseline'][pattern], 
                                          self.results['dqn'][pattern])):
                detailed_data.append({
                    'Pattern': pattern,
                    'Type': 'Baseline',
                    'Run': i+1,
                    'Queue': b['avg_queue'],
                    'Speed': b['avg_speed'],
                    'TravelTime': b['avg_travel_time'],
                    'Delay': b['avg_delay'],
                    'Throughput': b['throughput']
                })
                
                detailed_data.append({
                    'Pattern': pattern,
                    'Type': 'DQN',
                    'Run': i+1,
                    'Queue': d['avg_queue'],
                    'Speed': d['avg_speed'],
                    'TravelTime': d['avg_travel_time'],
                    'Delay': d['avg_delay'],
                    'Throughput': d['throughput']
                })
        
        # Save to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.OUTPUT_DIR}/dqn_comparison_all_patterns_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed', index=False)
            
            # Create pattern-specific sheets
            for pattern in config.HEADWAY_PATTERNS:
                pattern_data = [d for d in detailed_data if d['Pattern'] == pattern]
                if pattern_data:
                    pd.DataFrame(pattern_data).to_excel(writer, sheet_name=pattern.capitalize(), index=False)
        
        print(f"\n📁 All results saved to: {filename}")
        
        # Print overall summary
        self.print_overall_summary(comparisons)
    
    def print_overall_summary(self, comparisons):
        """Print overall summary across all patterns"""
        
        print("\n" + "="*80)
        print("OVERALL SUMMARY - ALL PATTERNS")
        print("="*80)
        
        for comp in comparisons:
            pattern = comp['pattern'].upper()
            print(f"\n{pattern}:")
            print("-"*40)
            
            metrics = comp['metrics']
            print(f"  Queue:     {metrics['avg_queue']['baseline']:.3f} → {metrics['avg_queue']['dqn']:.3f} "
                  f"({metrics['avg_queue']['improvement']:>+5.1f}%)")
            print(f"  Speed:     {metrics['avg_speed']['baseline']:.1f} → {metrics['avg_speed']['dqn']:.1f} "
                  f"({metrics['avg_speed']['improvement']:>+5.1f}%)")
            print(f"  Delay:     {metrics['avg_delay']['baseline']:.1f} → {metrics['avg_delay']['dqn']:.1f} "
                  f"({metrics['avg_delay']['improvement']:>+5.1f}%)")
            print(f"  Throughput: {metrics['throughput']['baseline']:.0f} → {metrics['throughput']['dqn']:.0f} "
                  f"({metrics['throughput']['improvement']:>+5.1f}%)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("DQN TRAFFIC SIGNAL CONTROL - ALL THREE HEADWAY PATTERNS")
    print("="*80)
    
    print(f"\nPatterns to test: {', '.join(config.HEADWAY_PATTERNS)}")
    print(f"Duration per episode: {config.DURATION/60:.0f} minutes")
    print(f"Training episodes per pattern: 3")
    print(f"Evaluation episodes per pattern: 3")
    
    # Create runner and run experiments
    runner = SimulationRunner()
    runner.run_all_patterns()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()