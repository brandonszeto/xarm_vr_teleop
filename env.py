import envlogger
import numpy as np
import time
from dm_env import specs

class RoboticArmEnv:
    def __init__(self, xarm):
        self.xarm = xarm
        self.state = self.get_joint_angles()
        self.observation_space = specs.Array(shape=(6,), dtype=np.float32,
                                             name='observation')
        self.action_space = specs.Array(shape=(6,), dtype=np.float32,
                                             name='action')
        self.reward_space = specs.Array(shape=(), dtype=np.float32,
                                             name='reward')
        self.discount_space = specs.BoundedArray(shape=(), dtype=np.float32,
                                             minimum=0.0, maximum=1.0,
                                                 name='discount')

    def reset(self):
        self.state = self.get_joint_angles()
        return self.state

    def step(self):
        self.state = self.get_joint_angles()

        # For simplicity right now
        reward = self.state
        done = self.state

        return self.state, reward, done, {}

    def get_joint_angles(self):
        joint_states, _ = self.xarm.arm.get_joint_states(is_radian=True)
        return joint_states
    
    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def discount_spec(self):
        return self.discount_space

    def reward_spec(self):
        return self.reward_space
