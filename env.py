import envlogger
import numpy as np
import time
from gym import spaces

class RoboticArmEnv:
    def __init__(self, xarm):
        self.xarm = xarm
        self.state = self.get_joint_angles()
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)

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
