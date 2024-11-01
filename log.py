import tensorflow as tf
import rlds
import os

class RLDSLogger:
    def __init__(self, log_dir='./rlds_log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'data.tfrecord')
        self.writer = tf.data.experimental.TFRecordWriter(self.log_path)
        self.episode_data = []
        self.current_episode = []
        
    def start_episode(self):
        self.current_episode = []
        print("Starting episode (log.py)")

    def log_step(self, xarm, discount=1.0, step_metadata=None):
        joint_states = xarm.arm.get_joint_states(is_radian=True)
        joint_ang, joint_vel, joint_eff = joint_states[1]

        observation = {
            "servo_angle" : joint_ang,
            "servo_velocity" : joint_vel,
            "servo_effort" : joint_eff
        }

        action = 0

        reward = 0

        step_data = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "discount": discount,
            "step_metadata": step_metadata if step_metadata else {}
        }
        self.current_episode.append(step_data)

    def end_episode(self):
        episode = rlds.from_list(self.current_episode)
        self.episode_data.append(episode)
        print("Ending episode (log.py)")

    def save(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.episode_data)
        self.writer.write(dataset)
        print(f"Data saved to {self.log_path}")
