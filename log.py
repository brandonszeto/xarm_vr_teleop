import tensorflow as tf
import rlds

class RLDSLogger:
    def __init__(self, log_dir='/rlds_log'):
        self.log_dir = log_dir
        self.writer = tf.data.experimental.TFRecordWriter(f'{log_dir}/data.tfrecord')
        self.episode_data = []
        self.current_episode = []
        
    def start_episode(self):
        self.current_episode = []

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

    def save(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.episode_data)
        self.writer.write(dataset)
