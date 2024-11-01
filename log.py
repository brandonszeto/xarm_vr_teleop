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
        for step in self.current_episode:
            example = self._create_tf_example(step)
            self.writer.write(example.SerializeToString())
        print("Ending episode (log.py)")

    def save(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.episode_data)
        self.writer.write(dataset)
        print(f"Data saved to {self.log_path}")

    def _create_tf_example(self, step):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]))
        
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        features = {
            "observation": _bytes_feature(step["observation"]),
            "action": _bytes_feature(step["action"]),
            "reward": _float_feature(step["reward"]),
            "discount": _float_feature(step["discount"])
        }

        if "step_metadata" in step:
            features["step_metadata"] = _bytes_feature(step["step_metadata"])

        return tf.train.Example(features=tf.train.Features(feature=features))
