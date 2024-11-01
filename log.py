import envlogger
import time
from env import RoboticArmEnv

def start_logging(xarm):
    log_path = './rlds_log'

    # Wrap environment in envlogger
    logged_env = envlogger.EnvLogger(RoboticArmEnv(xarm), data_directory=log_path)

    try:
        observation = logged_env.reset()
        done = False

        while True:
            action = None 
            observation, reward, done, info = logged_env.step(action)

            # time.sleep(0.1)

            if done:
                observation = logged_env.reset()

    except KeyboardInterrupt:
        print("Data collection interrupted, closing the logger.")
    finally:
        logged_env.close()
