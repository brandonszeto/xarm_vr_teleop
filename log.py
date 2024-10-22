import json

def write_log(xarm, log_file):
    observation = {
        "servo_angles" : xarm.arm.get_servo_angles,
        "servo_velocities" : xarm.arm.get_servo_velocities
    }

    json.dump(observation, log_file)
    log_file.write("\n")
