# robot_runner.py
import time
import socket
import json
import math
from utils.vicon_connector import ViconStateProvider

class RobotRunner:
    """
    A wrapper to run any MPC test setup on the real RC car.
    Handles Vicon state updates and UDP command sending.
    """
    def __init__(self, pi_ip, port=12345, wheelbase=0.3):
        """
        Initializes the network connection and car parameters.
        :param pi_ip: The IP address of the Raspberry Pi.
        :param port: The port for UDP communication.
        :param wheelbase: The wheelbase of the car in meters (for command translation).
        """
        self.pi_ip = pi_ip
        self.port = port
        self.wheelbase = wheelbase
        self.vicon = ViconStateProvider()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"RobotRunner initialized. Will send commands to {pi_ip}:{port}")

    def _translate_commands(self, mpc_output, vehicle_model):
        """
        Translates abstract MPC commands (a, w) into concrete hardware commands.
        :param mpc_output: The solution object from planner.solve_mpc().
        :return: A dictionary with 'throttle' and 'steering' values.
        """
        # Extract the FIRST control input and the SECOND predicted state
        next_w = mpc_output.control_history[0][1]
        predicted_v = mpc_output.trajectory_history[-1].get_states()[1].get("v")

        # 1. Translate velocity to throttle
        # Get velocity bounds from the vehicle model for robust mapping
        v_min, v_max, _ = vehicle_model.get_bounds("v")
        throttle_command = predicted_v / v_max if predicted_v > 0 else predicted_v / abs(v_min)
        throttle_command = max(-1.0, min(1.0, throttle_command)) # Clamp value

        # 2. Translate angular velocity (w) to steering angle
        if abs(predicted_v) < 0.1:
            physical_steering_rad = 0.0
        else:
            # Formula: delta = atan((w * L) / v)
            max_angle_rad = 0.55  # Max physical steering angle (tune this)
            arg = (next_w * self.wheelbase) / predicted_v
            arg_clamped = max(-math.tan(max_angle_rad), min(math.tan(max_angle_rad), arg))
            physical_steering_rad = math.atan(arg_clamped)

        # Map physical steering angle (-max_angle_rad to +max_angle_rad) to servo angle (0-180)
        # Assuming 90 is center. You may need to fine-tune these values.
        servo_center = 90
        steering_gain = servo_center / max_angle_rad
        servo_angle_command = servo_center - physical_steering_rad * steering_gain

        return {"throttle": throttle_command, "steering": servo_angle_command}

    def run_on_robot(self, setup_function, **setup_kwargs):
        """
        The main execution loop.
        :param setup_function: A function from a test file that sets up and returns (planner, data).
        :param setup_kwargs: A dictionary of arguments to pass to the setup_function (e.g., goal=(x,y)).
        """
        # --- 1. SETUP THE MPC PROBLEM using the provided function ---
        print("Running the provided MPC setup function...")
        planner, data, vehicle_model = setup_function(**setup_kwargs)
        dt = planner.solver.timestep
        print("MPC setup complete.")

        print("Press Enter to begin MPC control loop...")
        input()

        # --- 2. REAL-TIME CONTROL LOOP ---
        try:
            while True:
                loop_start_time = time.time()

                # --- SENSE: Get current state from Vicon ---
                vicon_state = self.vicon.get_current_state()
                print(f"State: x={vicon_state['x']:.2f}, y={vicon_state['y']:.2f}, psi={vicon_state['psi']:.2f}, v={vicon_state['v']:.2f}")

                # Update the planner's state
                current_mpc_state = planner.get_state()
                current_mpc_state.set("x", vicon_state['x'])
                current_mpc_state.set("y", vicon_state['y'])
                current_mpc_state.set("psi", vicon_state['psi'])
                current_mpc_state.set("v", vicon_state['v'])
                # You may need a better way to estimate 'spline' state
                planner.set_state(current_mpc_state)

                # --- PLAN: Solve the MPC problem ---
                output = planner.solve_mpc(data)

                # --- ACT: Translate and send commands ---
                if output.success:
                    command_data = self._translate_commands(output, vehicle_model)
                    print(f"MPC OK -> Sending: {command_data}")
                else:
                    print("MPC failed! Sending STOP command.")
                    command_data = {"throttle": 0.0, "steering": 90}

                message = json.dumps(command_data).encode('utf-8')
                self.client_socket.sendto(message, (self.pi_ip, self.port))

                # Maintain loop frequency and reset solver
                elapsed_time = time.time() - loop_start_time
                time.sleep(max(0, dt - elapsed_time))
                planner.solver.reset()

        except KeyboardInterrupt:
            print("\nControl loop interrupted. Sending final STOP command.")
        finally:
            stop_command = {"throttle": 0.0, "steering": 90}
            message = json.dumps(stop_command).encode('utf-8')
            self.client_socket.sendto(message, (self.pi_ip, self.port))
            self.client_socket.close()
            print("Client socket closed.")