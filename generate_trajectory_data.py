import numpy as np

def generate_trajectory(speed, steering_angle):
  # Initialize state variables
  x = 0.0  # initial x position
  y = 0.0  # initial y position
  theta = 0.0  # initial heading direction (facing along the x-axis)

  L = 2.5  # wheelbase of the vehicle (distance between front and rear axle)

  # Simulation settings
  dt = 0.1  # time step in seconds
  total_time = 2
  # Number of timesteps (excluding the last one for consistency)
  num_steps = int(total_time / dt)

  # Initialize a single NumPy array to store (x, y) pairs
  trajectory = np.zeros((num_steps, 2))

  thet = np.deg2rad(steering_angle)

  # Simulation loop
  for i in range(num_steps):
    # Update the vehicle's state
    x += speed * np.cos(theta) * dt
    y += speed * np.sin(theta) * dt
    theta += speed * np.tan(thet) / L * dt

    # Store positions in the trajectory array
    trajectory[i, 0] = x
    trajectory[i, 1] = y

  return trajectory

def generate_data(speed):
    angles = np.arange(-10, 10, 4)
    all_trajs = []
    for angle in angles:
        trajectory = generate_trajectory(speed,angle)
        all_trajs.append(trajectory)
    return speed, all_trajs

def save_trajectories(x = 0, y = 50, z = 1):
    speeds = np.arange(x, y , z)
    all_data = {"state" : [], "trajectory" : []}
    for current_speed in speeds:
        speed, all_trajs = generate_data(current_speed)
        state = np.array([0,0,current_speed,1])
        all_data["state"].append(state)
        all_data["trajectory"].append(all_trajs)
    return all_data

def process_data(data):
    states = []
    trajectories = []

    for state, trajs in zip(data["state"], data["trajectory"]):
        states.append(state)
        trajectories.append(trajs)
    
    return np.array(states), np.array(trajectories)