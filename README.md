# Multi-Trajectory Prediction

This repository contains code for multi-trajectory prediction, focusing on generating and predicting vehicle trajectories using various speeds and steering angles.

## Repository Structure

- **`generate_trajectory_data.py`**: Contains functions to create and save vehicle trajectory data by simulating various driving conditions.
- **`dataloader.py`**: Provides the data loading utilities for feeding the generated trajectory data into the model during training.
- **`model.py`**: Defines the model architecture used for predicting multiple trajectories.
- **`loss.py`**: Implements the loss functions used to train the model, handling the multimodal nature of trajectory prediction.
- **`utils.py`**: Contains utility functions, such as plotting the predicted and actual trajectories for visualization.
- **`train.py`**: Implements the training loop
- **`ADAPT.ipynb`**: Jupyter notebook that contains the complete code, integrating data generation, model training, and evaluation.


## Setup

### Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/kap2403/multi-trajectory-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd multi-trajectory-prediction
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   *(Note: Create a `requirements.txt` file if not present, to list all dependencies.)*

## Data Generation

The data for trajectory prediction is generated using vehicle kinematics to simulate various possible paths based on different speeds and steering angles:

1. **Vehicle Kinematics**: The vehicle's motion is modeled using kinematic equations that update the vehicle's position `(x, y)` and orientation (`theta)` at each time step. This simulates realistic vehicle trajectories.
   
2. **Simulation Settings**:
   - **Wheelbase (`L`)**: The distance between the front and rear axles, affecting the vehicle's turning radius.
   - **Time Step (`dt`)**: The interval at which the vehicle's state is updated (e.g., `0.1` seconds).
   - **Total Time**: The total duration for which the trajectory is simulated (e.g., `2` seconds).

3. **Generating Trajectories**:
   - The `generate_trajectory` function simulates the vehicle's trajectory for a given speed and steering angle. It iteratively updates the vehicle's state using equations like:
     ```python
     x += speed * np.cos(theta) * dt
     y += speed * np.sin(theta) * dt
     theta += (speed * np.tan(steering_angle) / L) * dt
     ```
   - By varying speeds and steering angles, multiple possible future trajectories are generated.
   
4. **Multiple Trajectories**: By varying the steering angles (e.g., -10° to 10°) and speeds, the process generates a dataset of various possible trajectories, simulating different possible outcomes of the vehicle's movement.

### Usage
To generate and save trajectories, use the `generate_trajectory.py` script:
```python
from data.generate_trajectory import save_trajectories

# Example usage
output_data = save_trajectories(1, 40, 0.1)
