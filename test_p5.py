import numpy as np
import torch
import gymnasium as gym
from utils.buffer import OfflineReplayBuffer
from dynamics_eval import train_dynamics

# Simulating the parameter configuration in Uni-O4 main.py
class SimpleArgs:
    def __init__(self):
        # --- Basic and Path Configurations ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = "mujoco/walker2d/expert-v0" 
        self.seed = 8
        self.algo_name = "mobile"       # Affects log folder naming
        self.is_eval_state_norm = False # Affects model path generation logic
        
        # --- Dynamics Model Architecture Parameters ---
        self.dynamics_hidden_dims = [200, 200, 200, 200] 
        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4] # 5 values
        self.dynamics_lr = 1e-3
        self.n_ensemble = 7             # Aligned with line 91 in dynamics_eval.py
        self.n_elites = 5
        self.model_type = 'probabilistic'

        # --- Core Training Control Parameters (Fixes for recent errors) ---
        self.dynamics_max_epochs = 100  # Resolves the latest error
        self.max_epochs_since_update = 5 # Resolves the previous error, early stopping threshold
        self.dynamics_steps = 100       # Your test step count
        self.rollout_batch_size = 512
        self.dynamics_update_freq = 1
        
        # --- Redundant Parameters for Logging System (KeyError prevention) ---
        # Logger.make_log_dirs iterates through these keys to generate experiment identifiers
        self.penalty_coef = 1.0
        self.rollout_length = 1000
        self.real_ratio = 0.05
        self.gamma = 0.99
        self.scale_strategy = 'dynamic'

def test_dynamics_learning():
    args = SimpleArgs()
    env_name = "mujoco/walker2d/expert-v0" # Or your local existing dataset
    gym_env_name = "Walker2d-v5"
    
    print(f"--- Starting Dynamics Model Test: Environment {env_name} ---")
    
    # 1. Load dataset into your OfflineReplayBuffer
    env = gym.make(gym_env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Assuming your Buffer already handles Minari loading
    buffer = OfflineReplayBuffer(args.device, state_dim, action_dim, max_size=1000000)
    # Note: Ensure your OfflineReplayBuffer has a load_dataset method
    # and has performed reward_normalize as required by Uni-O4
    buffer.load_dataset(env_name) 
    
    # 2. State Normalization (Dynamics models are very sensitive to this)
    # Uni-O4 must calculate the mean and variance of offline data before training the dynamics model
    mean, std = buffer.normalize_state()
    print(f"Data loading complete. State mean sample: {mean[:3]}, Std sample: {std[:3]}")

    # 3. Train Dynamics Model
    # Call the wrapper function in the ported dynamics_eval.py
    print("Training Dynamics Ensemble model...")
    dynamics_model = train_dynamics(args, env, buffer)
    
    # 4. One-step Validation
    print("\n--- Validating Prediction Accuracy ---")
    # Sample a set of real transitions (s, a, r, s_p, d) from the Buffer
    s_tensor, a_tensor, r_tensor, s_p_tensor, d_tensor = buffer.sample(batch_size=5)
    
    s = s_tensor.cpu().numpy()
    a = a_tensor.cpu().numpy()
    r = r_tensor.cpu().numpy()
    s_p = s_p_tensor.cpu().numpy()

    with torch.no_grad():
        # Passing NumPy arrays now to meet interface requirements
        pred_next_obs, pred_reward, _, _ = dynamics_model.step(s, a)
        
    # Calculate error (Note: If returning NumPy, use direct subtraction)
    obs_error = np.mean((pred_next_obs - s_p)**2)
    rew_error = np.mean((pred_reward - r)**2)
    
    print(f"One-step State Prediction MSE: {obs_error:.6f}")
    print(f"One-step Reward Prediction MSE: {rew_error:.6f}")
    
    if obs_error < 1e-3:
        print("Result: Dynamics model learning looks good; ready for policy evaluation.")
    else:
        print("Result: Error is too high; please check data normalization or model configuration.")

if __name__ == "__main__":
    test_dynamics_learning()