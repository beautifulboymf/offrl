import torch
import minari
import os
import numpy as np
from utils.buffer import OfflineReplayBuffer

# Import the official dynamics training interface (Ensure transition_model folder and dynamics_eval.py are in the root directory)
from dynamics_eval import train_dynamics

class DynamicsArgs:
    """
    Aligns with Uni-O4 official hyperparameters.
    Resolves KeyError in the Logger system and JSON serialization issues.
    """
    def __init__(self, full_env_name, device):
        # --- Core Environment Adaptation ---
        self.env = full_env_name.split('/')[1] if '/' in full_env_name else full_env_name
        self.algo_name = "mobile"       # Affects log and model saving paths
        self.seed = 8                   # Recommended to align with previous experiments
        self.device = str(device)
        
        # --- State Normalization Logic Alignment ---
        # During dynamics model training, the eval_buffer usually does not undergo external normalization,
        # because the model calls StandardScaler internally and termination functions depend on raw coordinates.
        self.is_state_norm = False
        self.is_eval_state_norm = False 
        
        # --- Dynamics Model Architecture Alignment ---
        # Standard configuration for MuJoCo tasks: 200-200-200-200
        self.dynamics_hidden_dims = [200, 200, 200, 200] 
        self.dynamics_weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
        self.dynamics_lr = 1e-3
        self.n_ensemble = 7             # 7 parallel probabilistic networks
        self.n_elites = 5               # Select the 5 with the lowest prediction error
        self.model_type = 'probabilistic'

        # --- Training Control ---
        self.dynamics_max_epochs = 1000  # Use None or a high value paired with early stopping
        self.max_epochs_since_update = 5 # Stop if no improvement for 5 epochs
        self.rollout_batch_size = 512    # Official default batch size

        # --- Redundant Parameter Safeguards (Aligns with Logger.make_log_dirs) ---
        # These parameters are not essential for training logic, but the Logger system will throw a KeyError if they are missing.
        self.penalty_coef = 1.0
        self.rollout_length = 1000       # Important: Default value of H for AM-Q offline evaluation
        self.real_ratio = 0.05
        self.gamma = 0.99
        self.scale_strategy = 'dynamic'

def main():
    # Recommended to use 'medium' datasets with diverse actions to better train the dynamics
    dataset_name = "mujoco/walker2d/medium-v0" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Minari dataset: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 1. Initialize a clean Buffer (Adapts to the sample_all interface of the dynamics model)
    print("Preparing Clean Buffer for Dynamics (Raw States)...")
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    buffer.load_dataset(dataset_name)
    
    # Note: Do NOT call buffer.normalize_state() here.
    # Otherwise, height checks during Rollouts will fail, resulting in extremely short AM-Q evaluations.
    
    # 2. Configure arguments
    args = DynamicsArgs(dataset_name, device)
    
    print(f"\nStarting Stage 3: Ensemble Dynamics Model Training for '{args.env}'")
    print("Training 7 Probabilistic Neural Networks with Early Stopping.")
    
    # 3. Launch official training procedure
    # train_dynamics will automatically create the 1saved_models_False/ directory and save .pt files
    dynamics = train_dynamics(args, env, buffer)
    
    print("\nStage 3 completed! Dynamics model is stored and ready for AM-Q evaluation in P4.")

if __name__ == "__main__":
    main()