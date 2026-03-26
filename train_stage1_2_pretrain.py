import os
import torch
import numpy as np
from tqdm import tqdm
from utils.buffer import OfflineReplayBuffer
from agents.critic import IQL_Q_V
from agents.bc_ensemble import BC_Ensemble

def run_pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "mujoco/walker2d/medium-v0" # Consistent with dynamics model
    
    # 0. Prepare save directories
    os.makedirs("checkpoints/stage1", exist_ok=True)
    os.makedirs("checkpoints/stage2", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    
    # 1. Initialize and load Replay Buffer
    state_dim = 17 # Walker2d observation dimension
    action_dim = 6 # Walker2d action dimension
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=1000000)
    buffer.load_dataset(env_name)
    
    # Compute Return-to-go (required for IQL) and Normalize Rewards
    buffer.compute_return(gamma=0.99)
    buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")
    
    # Standardize States and save statistics persistently
    stats_path = f"stats/{env_name.replace('/','_')}_stats.npz"
    mean, std = buffer.normalize_state() #
    np.savez(stats_path, mean=mean, std=std)
    print(f"State statistics saved to {stats_path}")

    # ==========================================
    # Stage 1: IQL Pre-training
    # ==========================================
    print("\n Starting Stage 1: IQL Value Function Pre-training")
    iql = IQL_Q_V(
        device=device, state_dim=state_dim, action_dim=action_dim,
        q_hidden_dim=1024, q_depth=2, Q_lr=1e-4, target_update_freq=2, tau=0.005,
        gamma=0.99, batch_size=256, v_hidden_dim=256, v_depth=3, v_lr=1e-4, 
        omega=0.7, is_double_q=True
    )
    
    iql_q_path = "checkpoints/stage1/iql_q.pt"
    iql_v_path = "checkpoints/stage1/iql_v.pt"
    
    # Check if IQL parameters already exist
    if os.path.exists(iql_q_path) and os.path.exists(iql_v_path):
        print(f"Found existing IQL weights. Loading and skipping Stage 1 training.")
        iql.load(iql_q_path, iql_v_path) #
    else:
        iql_steps = 500000 # Reduced for testing; paper usually uses 1M-2M
        for step in tqdm(range(iql_steps), desc="IQL Training"):
            q_loss, v_loss = iql.update(buffer)
            if step % 100000 == 0:
                print(f"Step {step}: Q Loss = {q_loss:.4e}, V Loss = {v_loss:.4e}")
                
        # Save IQL model parameters
        torch.save(iql._Q.state_dict(), iql_q_path)
        torch.save(iql._value.state_dict(), iql_v_path)
        print("Stage 1 completed and saved!")

    # ==========================================
    # Stage 2: BC Ensemble Pre-training
    # ==========================================
    print("\n Starting Stage 2: BC Ensemble Pre-training")
    num_policies = 5
    bc = BC_Ensemble(
        num_policies=num_policies, state_dim=state_dim, action_dim=action_dim, 
        device=device, lr=3e-4
    ) #
    
    bc_checkpoints = [f"checkpoints/stage2/bc_policy_{i}.pt" for i in range(num_policies)]
    all_bc_exist = all(os.path.exists(path) for path in bc_checkpoints)
    
    # Check if BC Ensemble parameters already exist
    if all_bc_exist:
        print(f"Found all {num_policies} BC policy weights. Loading and skipping Stage 2.")
        for i in range(num_policies):
            bc.ensemble[i].load_state_dict(torch.load(bc_checkpoints[i], map_location=device))
    else:
        bc_steps = 200000 # Reduced for testing; paper usually uses 400k
        alpha_penalty = 0.1 # Disagreement penalty coefficient
        
        for step in tqdm(range(bc_steps), desc="BC Training"):
            bc_losses = bc.joint_train(buffer, batch_size=256, alpha=alpha_penalty)
            if step % 50000 == 0:
                print(f"Step {step}: Avg BC Loss = {bc_losses.mean():.4f}")
                
        # Save BC policy parameters
        for i in range(num_policies):
            torch.save(bc.ensemble[i].state_dict(), bc_checkpoints[i])
        print("Stage 2 completed and saved!")

if __name__ == "__main__":
    run_pretrain()