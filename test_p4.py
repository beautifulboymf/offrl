import torch
import minari
import numpy as np
from tqdm import tqdm

from utils.buffer import OfflineReplayBuffer
from agents.critic import IQL_Q_V
from agents.bc_ensemble import BC_Ensemble
from agents.abppo import AdaptiveBehaviorPPO

def mock_dynamics_eval(bppo, iql_q, buffer):
    """only for debugging ABPO """
    return np.random.uniform(50, 100)

def test_abppo_pipeline():
    dataset_name = "mujoco/walker2d/expert-v0"
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. prepare data
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    buffer.load_dataset(dataset_name)
    buffer.normalize_state()
    buffer.compute_return(gamma=0.99)
    buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")

    # 2. IQL pretrain (Mock, we only train 500 steps)
    print("--- 1. Training IQL (Mock 500 steps) ---")
    iql = IQL_Q_V(device, state_dim, action_dim, 1024, 2, 1e-4, 2, 0.005, 0.99, 256, 256, 3, 1e-4, 0.7, True)
    for _ in tqdm(range(500)): iql.update(buffer)

    # 3. BC Ensemble pretrain (Mock 500 steps)
    print("--- 2. Training BC Ensemble (Mock 500 steps) ---")
    bc_ensemble = BC_Ensemble(num_policies=4, state_dim=state_dim, action_dim=action_dim, device=device)
    for _ in tqdm(range(500)): bc_ensemble.joint_train(buffer, 256, alpha=0.1)

    # 4. ABPO offline ft (P4 core)
    print("--- 3. Running ABPO Offline Optimization ---")
    # clip_ratio=0.25, lr=1e-4, omega=0.7
    abppo = AdaptiveBehaviorPPO(num_policies=4, state_dim=state_dim, action_dim=action_dim, 
                                device=device, lr=1e-4, clip_ratio=0.25, omega=0.7)
    
    # init abpo param from bc
    abppo.load_bc_weights(bc_ensemble)

    total_bppo_steps = 2000 # origin: 10000 steps
    eval_step = 100         # OPE evaluate for each 100 steps
    best_mean_qs = np.zeros(4)

    pbar = tqdm(range(total_bppo_steps), desc="ABPO Training")
    for step in pbar:
        # Clip Ratio
        current_clip = 0.25 * (1 - step / total_bppo_steps)
        
        losses = abppo.joint_train(buffer, 256, iql, current_clip)
        
        # AM-Q OPE
        if (step + 1) % eval_step == 0:
            current_qs = np.zeros(4)
            for i in range(4):
                # current_qs[i] = dynamics_eval(args, abppo.ensemble[i], iql.minQ, dynamics, eval_buffer, ...)
                current_qs[i] = mock_dynamics_eval(abppo.ensemble[i], iql.minQ, buffer) 
                
            improve_indices = np.where(current_qs > best_mean_qs)[0]
            if len(improve_indices) > 0:
                # (old_policy <- policy)
                best_mean_qs[improve_indices] = current_qs[improve_indices]
                abppo.replace_policies(improve_indices)
                pbar.write(f"Step {step+1}: Policies {improve_indices} improved and replaced! Best Qs: {best_mean_qs}")

    print("P4 Test Finished! ABPO multi-step optimization logic runs successfully.")

if __name__ == "__main__":
    test_abppo_pipeline()