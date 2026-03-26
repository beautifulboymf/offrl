import torch
import minari
import numpy as np
from tqdm import tqdm
from utils.buffer import OfflineReplayBuffer
from agents.bc_ensemble import BC_Ensemble

def test_bc_diversity():
    # 1. load dataset
    dataset_name = "mujoco/walker2d/medium-v0"
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=dataset.total_steps)
    buffer.load_dataset(dataset_name)
    buffer.normalize_state()

    # 2. init ensemble BC (4 Actor)
    alpha=0.1  # alpha = 0.2, larger alpha stands for larger similarity penalty
    bc_ensemble = BC_Ensemble(
        num_policies=4, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        lr=1e-4
    )

    # 3. Train for some steps
    train_steps = 10000
    pbar = tqdm(range(train_steps), desc="Training Ensemble BC")
    for _ in pbar:
        losses = bc_ensemble.joint_train(buffer, batch_size=256, alpha=alpha)
        if _ % 1000 == 0:
            pbar.set_postfix({"Avg_Loss": losses.mean()})

    # 4. verify diversity
    print("\n--- Verifying Action Diversity ---")
    # random select 5 states
    s_batch, _, _, _, _ = buffer.sample(5)
    
    for i in range(5):
        s_single = s_batch[i:i+1]
        # acquire mean action of 4 Actor 
        actions = bc_ensemble.get_ensemble_actions(s_single).squeeze() # [4, action_dim]
        
        # calculate Actor std (across the 4 members)
        action_std = np.std(actions, axis=0).mean()
        
        print(f"State {i+1} - Action Mean Std across Ensemble: {action_std:.6f}")
        if action_std > 1e-4:
            print(f"  Result: Diverse actions detected.")
        else:
            print(f"  Warning: Actions are too identical. Check alpha or training.")

    print("\nP3 Test Finished!")

if __name__ == "__main__":
    test_bc_diversity()