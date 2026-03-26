import torch
import minari
from utils.buffer import OfflineReplayBuffer


def test_offline_buffer():
    dataset_name = "mujoco/walker2d/expert-v0"

    if dataset_name not in minari.list_local_datasets():
        print(f"Downloading dataset {dataset_name}...")
        minari.download_dataset(dataset_name)

    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Total Steps
    total_steps = dataset.total_steps
    print(f"Total steps in dataset: {total_steps}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init Buffer and load dataset
    buffer = OfflineReplayBuffer(device, state_dim, action_dim, max_size=total_steps)
    buffer.load_dataset(dataset_name)

    # post process...
    buffer.compute_return(gamma=0.99)
    mean, std = buffer.normalize_state()
    buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")

    s, a, r, s_next, done = buffer.sample(batch_size=256)
    print(f"Sampled batch shapes -> s: {s.shape}, a: {a.shape}")
    print("P1 Test Passed!")


if __name__ == "__main__":
    test_offline_buffer()
