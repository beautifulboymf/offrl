import torch
import minari
import numpy as np
from tqdm import tqdm
from utils.buffer import OfflineReplayBuffer
from agents.critic import IQL_Q_V


def test_iql_value_correlation():
    dataset_name = "mujoco/walker2d/medium-v0"
    print(f"Loading {dataset_name}...")
    dataset = minari.load_dataset(dataset_name, download=True)
    env = dataset.recover_environment()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 Buffer (复用 P1 逻辑)
    buffer = OfflineReplayBuffer(
        device, state_dim, action_dim, max_size=dataset.total_steps
    )
    buffer.load_dataset(dataset_name)
    buffer.compute_return(gamma=0.99)
    buffer.normalize_state()
    buffer.reward_normalize(gamma=0.99, scale_strategy="dynamic")

    # 初始化 IQL 网络 (参数对齐 Uni-O4 Table 5)
    iql = IQL_Q_V(
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
        q_hidden_dim=1024,
        q_depth=2,
        Q_lr=1e-4,
        target_update_freq=2,
        tau=0.005,
        gamma=0.99,
        batch_size=256,
        v_hidden_dim=256,
        v_depth=3,
        v_lr=1e-4,
        omega=0.7,
        is_double_q=True,
    )

    # 训练 50000 步用来验证有效性 (完整训练需 ~2M 步)
    train_steps = 500000
    pbar = tqdm(range(train_steps), desc="Training IQL")
    for step in pbar:
        q_loss, v_loss = iql.update(buffer)
        if step % 1000 == 0:
            pbar.set_postfix({"Q_loss": q_loss, "V_loss": v_loss})

    # ====== 评估 V(s) 的相关性 ======
    print("\n--- Evaluating V(s) Correlation ---")
    iql._value.eval()

    # 从 Buffer 中随机抽取 5000 条数据评估
    test_batch_size = 5000
    ind = np.random.randint(0, buffer.size, size=test_batch_size)
    s_test = torch.FloatTensor(buffer.s[ind]).to(device)
    real_returns = buffer.returns[ind].flatten()

    with torch.no_grad():
        pred_v = iql._value(s_test).cpu().numpy().flatten()

    # 计算皮尔逊相关系数
    correlation_matrix = np.corrcoef(pred_v, real_returns)
    pearson_r = correlation_matrix[0, 1]

    print(
        f"Mean Predicted V: {pred_v.mean():.4f} | Mean Real Return (Scaled): {real_returns.mean():.4f}"
    )
    print(f"Pearson Correlation Coefficient (r): {pearson_r:.4f}")

    if pearson_r > 0.4:
        print("P2 Test Passed! V(s) successfully learned the return distribution.")
    else:
        print("Warning: Correlation is weak. May need more training steps.")


if __name__ == "__main__":
    test_iql_value_correlation()
