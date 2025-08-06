import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import numpy as np
from envs.medical_pomdp_gym import MedicalPOMDPEnv
from agents.ppo_agent import PPOAgent
import matplotlib.pyplot as plt

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 하이퍼파라미터
EPISODES = config['training']['episodes']
UPDATE_EPOCHS = config['training']['update_epochs']
BATCH_SIZE = config['training']['batch_size']
HIDDEN_DIM = config['agent']['hidden_dim']
INPUT_DIM = config['agent']['input_dim']
N_ACTIONS = config['agent']['n_actions']
LR = config['agent']['lr']
GAMMA = config['agent']['gamma']
EPS_CLIP = config['agent']['eps_clip']

# 로그 저장용
reward_history = []

# 환경, 에이전트 초기화
env = MedicalPOMDPEnv()
agent = PPOAgent(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS, lr=LR, gamma=GAMMA, eps_clip=EPS_CLIP)

# 학습 루프
for episode in range(EPISODES):
    state = env.reset()
    done = False
    ep_reward = 0
    hidden = None

    while not done:
        action, logprob, value, hidden = agent.select_action(state, hidden)
        next_state, reward, done, _ = env.step(action)

        agent.store_transition(state, action, logprob, reward, done)
        state = next_state
        ep_reward += reward

    reward_history.append(ep_reward)

    # 일정 주기마다 업데이트
    if (episode + 1) % BATCH_SIZE == 0:
        agent.update(update_epochs=UPDATE_EPOCHS)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1} | Reward: {ep_reward:.2f}")

# 결과 저장
os.makedirs("results/logs", exist_ok=True)
np.save("results/logs/ppo_rewards.npy", reward_history)

# 시각화
os.makedirs("results/plots", exist_ok=True)
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Training Reward Curve")
plt.savefig("results/plots/ppo_reward_curve.png")
plt.close()