import copy
import torch  # torch는 PyTorch라는 딥러닝 프레임워크의 핵심 모듈, 텐서 연산 빠르게 in GPU (vs numpy in only CPU)
import torch.nn as nn
import torch.optim as optim
from agents.policy_network import PolicyNetwork

class PPOAgent:
    def __init__(self, input_dim, hidden_dim, n_actions, lr, gamma, eps_clip):
        # 정책 네트워크와 학습 파라미터 초기화
        self.gamma = gamma  # gamma: 할인 계수
        self.eps_clip = eps_clip

        self.policy = PolicyNetwork(input_dim, hidden_dim, n_actions)  # model(x)처럼 실행하면 forward(x)가 자동으로 호출됨
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        #메모리 초기화
        self.reset_memory()

    def reset_memory(self):                       # rollout 메모리 초기화
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': []
        }

    def select_action(self, state, hidden=None):  # 현재 상태에서 행동 샘플링
        # state: (blood, symptom) 튜플
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        # torch.tensor(data)는 리스트나 배열을 PyTorch 텐서로 바꾸는 함수, 텐서는 스칼라, 벡터, 행렬, 그 이상 차원의 수 덩어리를 말함.
        # torch.long은 int64 정수 타입, LSTM 등에서 embedding layer를 쓸 경우 input이 정수여야 함.
        # .unsqueeze(0): 0번 위치에 차원 추가
        logits, value, hidden = self.policy(state_tensor, hidden)  # forward pass

        probs = torch.softmax(logits, dim=-1)  # dim=-1, 즉 마지막 차원을 대상으로 "확률 분포"로 변환함. ex) [2.0, 1.0, 0.1] -> [0.659, 0.242, 0.099]
        dist = torch.distributions.Categorical(probs)  # probs를 기반으로 "Categorical 분포 객체"를 생성
        action = dist.sample()  # 확률 분포에서 행동을 샘플링함 (탐험용)
        logprob = dist.log_prob(action)  # 행동의 로그 확률값 계산(log (prob))

        return action.item(), logprob.item(), value.item(), hidden  # .item(): 텐서 값을 Python의 기본 자료형으로 꺼내는 함수
    
    def store_transition(self, state, action, logprob, reward, done):  # 샘플한 transition 저장
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)

    def compute_returns(self, next_value):        # discounted return 계산
        # next_value: 마지막 시점의 value function 출력
        returns = []
        R = next_value
        for reward, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
        # done == True이면 R=0으로 초기화하며 G_t = r_t + γ G_{t+1}이므로, 뒤에서부터 계산
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float)  # 이후 학습 시 loss 계산 등에 사용됨(리스트 -> float 타입 텐서)
    
    def update(self, update_epochs):                             # PPO 알고리즘을 통한 policy/value 업데이트
        # 1. 메모리 -> 텐서화
        state_tensor = torch.tensor(self.memory['states'], dtype=torch.long).unsqueeze(1)  # [batch, 1, 2]
        action_tensor = torch.tensor(self.memory['actions'])  # [batch]
        old_logprobs = torch.tensor(self.memory['logprobs'])  # [batch]

        # 2. Discounted return 계산 (target fot value loss)
        returns = self.compute_returns(next_value=0)

        # 3. 현재 policy 복사 -> old_policy로 고정
        old_policy = copy.deepcopy(self.policy)
        with torch.no_grad():
            old_logits, _, _ = old_policy(state_tensor)
            old_probs = torch.softmax(old_logits, dim=-1)
            old_dist = torch.distributions.Categorical(old_probs)
            old_logprobs = old_dist.log_prob(action_tensor)
        # 4. 여러 epoch 동안 policy 업데이트
        for _ in range(update_epochs):
        # 5. 현재 policy의 확률 분포 및 logprob 계산
            logits, values, _ = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(action_tensor)
            entropy = dist.entropy()
        # 6. Advantage 계산
            advantages = returns - values.squeeze()
        # 7. ratio 및 clip된 surrogate objective
            ratio = torch.exp(new_logprobs - old_logprobs)
            clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        # 8. 손실 계산
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
            value_loss = nn.functional.mse_loss(values.squeeze(), returns)
            entropy_loss = -entropy.mean()
        # 9. 최종 손실 + 파라미터 업데이트
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 10. 메모리 초기화
        self.reset_memory()