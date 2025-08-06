import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):  # nn.Model = 모델을 구성하는 기본 단위 __init__()과 forward()로 구성
    def __init__(self, input_dim, hidden_dim, action_dim):  # 모델의 구성요소들을(부품들을) 준비
        super().__init__()
        self.embedding = nn.Embedding(3, 4)
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):  # 입력 데이터를 처리하는 방법을 정의
        # x: (batch, seq_len, 2) -> blood, symptom
        b, t, f = x.shape
        x = self.embedding(x)          # (b, t, f, emb)
        x = x.view(b, t, -1)           # flatten features
        out, hidden = self.lstm(x, hidden)
        logits = self.policy_head(out[:, -1])  # 마지막 타임스텝의 output
        value = self.value_head(out[:, -1])
        return logits, value, hidden