# Medical Treatment RL

강화학습(PPO)을 활용한 POMDP 기반 의료 순차 의사결정 프로젝트

## 🔍 프로젝트 개요

- **목표**: 의료 상황에서 검진/치료/대기 등의 순차적 의사결정을 통해 최적의 치료 정책을 학습
- **환경**: POMDP 환경 설계 (부분 관측 상태)
- **관측값**: 혈액검사 결과, 증상 → 임베딩하여 RNN에 입력
- **행동**: 치료, 검사, 대기 등
- **보상**: 건강 상태 향상 → 높은 보상 / 부적절한 행동 → 벌점

## 🛠️ 사용 알고리즘

- **PPO (Proximal Policy Optimization)**
- Advantage 계산
- Clipped Surrogate Objective
- LSTM 기반 정책 네트워크

## 📁 폴더 구조
Medical-Treatment-RL/
├── envs/ # POMDP 환경 정의
│ ├── medical_pomdp_simulator.py
│ └── medical_pomdp_gym.py
├── agents/ # 에이전트 및 정책 네트워크
│ ├── ppo_agent.py
│ └── policy_network.py
├── train/ # 학습 루프
│ └── train_ppo.py
├── configs/ # 하이퍼파라미터 설정 (선택)
│ └── config.yaml
├── results/
│ ├── logs/ # 리워드 로그
│ └── plots/ # 학습 결과 시각화
├── tests/ # 테스트 스크립트
│ └── test_env.py
├── main.py # 메인 실행 파일 (옵션)
├── requirements.txt
└── README.md
## 📊 결과 예시

- PPO 학습 리워드 그래프:
![Reward Curve](results/plots/ppo_reward_curve.png)

## ⚙️ 하이퍼파라미터

| 파라미터        | 값       |
|-----------------|----------|
| 업데이트 주기   | 32 에피소드 |
| PPO Epoch       | 4        |
| Learning Rate   | 1e-3     |
| Discount Factor | 0.99     |
| Hidden Dim      | 32       |