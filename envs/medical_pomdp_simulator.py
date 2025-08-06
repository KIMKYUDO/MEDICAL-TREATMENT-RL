import numpy as np

class MedicalPOMDPSimulator:
    def __init__(self):      # 환경 초기화
        # 상태: 0=Healthy, 1=Mild, 2=Severe, 3=Critical
        # 행동: 0=검사, 1=약물A, 2=약물B, 3=수술
        self.state_space = [0, 1, 2, 3]
        self.action_space = [0, 1, 2, 3]

        # obs(Partially Observable)
        # 혈액검사: 0=Normal, 1=Abnormal
        # 증상: 0=None, 1=Fever, 2=High Fever
        self.blood_obs_space = [0, 1]
        self.symptom_obs_space = [0, 1, 2]

        self.transion_table = self._init_transition_table()
        self.symptom_obs = self._init_symptom_obs()
        self.blood_obs = self._init_blood_obs()

        self.reset()

    def reset(self):         # 횐경 초기화 및 첫 관측 반환
        self.state = np.random.choice([1, 2])
        return self._get_observation(do_test=False)
    
    def step(self, action):  # 행동 실행 -> 다음 관측, 보상, 종료여부 반환
        done = False
        reward = 0

        # 상태 전이 항상 발생
        probs = self.transion_table[self.state][action]
        self.state = np.random.choice(self.state_space, p=probs)

        if action == 0:  # 검사
            obs = self._get_observation(do_test=True)
            reward = -1  # 검사 비용
        else:
            obs = self._get_observation(do_test=False)
            reward = self._get_reward(self.state)
        
        if self.state == 3:  # Critical 상태는 종료
            done = True
        
        return obs, reward, done
    
    def _get_observation(self, do_test):     # 현재 상태 기반으로 관측 샘플링
        symptom = np.random.choice(self.symptom_obs_space, p=self.symptom_obs[self.state])
        if do_test:
            blood = np.random.choice(self.blood_obs_space, p=self.blood_obs[self.state])
        else:
            blood = None
        return (blood, symptom)
    
    def _get_reward(self, state):            # 상태에 따른 보상 반환
        if state == 0:
            return 10
        elif state == 3:
            return -100
        else:
            return -5
        

    def _init_transition_table(self):        # 상태 전이 확률 테이블 생성
        return {
            0: {
                0: [0.99, 0.01, 0.0, 0.0],
                1: [1.0, 0.0, 0.0, 0.0],
                2: [1.0, 0.0, 0.0, 0.0],
                3: [1.0, 0.0, 0.0, 0.0],
            },
            1: {
                0: [0.2, 0.6, 0.2, 0.0],
                1: [0.7, 0.2, 0.1, 0.0],
                2: [0.5, 0.3, 0.2, 0.0],
                3: [0.8, 0.1, 0.1, 0.0],
            },
            2: {
                0: [0.1, 0.3, 0.4, 0.2],
                1: [0.2, 0.4, 0.3, 0.1],
                2: [0.1, 0.3, 0.4, 0.2],
                3: [0.4, 0.4, 0.1, 0.1],
            },
            3: {
                0: [0.0, 0.1, 0.3, 0.6],
                1: [0.0, 0.2, 0.4, 0.4],
                2: [0.0, 0.3, 0.3, 0.4],
                3: [0.3, 0.4, 0.2, 0.1],
            },
        }
    
    def _init_symptom_obs(self):              # 증상 관측 확률 테이블 생성
        return {
            0: [0.9, 0.1, 0.0],
            1: [0.4, 0.5, 0.1],
            2: [0.1, 0.4, 0.5],
            3: [0.0, 0.2, 0.8],
        }
    
    def _init_blood_obs(self):               # 혈액검사 관측 확률 테이블 생성
        return {
            0: [0.9, 0.1],
            1: [0.5, 0.5],
            2: [0.2, 0.8],
            3: [0.1, 0.9],
        }