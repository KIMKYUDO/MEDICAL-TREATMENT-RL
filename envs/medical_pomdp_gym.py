import gymnasium
from gymnasium import spaces
from envs.medical_pomdp_simulator import MedicalPOMDPSimulator

class MedicalPOMDPEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.sim = MedicalPOMDPSimulator()

        self.action_space = spaces.Discrete(4)  # 검사, 약물A, 약물B, 수술
        self.observation_space = spaces.Tuple((
            spaces.Discrete(3),  # 혈액검사: 0=정상, 1=이상, 2=None
            spaces.Discrete(3),  # 증상: 0=무증상, 1=미열, 2=고열
        ))

    def reset(self):
        obs = self.sim.reset()
        return self._convert_obs(obs)
    
    def step(self, action):
        obs, reward, done = self.sim.step(action)
        return self._convert_obs(obs), reward, done, {}
    
    def _convert_obs(self, obs):
        blood, symptom = obs
        blood = 2 if blood is None else blood  # None을 2로 변환
        return (blood, symptom)
    
    def render(self, mode='human'):
        print(f"현재 상태: 숨겨짐, 최근 관측: {self.sim._get_observation(do_test=False)}")