import numpy as np
import gym
from gym_puyopuyo import register
from gym_puyopuyo.agent import SmallTreeSearchAgent

register()
env = gym.make("PuyoPuyoEndlessSmall-v2")
ag = SmallTreeSearchAgent()
gamma = 0.95

def _eval_treeagent(render=False, epsilon=0.01):
    _, done = env.reset(), False
    if render: 
        print("====")
        env.render()
        _ = input("new episode...")
    G = 0.
    i = 0
    state = env.get_root()
    while not done and i < 100:
        if np.random.rand() < epsilon:
            a = np.random.randint(0, 10)
        else:
            a = ag.get_action(state)
        _,r,done,info = env.step(a)
        state = info["state"]
        if render: 
            env.render()
            _ = input("reward = %d..." % r)
        G += r * gamma ** i
        i += 1
    return G

Gs = [_eval_treeagent() for _ in  range(10)]
print("tree search (eps-greedy):", np.mean(Gs))
# tree search: 25.136939041765675
# tree search (eps-greedy): 23.069859483168393 