import numpy as np
import gym
from gym_puyopuyo import register
from sarsa_lambda import SarsaLambda, StateActionFeatureVectorWithTile
from gym_puyopuyo.agent import SmallTreeSearchAgent

def test_sarsa_lamda():
    register()
    env = gym.make("PuyoPuyoEndlessSmall-v2")

    X = StateActionFeatureVectorWithTile()
    
    ag = SmallTreeSearchAgent()

    gamma = 0.95



    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        s, done = env.reset(), False
        if render: 
            print("====")
            env.render()
            _ = input("new episode...")
        G = 0.
        i = 0
        while not done and i < 100:
            a = greedy_policy(s,done)
            s,r,done,info = env.step(a)
            if render: 
                env.render()
                _ = input("reward = %d..." % r)
            G += r * gamma ** i
            i += 1
        return G

    w = np.zeros(900,)
    Gs = [_eval() for _ in  range(1000)]
    print("zeros:", np.mean(Gs))
    # zeros: ~= 0

    w = np.random.rand(900,)
    Gs = [_eval() for _ in  range(1000)]
    print("random:", np.mean(Gs))
    # random: ~= 0

    # w = SarsaLambda(env, gamma=gamma, lam=0.8, alphas=[0.005, 0.001], X=X, num_episodes=[20000, 20000])
    # Gs = [_eval() for _ in  range(1000)]
    # print("sarsa(lambda):",np.mean(Gs))
    # sarsa(lambda): 9.621066880097173

if __name__ == "__main__":
    test_sarsa_lamda()
