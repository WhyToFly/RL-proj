import numpy as np
import gym
from gym_puyopuyo import register
from n_step import semi_gradient_n_step_td
from policy import EpsGreedyPolicy 
from nn import ValueFunctionWithNN
register()

def test_nn(n, plot):
    env = gym.make("PuyoPuyoEndlessSmall-v2")
    print(env.observation_space, env.action_space.n)

    # V = ValueFunctionWithNN(env.observation_space.shape[0], env.action_space.n)
    V = ValueFunctionWithNN(3*3*2+3*8*3, env.action_space.n, alpha=0.001)

    policy = EpsGreedyPolicy(V=V, action_nums=env.action_space.n, eps=0.01)

    semi_gradient_n_step_td(env,0.95,policy,n,V,10000,plot)

    # Vs = [V(s) for s in testing_states]
    # print(Vs)
    # assert np.allclose(Vs,correct_values,0.20,5.), f'{correct_values} != {Vs}, but it might due to stochasticity'

if __name__ == "__main__":
    test_nn(1, "n=1.png")
    # test_nn(2, "n=2.png")
    # test_nn(4, "n=4.png")
    # test_nn(6, "n=6.png")
    # test_nn(8, "n=8.png")
