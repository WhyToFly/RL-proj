import numpy as np
import matplotlib.pyplot as plt
import tqdm

class StateActionFeatureVectorWithTile():
    def __init__(self):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here
        pass

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return 10*(3*3*2+3*8*3)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        x = np.zeros((10, 3*3*2+3*8*3))
        if not done:
            x[a, :] = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        return x.reshape(-1)

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alphas:list, # step size
    X:StateActionFeatureVectorWithTile,
    num_episodes:list,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=0.01):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    Gs = []

    #TODO: implement this function
    for alpha, num_episode in zip(alphas, num_episodes):
        for i in tqdm.tqdm(range(num_episode)):
            # print(i)
            state = env.reset()
            act = epsilon_greedy_policy(state, False, w)
            x = X(state, False, act)
            z = np.zeros_like(x)
            q_old = 0
            G = 0.
            k = 0
            while True:
                state, reward, done, info = env.step(act)
                G += reward * gamma ** k
                k += 1
                act = epsilon_greedy_policy(state, done, w)
                x_ = X(state, False, act)
                q = np.dot(w, x)
                q_ = np.dot(w, x_)
                delta = reward + gamma * q_ - q
                z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
                w = w + alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
                q_old = q_
                x = x_
                if done == True:
                    break
            Gs.append(G)
    plt.plot(Gs)
    Gs_avg = np.convolve(Gs, np.ones(100), 'valid') / 100
    plt.plot(Gs_avg)
    plt.savefig("sarsa_lambda.png")
    
    return w
