from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here
        self.model = nn.Sequential(
            # nn.Linear(state_dims, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, num_actions),
            nn.Linear(state_dims, num_actions),
            nn.Softmax())
        # assert alpha <= 3e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.num_actions = num_actions

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

    def __call__(self,s) -> int:
        # TODO: implement this method
        self.model.eval()
        s = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        s = torch.Tensor(s)
        p = self.model(s).detach().numpy()
        return np.random.choice(self.num_actions, p=p)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        self.model.train()
        s = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        s = torch.Tensor(s)
        loss = - torch.log(self.model(s)[a]) * gamma_t * delta
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        # self.model = nn.Sequential(
        #     nn.Linear(state_dims, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1))
        # assert alpha <= 3e-4
        self.model = nn.Linear(state_dims, 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        # TODO: implement this method
        self.model.eval()
        s = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        s = torch.Tensor(s)
        return float(self.model(s))

    def update(self,s,G):
        # TODO: implement this method
        self.model.train()
        self.optimizer.zero_grad()
        s = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        s = torch.Tensor(s)
        loss = 0.5 * (G - self.model(s)) * (G - self.model(s))
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    G0 = []
    for i in tqdm.tqdm(range(num_episodes)):
        state = env.reset()
        states = []
        acts = []
        rews = []
        while True:
            act = pi(state)
            states.append(state)
            acts.append(act)
            state, reward, done, info = env.step(act)
            rews.append(reward)
            if done:
                break
        T = len(states)
        for t in range(T):
            G = 0.0
            for k in range(t, T):
                G += gamma ** (k - t) * rews[k]
            if t == 0:
                G0.append(G)
            delta = G - V(states[t])
            V.update(states[t], G)
            pi.update(states[t], acts[t], gamma ** t, delta)
    return G0

# # test
# pi = PiApproximationWithNN(3, 4, 0.0003)
# print(pi([1,2,3]))