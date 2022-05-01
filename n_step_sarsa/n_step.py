import numpy as np
import tqdm
from policy import EpsGreedyPolicy
import matplotlib.pyplot as plt

class ValueFunctionWithApproximation(object):
    def __call__(self,s,a) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
            action
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,G,s_tau,a_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau,a_tau;w)] \nabla\hat{v}(s_tau,a_tau;w)

        input:
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
            a_tau: target action for updating
        ouptut:
            None
        """
        raise NotImplementedError()


def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:EpsGreedyPolicy,
    n:int,
    V:ValueFunctionWithApproximation,
    num_episode:int,
    max_steps:int,
    plot:str,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    Gs_ = []
    for i in tqdm.tqdm(range(num_episode)):
        # generate traj
        state = env.reset()
        traj = []
        G_ = 0.0
        t = 0
        for step in range(max_steps):
            a = pi.action(state)
            old_state = state
            state, r, done, info = env.step(a)
            traj.append((old_state, a, r, state))
            tau = t - n
            old_tau = tau
            if tau >= 0:
                G = 0.0
                for i in range(tau, tau+n):
                    G += gamma ** (i - tau) * traj[i][2]
                G += gamma ** (i - tau + 1) * V(traj[i+1][0], traj[i+1][1])
                V.update(G, traj[tau][0], traj[tau][1])

            G_ += r * gamma ** t
            t += 1
            if done == True:
                break
        Gs_.append(G_)
        T = len(traj)
        for tau in range(max(old_tau+1, 0), len(traj)):
            G = 0
            for i in range(tau, min(tau + n, T)):
                G += gamma ** (i - tau) * traj[i][2]
            if i < T-1:
                G += gamma ** (i - tau + 1) * V(traj[i+1][0], traj[i+1][1])
            V.update(G, traj[tau][0], traj[tau][1])

    plt.figure()
    plt.plot(Gs_)
    Gs_avg = np.convolve(Gs_, np.ones(100), 'valid') / 100
    plt.plot(Gs_avg)
    plt.savefig(plot)
    plt.close()
    return V
