import argparse
from os import path
from datetime import datetime

import numpy as np
import torch

from spinup import ppo_pytorch as ppo
from spinup.utils.run_utils import ExperimentGrid
import gym
from gym_puyopuyo import register

from nn import CNNActorCritic

# wrapper to process state, make it ready for CNN
class ProcessStateCNN(gym.ObservationWrapper):
    def __init__(self, env=None, consider_future=False):
        super(ProcessStateCNN, self).__init__(env)
        self.env = env

        orig_shape = env.observation_space.spaces[1].shape

        self.consider_future = consider_future

        if consider_future:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, orig_shape[1], orig_shape[2]), dtype=int)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, orig_shape[1], orig_shape[2]), dtype=int)

    def observation(self, obs):
        '''
        Take game field and piece data and encode combination.
        This is supposed to solve the issue of multiple possible game states being equivalent; it does not really matter if a
        field is blue or red; the relation between the field and the other fields and the falling game pieces is what matters.
        So we're encoding a matric of size (3 (or 7 if considering future pieces), field_height, field_width) the following way:
        first channel: 1 for every field that matches color of left piece of current puyo
        second channel: 1 for every field that matches color of right piece of current puyo
        (if considering future pieces: third-sixth channel based on future pieces instead of the current one)
        third channel: 1 for every field that is not empty
        '''

        # get field data, piece data
        field_data = obs[1]
        pieces_data = obs[0]

        # transpose for simplicity
        field_data = field_data.transpose([1,2,0])
        pieces_data = pieces_data.transpose([1,2,0])

        # find out where fields don't have color
        no_color = np.expand_dims((field_data.sum(axis=-1) == 0).astype(int), axis=-1)
        # inverse
        filled = (no_color == 0).astype(int)

        # combine arrays (no color is now a new color)
        field_arr = np.concatenate((field_data, no_color), axis=-1)

        # turn into indices instead of one-hot
        argmax_arr = np.argmax(field_arr, axis = -1)
        argmax_color = np.argmax(pieces_data, axis = -1)


        channels_list = []

        if self.consider_future:
            for i in range(argmax_color.shape[0]):
                for j in range(argmax_color[0].shape[0]):
                    channels_list.append(np.expand_dims((argmax_arr == argmax_color[i][j]).astype(int), axis=-1))
        else:
            for i in range(argmax_color[0].shape[0]):
                channels_list.append(np.expand_dims((argmax_arr == argmax_color[0][i]).astype(int), axis=-1))

        channels_list.append(filled)

        # combine all, transpose into original (colors, height, width) again
        return np.expand_dims(np.stack(channels_list, axis=-1).squeeze().transpose([2,0,1]), axis=0)


# wrapper to process state, make it ready for MLP network
class ProcessStateMLP(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessStateMLP, self).__init__(env)
        self.env = env

        orig_shape = env.observation_space.spaces[1].shape

        # calculate size of flattened observation
        self.observation_space = 1

        # multiply dimensions of game field
        for dim in orig_shape:
            self.observation_space *= dim

        # add size of future puyo stack
        self.observation_space += 3 * 6 * 2


    def observation(self, obs):
        '''
        simply concatenate flattened game field, future puyos
        '''
        return np.concatenate(obs[0].flatten(), obs[1].flatten())

def test_nn(consider_future, small, cnn, gamma):
    register()

    if cnn:
        run_name = "ppo_cnn_consider_future_" + str(consider_future)
    else:
        run_name = "ppo_mlp"


    if small:
        env = gym.make("PuyoPuyoEndlessSmall-v2")
        run_name += "_small"
    else:
        env = gym.make("PuyoPuyoEndlessWide-v2")
        run_name += "_wide"

    run_name += "_gamma-" + str(gamma)

    logger_kwargs = dict(output_dir=path.join(".logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S)") + "_" + run_name), exp_name=run_name)

    if cnn:
        # wrap environment to get observations usable in nn
        # TODO: try linear processing (concatenate observation using ProcessStateMLP), network
        env = ProcessStateCNN(env, consider_future)

        env_fn = lambda : env

        ac_kwargs = dict(layers=[8,16,16])
        ppo(env_fn=env_fn,actor_critic=CNNActorCritic, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=200, max_ep_len=250, gamma=gamma, logger_kwargs=logger_kwargs)
    else:
        raise NotImplementedError

def create_env():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessWide-v2")
    env = ProcessStateCNN(env, False)
    return env

def create_env_future():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessWide-v2")
    env = ProcessStateCNN(env, True)
    return env

if __name__ == "__main__":
    # replaced cmd arguments with gridsearch for wide env after proving that small env can be solved by ppo

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_fn', [create_env, create_env_future], 'env_fn')
    eg.add('actor_critic', CNNActorCritic)
    eg.add('gamma', [0.9,0.95,0.98,0.99,0.999], 'gamma')
    eg.add('epochs', 50)
    eg.add('steps_per_epoch', 4000)
    eg.add('max_ep_len', 100)
    eg.add('ac_kwargs:layers', [[4], [8], [8,16], [8,16,16], [32,64,64]], 'layers')
    eg.run(ppo, num_cpu=6)


    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--consider_future', default='False', choices=['True', 'False'])
    parser.add_argument('-n', '--network', default='cnn', choices=['cnn', 'mlp'])
    parser.add_argument('-s', '--size', default='small', choices=['small', 'wide'])
    parser.add_argument('-g', '--gamma', type=float, default=0.99)

    args = parser.parse_args()

    consider_future = args.consider_future == "True"
    small = args.size == "small"
    cnn = args.network == "cnn"

    test_nn(consider_future, small, cnn, args.gamma)
    '''
