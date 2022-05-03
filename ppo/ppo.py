import argparse

import numpy as np
import torch

from spinup import ppo_pytorch as ppo
import gym
from gym_puyopuyo import register

import nn

register()

# wrapper to process state, make it ready for CNN
class ProcessStateCNN(gym.ObservationWrapper):
    def __init__(self, env=None, consider_future=False):
        super(ProcessStateCNN, self).__init__(env)
        self.env = env

        orig_shape = env.observation_space.spaces[1].shape

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

        if consider_future:
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

def test_nn(consider_future=False, small=True):
    if small:
        env = gym.make("PuyoPuyoEndlessSmall-v2")
        run_name = "ppo_consider_future-" + str(consider_future) + "_small"
    else:
        env = gym.make("PuyoPuyoEndlessWide-v2")
        run_name = "ppo_consider_future-" + str(consider_future) + "_wide"

    plot = run_name + ".png"

    if consider_future:
        in_channels = 7
    else:
        in_channels = 3

    # wrap environment to get observations usable in nn
    # TODO: try linear processing (concatenate observation using ProcessStateMLP), network
    env = ProcessStateCNN(env, consider_future)

    env_fn = lambda : env
    ac_kwargs = dict(layers=[8,16,16])

    logger_kwargs = dict(output_dir='spinup', exp_name=run_name)

    ppo(env_fn=env_fn,actor_critic=nn.CNNActorCritic, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=50, gamma=0.95, logger_kwargs=logger_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--consider_future', required=True, choices=['True', 'False'])
    parser.add_argument('-s', '--size', default='small', choices=['small', 'wide'])

    args = parser.parse_args()

    consider_future = args.consider_future == "True"
    small = args.size == "small"

    test_nn(consider_future, small)