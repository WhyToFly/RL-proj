from gym_puyopuyo import register
from gym import make
# from sarsa import StateActionFeatureVectorWithTile

# encoder = StateActionFeatureVectorWithTile()

register()

small_env = make("PuyoPuyoEndlessSmall-v2")


for i in range(100):
    print("==")
    action = small_env.action_space.sample()
    state, reward, done, info = small_env.step(action)
    print(reward, state[0].shape, state[1].shape)
    # print(encoder(state, done, action).shape)
    small_env.render()
    if done:
        break