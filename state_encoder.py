import numpy as np


def encode_state(state, consider_future=False):
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
    field_data = state[1]
    pieces_data = state[0]

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
    return np.stack(channels_list, axis=-1).squeeze().transpose([2,0,1])
