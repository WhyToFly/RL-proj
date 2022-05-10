import numpy as np
import torch

def encode_state(state, consider_future=False, arch="conv"):
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # get field data, piece data
    field_data = torch.Tensor(state[1]).to(device)
    pieces_data = torch.Tensor(state[0]).to(device)

    if arch == "conv":


        # transpose for simplicity
        field_data = field_data.permute([1,2,0])
        pieces_data = pieces_data.permute([1,2,0])

        # find out where fields don't have color
        no_color = torch.unsqueeze((field_data.sum(axis=-1) == 0).float(), -1)
        # inverse
        filled = (no_color == 0).float()

        # combine arrays (no color is now a new color)
        field_arr = torch.concat((field_data, no_color), axis=-1)

        # turn into indices instead of one-hot
        argmax_arr = torch.argmax(field_arr, axis = -1)
        argmax_color = torch.argmax(pieces_data, axis = -1)


        channels_list = []

        if consider_future:
            for i in range(argmax_color.shape[0]):
                for j in range(argmax_color[0].shape[0]):
                    channels_list.append((argmax_arr == argmax_color[i][j]).float().unsqueeze(-1))
        else:
            for i in range(argmax_color[0].shape[0]):
                channels_list.append((argmax_arr == argmax_color[0][i]).float().unsqueeze(-1))

        channels_list.append(filled)

        # combine all, transpose into original (colors, height, width) again
        return torch.stack(channels_list, axis=-1).squeeze().permute([2,0,1]).unsqueeze(0)
    else:
        assert arch == "mlp"
        return torch.concat((field_data.reshape(-1), pieces_data.reshape(-1))).unsqueeze(0)
