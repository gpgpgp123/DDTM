import numpy as np
IMAGE_MEANS =np.array([117.67, 130.39, 121.52, 162.92]) # The setting here is for Chesapeake dataset
IMAGE_STDS = np.array([39.25,37.82,24.24,60.03])
LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
LABEL_CLASS_COLORMAP = { # Color map for Chesapeake dataset
    0:  (0, 0, 0),
    11: (84, 117, 168),
    12: (255, 255, 255),
    21: (204, 177, 177),
    22: (226, 158, 140),
    23: (255, 0, 0),
    24: (181, 0, 0),
    31: (210, 205, 192),
    41: (133, 199, 126),
    42: (56, 129, 78),
    43: (212, 231, 176),
    52: (220, 202, 143),
    71: (253, 233, 170),
    81: (251, 246, 93),
    82: (202, 145, 70),
    90: (200, 230, 248),
    95: (100, 179, 213)
}

target_class = {0, 1, 2, 3, 4}
target_color = {
    0: (0, 0, 0),
    1: (222, 34, 7),
    2: (34, 97, 38),
    3: (0, 255, 36),
    4: (70, 107, 159)
}


LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}

LABEL_IDX_UNIFY_COLORMAP = {
    idx: target_color[c]
    for idx, c in enumerate(target_class)
}

def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1]+1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()

def trans_label(output, dataset):
    # h, w = output.shape
    # if dataset == 'CP_de':      # 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95
        # for i in range(h):      # 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16
        #     for j in range(w):
        #         if output[i, j] == 3 or output[i, j] == 4 or output[i, j] == 5 or output[i, j] == 6:
        #             output[i, j] = 1
        #         elif output[i, j] == 8 or output[i, j] == 9 or output[i, j] == 10 or output[i, j] == 15:
        #             output[i, j] = 2
        #         elif output[i, j] == 7 or output[i, j] == 11 or output[i, j] == 12 or output[i, j] == 13 or output[i, j] == 14 or output[i, j] == 16:
        #             output[i, j] = 3
        #         elif output[i, j] == 1:
        #             output[i, j] = 4
        #         else:
        #             output[i, j] = 0
    output = np.array(output)
    mask_1 = (output == 3) | (output == 4) | (output == 5) | (output == 6)
    mask_2 = (output == 8) | (output == 9) | (output == 10) | (output == 15)
    mask_3 = (output == 7) | (output == 11) | (output == 12) | (output == 13) | (output == 14) | (output == 16)
    mask_4 = (output == 1)
    output[mask_1] = 1
    output[mask_2] = 2
    output[mask_3] = 3
    output[mask_4] = 4
    output[~(mask_1 | mask_2 | mask_3 | mask_4)] = 0
    # else:
    #     raise ValueError('Invalid dataset name')
    return output
