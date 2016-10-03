import numpy as np


def transpose_horizontally(array):
    return np.transpose(array)


def transpose_diagonally(array, bottom_to_top=False):
    transposed_cells = []
    y_length = len(array)
    x_length = len(array[0])
    max_length = max(x_length, y_length)
    for k in range(0, 2 * max_length + 1):
        row = []
        for y in range(y_length - 1, -1, -1):
            x = k - ((y_length - y) if bottom_to_top else y)
            if x >= 0 and x < x_length:
                row.append(array[y][x])
        if len(row) > 0:
            transposed_cells.append(row)
    return transposed_cells
