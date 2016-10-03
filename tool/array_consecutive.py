
def number_of_consecutive_values(array, how_many_consecutives=4):
    consecutive_cells = 1
    nb_consecutive = {}
    for i in range(1, len(array), 1):
        if array[i - 1] == array[i]:
            consecutive_cells += 1
        else:
            consecutive_cells = 1
        if consecutive_cells == how_many_consecutives:
            if not array[i] in nb_consecutive:
                nb_consecutive[array[i]] = 0
            nb_consecutive[array[i]] += 1
            consecutive_cells = 0
    return nb_consecutive


def number_of_consecutive_values_2d(array, how_many_consecutives=4):

    nb_consecutive = {}
    for sub_array in array:
        sub_consecutive = number_of_consecutive_values(
            sub_array, how_many_consecutives)
        for key in sub_consecutive:
            if not key in nb_consecutive:
                nb_consecutive[key] = 0
            nb_consecutive[key] += sub_consecutive[key]
    return nb_consecutive
