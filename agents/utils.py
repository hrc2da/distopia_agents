import numpy as np


def hierarchical_sort(lst, direction='lr', order='descending'):
    assert len(lst) > 0
    to_sort = lst[:]
    sample = to_sort[0]
    sort_order = -1 if order == 'descending' else 1
    if direction == 'rl':
        for i in range(len(sample)):
            to_sort.sort(key=lambda x: sort_order * x[i])
    elif direction == 'lr':
        for j in range(len(sample)-1, -1, -1):
            to_sort.sort(key=lambda x: sort_order * x[j])
    return to_sort


def str2arr(string: str):
    ''' convert a stringified np array back to the array
    :param string: the stringified np array
    :return: a np array
    '''
    assert type(string) == str
    assert string[0] == '['
    assert string[-1] == ']'
    return np.array(string[1:-1].split(), dtype=float)
