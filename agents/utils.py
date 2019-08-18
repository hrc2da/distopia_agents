def hierarchical_sort(lst,direction='lr',order='descending'):
    assert len(lst) > 0
    to_sort = lst[:]
    sample = to_sort[0]
    sort_order = -1 if order == 'descending' else 1
    if direction == 'lr':
        for i in range(len(sample)):
            to_sort.sort(key = lambda x : sort_order * x[i])
    elif direction == 'rl':
        for j in range(len(sample),0,-1):
            to_sort.sort(key = lambda x : sort_order * x[i])
    return to_sort