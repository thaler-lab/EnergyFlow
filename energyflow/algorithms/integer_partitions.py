import itertools

__all__ = ['int_partition_ordered', 'int_partition_unordered']

# gets ordered integer partitions of d of length e
def int_partition_ordered(d, e):
    for part in int_partition_unordered(d):
        if len(part) != e: continue
        for ordered_part in set(itertools.permutations(part)):
            yield ordered_part

# gets unordered integer partitions of the integer d
def int_partition_unordered(d):
    a = [0]*(d+1)
    k = 1
    y = d - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
