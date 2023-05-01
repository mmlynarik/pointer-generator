import itertools


class A:
    def __getitem__(self, index):
        if index >= 10:
            raise IndexError
        return index * 111


it = iter(A())

print(list(itertools.islice(it, 2)))
print(list(itertools.islice(it, 2)))
