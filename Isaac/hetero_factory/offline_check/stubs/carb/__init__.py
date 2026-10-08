class Float3(tuple):
    def __new__(cls, *v): return tuple.__new__(cls, v)
