"""Minimal stand-ins for Isaac Lab config classes: store kwargs, support .replace() and .func()."""
import copy


class Cfg:
    def __init__(self, *args, **kw):
        self.__dict__.update(kw)

    def replace(self, **kw):
        new = copy.deepcopy(self)
        new.__dict__.update(kw)
        return new

    def func(self, *a, **k):
        """Spawner stand-in: remember which prim path expression was spawned (the fake runtime reads it)."""
        if a and isinstance(a[0], str):
            from isaaclab._fake_runtime import STATE

            STATE["spawned"].append(a[0])

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class Thing:
    def __init__(self, *a, **k):
        pass
