import sys

sys.path.append("../model")
import constants as c
import numpy as np


class Car:
    def __init__(self, v):
        self.v = v

    def act(self):
        return c.KEEP_SPEED

    def get_v(self):
        return self.v

    def set_v(self, v):
        self.v = v

    def kind(self):
        return None

    def acc(self, a):
        self.v += a


class Grandma(Car):
    def __init__(self):
        super().__init__(c.GRANDMA_SPEED)

    def __str__(self):
        return "G"

    def __repr__(self):
        return "G"

    def kind(self):
        return "G"


class Speed_Maniac(Car):
    def __init__(self):
        super().__init__(c.MANIAC_SPEED)

    def __str__(self):
        return "S"

    def __repr__(self):
        return "S"

    def kind(self):
        return "S"


class Agent(Car):
    def __init__(self):
        super().__init__(c.GRANDMA_SPEED + 1)

    def __str__(self):
        return "A"

    def __repr__(self):
        return "A"

    def kind(self):
        return "A"

    def accelerate(self, acc):
        self.v = np.max([c.SPEEDMIN_AGENT, np.min([c.SPEEDLIMIT_AGENT, self.v + acc])])
