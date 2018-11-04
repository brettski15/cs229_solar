import numpy as np


class SolarLabel(object):
    area = None
    systems = None
    tiles = None

    def __init__(self, area, systems, tiles):
        self.area = area
        self.systems = systems
        self.tiles = tiles

    def __str__(self):
        return f"Label area: {self.area}, System Count: {self.systems}, Tile Count: {self.tiles}"

    def __repr__(self):
        return f"SolarLabel(area={self.area}, systems={self.systems}, tiles={self.tiles}"


class SolarExample(object):
    label = None
    column_vals = None

    def __init__(self, values, label):
        self.column_vals = np.array(values)
        self.label = label


class SolarMatrix(object):
    data = None
    headers = []

    def __init__(self, examples, headers):
        self.data = examples
        self.headers = headers
