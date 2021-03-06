import numpy as np


class SolarLabel(object):
    """
    A class to contain the labels for the dataset. A label is comprised of:
    Total Solar Panel Area (per <unknown unit>)
    Number of Solar Systems (per <unknown unit>)
    Number of Solar Tiles (per <unknown unit>)
    """
    area = None
    systems = None
    tiles = None

    def __init__(self, area, systems, tiles):
        self.area = area
        self.systems = systems
        self.tiles = tiles

    def __str__(self):
        return f"SolarLabel(Area: {self.area}, System Count: {self.systems}, Tile Count: {self.tiles})"

    def __repr__(self):
        return f"SolarLabel(area={self.area}, systems={self.systems}, tiles={self.tiles})"


# class SolarExample(object):
#     """
#     A class that contains all information for a given example.
#     The xs are the pieces of data, and the label is an instance of SolarLabel from above
#     """
#     label = None
#     xs = None
#
#     def __init__(self, values, label):
#         self.xs = values
#         self.label = label
#
#
# class SolarMatrix(object):
#     """
#     A class to contain the entire set of data.
#     data is an array of SolarExamples
#     labels is the corresponding array of labels
#     headers is an array of strings that label each of the x columns (excluding label columns)
#     """
#     data = None
#     labels = None
#     headers = []
#
#     def __init__(self, examples, labels, headers):
#         self.data = examples
#         self.labels = labels
#         self.headers = headers
#
#     def __repr__(self):
#         return f"SolarMatrix(examples: {len(self.data)}. columns: {len(self.headers)})"


class SimpleMatrix(object):
    X = None
    labels = None

    def __init__(self, xs, labels):
        self.X = np.array(xs)
        self.labels = np.array(labels)

    def get_area_labels(self):
        return np.array([ l.area for l in self.labels ])

    def get_tile_count_labels(self):
        return np.array([l.tiles for l in self.labels])

    def get_system_count_labels(self):
        return np.array([l.systems for l in self.labels])
