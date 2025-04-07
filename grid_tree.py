import numpy as np
from addict import Dict


class GridTree:
    def __init__(self, data: np.ndarray, label: np.ndarray):
        """
        data: N * 2 float array
        label: N * 1 int array
        """
        self.data = data
        self.label = label
        assert len(data) == len(label)
        self.levels = [4, 16, 32, 64, 128, 256, 512]
        # for i in range(1, 1000, 10):
        #     grids = self.build(i)
        #     # output mid and max grid number
        #     flat = sum(grids, [])
        #     flat_count = np.array([len(x) for x in flat])
        #     flat_count.sort()
        #     print(i, flat_count[len(flat_count) // 2], flat_count[-1])
        self.trees = []
        for level in self.levels:
            self.trees.append(self.build(level))

    @staticmethod
    def calc_gird(data, min, delta):
        """
        return grid number i, j of data
        """
        grid_x = int((data[0] - min[0]) / delta[0])
        grid_y = int((data[1] - min[1]) / delta[1])
        return grid_x, grid_y

    def build(self, grid_number):
        """
        build grid tree with grid number. Return builded grid tree.
        """
        res = Dict()
        res.min = [np.min(self.data[:, 0]), np.min(self.data[:, 1])]
        res.max = [np.max(self.data[:, 0]), np.max(self.data[:, 1])]
        res.max[0] += (res.max[0] - res.min[0]) / 10000
        res.max[1] += (res.max[1] - res.min[1]) / 10000
        res.data = [[[] for _ in range(grid_number)] 
                    for _ in range(grid_number)]
        res.label = [[[] for _ in range(grid_number)] 
                     for _ in range(grid_number)]
        res.delta = [
            (res.max[0] - res.min[0]) / grid_number,
            (res.max[1] - res.min[1]) / grid_number,
        ]
        for i in range(len(self.data)):
            grid_x, grid_y = self.calc_gird(self.data[i], res.min, res.delta)
            res.data[grid_x][grid_y].append(self.data[i])
            res.label[grid_x][grid_y].append(self.label[i])
        for i in range(grid_number):
            for j in range(grid_number):
                if len(res.data[i][j]) > 0:
                    res.data[i][j] = np.array(res.data[i][j])
                    res.label[i][j] = np.array(res.label[i][j])
        return res

    def query(self, point, k=1):
        """
        point: 2-d float array
        k: how many nearest neighbor to return (currently only support 1)

        return: nearest neighbor point and its label
        """
        assert k == 1, 'currently only support k=1'
        gnums = []
        for tree, level in zip(self.trees, self.levels):
            gx, gy = self.calc_gird(point, tree.min, tree.delta)
            if gx < 0:
                gx = 0
            if gy < 0:
                gy = 0
            if gx >= level:
                gx = level - 1
            if gy >= level:
                gy = level - 1
            gnums.append((gx, gy))
        select_level = 0
        for i in range(len(gnums)):
            if len(self.trees[i].data[gnums[i][0]][gnums[i][1]]) > 0:
                select_level = i
        tree = self.trees[select_level]
        min_point = None
        min_dist = None
        min_label = None
        midx, midy = gnums[select_level]
        spand = 2
        if len(tree.data[midx][midy]) != 0:
            delta = tree.data[midx][midy] - point
            dist2 = (delta ** 2).sum(axis = -1)
            idx = np.argmin(dist2)
            min_point = tree.data[midx][midy][idx]
            min_label = tree.label[midx][midy][idx]
            min_dist = dist2[idx]
            if min_dist < tree.delta[0] ** 2:
                spand = 1
        for i in range(midx - spand, midx + 1 + spand):
            for j in range(midy - spand, midy + 1 + spand):
                if i == midx and j == midy:
                    continue
                if i < 0 or i >= len(tree.data) or j < 0 or j >= len(tree.data):
                    continue
                if len(tree.data[i][j]) > 0:
                    delta = tree.data[i][j] - point
                    dist2 = (delta ** 2).sum(axis = -1)
                    idx = np.argmin(dist2)
                    if min_dist is None or dist2[idx] < min_dist:
                        min_dist = dist2[idx]
                        min_point = tree.data[i][j][idx]
                        min_label = tree.label[i][j][idx]
        if min_point is None or min_label is None:
            raise Exception('no point found')
        return min_point, min_label
