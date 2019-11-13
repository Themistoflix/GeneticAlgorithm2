import random
import numpy as np
from collections import namedtuple

CuttingPlane = namedtuple('CuttingPlane', 'anchor normal')


class CuttingPlaneGenerator:
    def __init__(self, center):
        self.center = center

    def generateNewCuttingPlane(self):
        raise NotImplementedError()

    def setCenter(self, center):
        self.center = center

    def createAxisParallelCuttingPlane(self, position):
        anchor = position
        normal = np.array([0, 0, 0])
        normal[random.randrange(2)] = 1.0

        return CuttingPlane(anchor, normal)


class SphericalCuttingPlaneGenerator(CuttingPlaneGenerator):
    def __init__(self, minRadius, maxRadius, center=None):
        super().__init__(center)
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def generateNewCuttingPlane(self):
        normal = np.array([random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1])
        normal = normal / np.linalg.norm(normal)
        anchor = normal * (self.minRadius + random.random() * (self.maxRadius - self.minRadius))
        anchor = anchor + self.center

        return CuttingPlane(anchor, normal)


# TODO check correctness
class EllipticalCuttingPlaneGenerator(CuttingPlaneGenerator):
    def __init__(self, minW, maxW, minL, maxL, minH, maxH, center=None):
        super().__init__(center)
        self.minW = minW
        self.maxW = maxW

        self.minL = minL
        self.maxL = maxL

        self.minH = minH
        self.maxH = maxH

    def generateNewCuttingPlane(self):
        normal = np.array([0, 0, 0])
        normal[0] = (self.minW + random.random() * (self.maxW - self.minW)) * (2 * random.randrange(2) - 1)
        normal[1] = (self.minL + random.random() * (self.maxL - self.minL)) * (2 * random.randrange(2) - 1)
        normal[2] = (self.minH + random.random() * (self.maxH - self.minH)) * (2 * random.randrange(2) - 1)

        anchor = normal + self.center
        normal = normal / np.linalg.norm(normal)

        return CuttingPlane(anchor, normal)
    
    