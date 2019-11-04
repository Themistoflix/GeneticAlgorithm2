import numpy as np

class BoundingBox:
    def __init__(self, w, l, h, position):
        self.width = w
        self.length = l
        self.height = h

        self.position = position

    def getCenter(self):
        return self.position + np.array([self.width/2, self.length/2, self.height/2])
