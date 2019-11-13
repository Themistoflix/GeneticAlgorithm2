import numpy as np

class BoundingBox:
    def __init__(self, w, l, h, position):
        self.width = w
        self.length = l
        self.height = h

        self.position = position

    def getCenter(self):
        return self.position + np.array([self.width/2, self.length/2, self.height/2])

    def print(self):
        print('Bounding Box:')
        print("width: {0} length: {1} height: {2}".format(self.width, self.length, self.height))
        print("AnchorPoint: x = {0}, y = {1}, z = {2}".format(self.position[0], self.position[1], self.position[2]))
        print("Center: {0}".format(self.getCenter()))
