import numpy as np


class FCCLattice:
    # Use a right-handed cartesian coordinate system with only positive coordinates
    # width  is associated with the x component
    # length is associated with the y component
    # height is associated with the z component
    def __init__(self, width, length, height, latticeConstant):
        assert width % 2 == 1 and length % 2 == 1 and height % 2 == 1, "box dimensions need to be odd!"
        self.width = width
        self.length = length
        self.height = height
        self.latticeConstant = latticeConstant

        self.latticePointsPerEvenLayer = (int(self.width / 2) + 1) ** 2 + (int(self.width / 2)) ** 2
        self.latticePointsPerOddLayer = (width - 1) / 2 * (length + 1) / 2 + (width + 1) / 2 * (length - 1) / 2

    def isValidIndex(self, index):
        if (index < 0) or (index > (self.height + 1) / 2 * self.latticePointsPerEvenLayer + (
                self.height - 1) / 2 * self.latticePointsPerOddLayer):
            return False

        return True

    def isValidPosition(self, position):
        # check boundaries
        if position[0] < 0 or position[1] < 0 or position[2] < 0:
            return False

        if position[0] > self.width or position[1] > self.length or position[2] > self.height:
            return False

        # check if the position specifies as lattice point
        if position[2] % 2 == 0:
            if (position[0] % 2 == 0 and position[1] % 2 == 0) or (position[0] % 2 == 1 and position[1] % 2 == 1):
                return True
            else:
                return False
        if position[2] % 2 == 1:
            if (position[0] % 2 == 0 and position[1] % 2 == 1) or (position[0] % 2 == 1 and position[1] % 2 == 0):
                return True
            else:
                return False

    def getIndexFromLatticePosition(self, position):
        def indexInXYPlane(x, y, z, width):
            if z % 2 == 0:
                if x % 2 == 0:
                    return y / 2 * (width + 1) / 2 + (y / 2) * (width - 1) / 2 + (x / 2)
                else:
                    return (y + 1) / 2 * (width + 1) / 2 + (y - 1) / 2 * (width - 1) / 2 + (x - 1) / 2
            else:
                if x % 2 == 0:
                    return (y + 1) / 2 * (width - 1) / 2 + (y - 1) / 2 * (width + 1) / 2 + (x / 2)
                else:
                    return y / 2 * (width + 1) / 2 + y / 2 * (width - 1) / 2 + (x - 1) / 2

        oddLayersBelow = 0
        evenLayersBelow = 0
        if position[2] % 2 == 0:
            oddLayersBelow = evenLayersBelow = position[2] / 2
        else:
            oddLayersBelow = (position[2] - 1) / 2
            evenLayersBelow = (position[2] + 1) / 2
        index = oddLayersBelow * self.latticePointsPerOddLayer + evenLayersBelow * self.latticePointsPerEvenLayer \
                + indexInXYPlane(position[0], position[1], position[2], self.width)

        return index

    def getIndexFromCartesianPosition(self, position):
        return self.getIndexFromLatticePosition(position/self.latticeConstant)

    def getLatticePositionFromIndex(self, index):
        def positionFromPlaneIndex(planeIndex, z, width):
            # a 'full line' is a row with length l and the following row with length l - 1 (or l + 1)
            fullLinesAbove = int(planeIndex / width)
            # shift the index into the first line
            newIndex = planeIndex - fullLinesAbove * width
            if int(z) % 2 == 0:
                # check in which row the index is
                x = y = 0
                if newIndex >= (width + 1) / 2:
                    y = 1
                else:
                    y = 0

                if int(y) == 0:  # upper row
                    x = newIndex * 2
                else:  # lower row
                    newIndex = newIndex - (width + 1) / 2
                    x = newIndex * 2 + 1

            if z % 2 == 1:
                # check in which row the index is
                if newIndex < (width - 1) / 2:
                    y = 0
                else:
                    y = 1

                if y == 0:
                    x = newIndex * 2 + 1
                else:
                    newIndex = newIndex - (width - 1) / 2
                    x = newIndex * 2

            return x, y + 2 * fullLinesAbove

        # a 'full block' is an even and an odd layer
        fullBlocksBelow = int(index / (self.latticePointsPerEvenLayer + self.latticePointsPerOddLayer))
        # shift the index into the first line
        newIndex = index - fullBlocksBelow * (self.latticePointsPerEvenLayer + self.latticePointsPerOddLayer)

        # check if it is in the upper (odd) or lower(even) layer
        if newIndex >= self.latticePointsPerEvenLayer:
            z = 1
        else:
            z = 0
        planeIndex = newIndex - z * self.latticePointsPerEvenLayer
        x, y = positionFromPlaneIndex(planeIndex, z, self.width)

        return np.array([x, y, z + 2 * fullBlocksBelow])

    def getCartesianPositionFromIndex(self, index):
        return self.getLatticePositionFromIndex(index) * self.latticeConstant

    def getAnchorOfCenteredBox(self, w, h, l):
        anchorPointX = int((self.width - w - 1) / 2)
        anchorPointY = int((self.length - l - 1) / 2)
        anchorPointZ = int((self.height - h - 1) / 2)

        if not self.isValidPosition(np.array([anchorPointX, anchorPointY, anchorPointZ])):
            anchorPointZ = anchorPointZ + 1

        return np.array([anchorPointX, anchorPointY, anchorPointZ])
