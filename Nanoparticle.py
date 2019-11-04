import numpy as np
import random
from ase import Atoms

import FCCLattice
import BoundingBox


class Nanoparticle:
    def __init__(self, lattice):
        self.lattice = lattice

        self.indicesOnLattice = list()
        self.atomicSymbols = list()
        self.boundingBox = BoundingBox.BoundingBox(0, 0, 0, np.array([0, 0, 0]))

    def rectangularPrism(self, w, l, h, symbol='X'):
        anchorPoint = self.lattice.getAnchorOfCenteredBox(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    curPosition = anchorPoint + np.array([x, y, z])

                    if self.lattice.isValidPosition(curPosition):
                        index = self.lattice.getIndexFromLatticePosition(curPosition)

                        self.indicesOnLattice.append(index)
                        self.atomicSymbols.append(symbol)

        self.findBoundingBox()

    def convexShape(self, numberOfAtomsOfEachKind, atomicSymbols, w, l, h):
        def drawCutPlaneFromSphere(minRadius, maxRadius, center):
            cutPlaneNormal = np.array([random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1])
            cutPlaneNormal = cutPlaneNormal / np.linalg.norm(cutPlaneNormal)
            cutPlaneAnchor = cutPlaneNormal * (minRadius + random.random() * (maxRadius - minRadius))
            cutPlaneAnchor = cutPlaneAnchor + center

            return cutPlaneAnchor, cutPlaneNormal

        finalNumberOfAtoms = sum(numberOfAtomsOfEachKind)
        self.rectangularPrism(w, l, h)

        currentAtoms = set(self.indicesOnLattice)
        MAX_CUTTING_ATTEMPTS = 50
        currentCuttingAttempt = 0

        while len(currentAtoms) > finalNumberOfAtoms and currentCuttingAttempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            cutPlaneAnchor, cutPlaneNormal = drawCutPlaneFromSphere(min(w, l, h)*0.9, min(w, l, h), self.boundingBox.getCenter())

            # count atoms to be removed, if new Count >= final Number remove
            atomsToBeRemoved = set()
            for atom in currentAtoms:
                position = self.lattice.getCartesianPositionFromIndex(atom)
                if np.dot((position - cutPlaneAnchor), cutPlaneNormal) > 0.0:
                    atomsToBeRemoved.add(atom)

            if len(atomsToBeRemoved) != 0.0 and len(currentAtoms) - len(atomsToBeRemoved) >= finalNumberOfAtoms:
                currentAtoms = currentAtoms.difference(atomsToBeRemoved)
                print(cutPlaneNormal, cutPlaneAnchor)
                print(len(atomsToBeRemoved))
                currentCuttingAttempt = 0
            else:
                currentCuttingAttempt = currentCuttingAttempt + 1

        if currentCuttingAttempt == MAX_CUTTING_ATTEMPTS:
            # place cutting plane parallel to one of the axes and at the anchor point
            cutPlaneAnchor = self.boundingBox.position
            cutPlaneNormal = np.array([0, 0, 0])
            cutPlaneNormal[random.randrange(2)] = 1.0

            # shift till too many atoms would get removed
            numberOfAtomsYetToBeRemoved = len(currentAtoms) - finalNumberOfAtoms
            atomsToBeRemoved = set()
            while len(atomsToBeRemoved) < numberOfAtomsYetToBeRemoved:
                atomsToBeRemoved.clear()
                cutPlaneAnchor = cutPlaneAnchor + cutPlaneNormal * self.lattice.latticeConstant
                for atom in currentAtoms:
                    position = self.lattice.getCartesianPositionFromIndex(atom)
                    if np.dot((position - cutPlaneAnchor), cutPlaneNormal) < 0.0:
                        atomsToBeRemoved.add(atom)

            # remove atoms till the final number is reached "from the ground up"
            positions = list()
            for atom in atomsToBeRemoved:
                positions.append(self.lattice.getCartesianPositionFromIndex(atom))

            # TODO implement sorting prioritzing the different directions in random order
            def directionSort(position):
                return position[0]

            positions.sort(key=directionSort)
            positions = positions[:numberOfAtomsYetToBeRemoved]

            atomsToBeRemoved.clear()
            for position in positions:
                atomsToBeRemoved.add(self.lattice.getIndexFromCartesianPosition(position))

            currentAtoms = currentAtoms.difference(atomsToBeRemoved)

        # update the list of atoms of the particle
        self.indicesOnLattice.clear()
        for atom in currentAtoms:
            self.indicesOnLattice.append(atom)

        self.findBoundingBox()

        # redistribute the different elements randomly
        self.atomicSymbols.clear()
        for index, symbol in enumerate(atomicSymbols):
            for i in range(numberOfAtomsOfEachKind[index]):
                self.atomicSymbols.append(symbol)
        random.shuffle(self.atomicSymbols)

    def findBoundingBox(self):
        minCoordinates = np.array([1e10, 1e10, 1e10])
        maxCoordinates = np.array([-1e10, -1e10, -1e10])

        for atom in self.indicesOnLattice:
            curPosition = self.lattice.getCartesianPositionFromIndex(atom)
            for coordinate in range(3):
                if curPosition[coordinate] < minCoordinates[coordinate]:
                    minCoordinates[coordinate] = curPosition[coordinate]
                if curPosition[coordinate] > maxCoordinates[coordinate]:
                    maxCoordinates[coordinate] = curPosition[coordinate]

        w = maxCoordinates[0] - minCoordinates[0]
        l = maxCoordinates[1] - minCoordinates[1]
        h = maxCoordinates[2] - minCoordinates[2]

        self.boundingBox = BoundingBox.BoundingBox(w, l, h, minCoordinates)

    def getLatticePositions(self):
        positions = list()
        for atomIndex in self.indicesOnLattice:
            positions.append(self.lattice.getLatticPositionFromIndex(atomIndex))

        return positions

    def getCartesianPositions(self):
        positions = list()
        for atomIndex in self.indicesOnLattice:
            positions.append(self.lattice.getCartesianPositionFromIndex(atomIndex))

        return positions

    def getAtoms(self, centered=True):
        atomPositions = self.getCartesianPositions()
        if centered:
            centerOfMass = np.array([0, 0, 0])
            for position in atomPositions:
                centerOfMass = centerOfMass + position
                centerOfMass = centerOfMass / len(atomPositions)

            return Atoms(positions=(atomPositions - centerOfMass), symbols=self.atomicSymbols)
        else:
            return Atoms(positions=atomPositions, symbols=self.atomicSymbols)
