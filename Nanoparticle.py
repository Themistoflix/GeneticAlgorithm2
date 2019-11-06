import numpy as np
import random
import copy
from collections import namedtuple
from ase import Atoms

import FCCLattice
import BoundingBox


class Nanoparticle:
    Atom = namedtuple('Atom', 'listIndex latticeIndex symbol')

    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = list()
        self.boundingBox = BoundingBox.BoundingBox(0, 0, 0, np.array([0, 0, 0]))
        self.neighborList = list()

    def rectangularPrism(self, w, l, h, symbol='X'):
        anchorPoint = self.lattice.getAnchorOfCenteredBox(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    curPosition = anchorPoint + np.array([x, y, z])

                    if self.lattice.isValidLatticePosition(curPosition):
                        latticeIndex = self.lattice.getIndexFromLatticePosition(curPosition)

                        newAtom = self.Atom(len(self.atoms), latticeIndex, symbol)

                        self.atoms.append(newAtom)

        self.findBoundingBox()
        self.constructNeighborList()

    def fromParticleData(self, atoms):
        self.atoms = copy.deepcopy(atoms)
        self.findBoundingBox()
        self.constructNeighborList()

    def splitAtomsAlongPlane(self, cutPlaneAnchor, cutPlaneNormal, atoms=None):
        if atoms is None:
            atoms = self.atoms

        indicesInPositiveSubspace = set()
        indicesInNegativeSubspace = set()

        for atom in atoms:
            position = self.lattice.getCartesianPositionFromIndex(atom.latticeIndex)
            if np.dot((position - cutPlaneAnchor), cutPlaneNormal) >= 0.0:
                indicesInPositiveSubspace.add(atom)
            else:
                indicesInNegativeSubspace.add(atom)
        return indicesInPositiveSubspace, indicesInNegativeSubspace

    def convexShape(self, numberOfAtomsOfEachKind, atomicSymbols, w, l, h):
        def drawCutPlaneFromSphere(minRadius, maxRadius, center):
            cutPlaneNormal = np.array([random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1])
            cutPlaneNormal = cutPlaneNormal / np.linalg.norm(cutPlaneNormal)
            cutPlaneAnchor = cutPlaneNormal * (minRadius + random.random() * (maxRadius - minRadius))
            cutPlaneAnchor = cutPlaneAnchor + center

            return cutPlaneAnchor, cutPlaneNormal

        def drawCutPlaneFromRectangularPrism(minW, maxW, minL, maxL, minH, maxH, center):
            cutPlaneNormal = np.array([0, 0, 0])
            cutPlaneNormal[0] = (minW + random.random()*(maxW - minW))*(2*random.randrange(2) - 1)
            cutPlaneNormal[1] = (minL + random.random() * (maxL - minL)) * (2*random.randrange(2) - 1)
            cutPlaneNormal[2] = (minH + random.random() * (maxH - minH)) * (2*random.randrange(2) - 1)

            cutPlaneAnchor = cutPlaneNormal + center
            cutPlaneNormal = cutPlaneNormal/np.linalg.norm(cutPlaneNormal)

            return cutPlaneAnchor, cutPlaneNormal

        finalNumberOfAtoms = sum(numberOfAtomsOfEachKind)
        self.rectangularPrism(w, l, h)

        currentAtoms = set(self.atoms)
        MAX_CUTTING_ATTEMPTS = 50
        currentCuttingAttempt = 0

        while len(currentAtoms) > finalNumberOfAtoms and currentCuttingAttempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            #cutPlaneAnchor, cutPlaneNormal = drawCutPlaneFromSphere(min(w, l, h)*0.9, min(w, l, h), self.boundingBox.getCenter())
            cutPlaneAnchor, cutPlaneNormal = drawCutPlaneFromRectangularPrism(w/2.*0.9, w/2, l/2.*0.9, l/2.,  h/2.*0.9, h/2., self.boundingBox.getCenter())

            # count atoms to be removed, if new Count >= final Number remove
            atomsToBeRemoved, atomsToBeKept = self.splitAtomsAlongPlane(cutPlaneAnchor, cutPlaneNormal, currentAtoms)

            if len(atomsToBeRemoved) != 0.0 and len(currentAtoms) - len(atomsToBeRemoved) >= finalNumberOfAtoms:
                currentAtoms = currentAtoms.difference(atomsToBeRemoved)
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
                cutPlaneAnchor = cutPlaneAnchor + cutPlaneNormal * self.lattice.latticeConstant
                atomsToBeKept, atomsToBeRemoved = self.splitAtomsAlongPlane(cutPlaneAnchor, cutPlaneNormal, currentAtoms)

            # remove atoms till the final number is reached "from the ground up"

            # TODO implement sorting prioritzing the different directions in random order
            def sortByPosition(atom):
                return self.lattice.getLatticePositionFromIndex(atom.latticeIndex)[0]

            atomsToBeRemoved = list(atomsToBeRemoved)
            atomsToBeRemoved.sort(key=sortByPosition)
            atomsToBeRemoved = atomsToBeRemoved[:numberOfAtomsYetToBeRemoved]

            atomsToBeRemoved = set(atomsToBeRemoved)
            currentAtoms = currentAtoms.difference(atomsToBeRemoved)

        # redistribute the different elements randomly
        newAtomicSymbols = list()
        for index, symbol in enumerate(atomicSymbols):
            for i in range(numberOfAtomsOfEachKind[index]):
                newAtomicSymbols.append(symbol)

        random.shuffle(newAtomicSymbols)

        self.atoms.clear()
        _, latticeIndices, _ = zip(*currentAtoms)
        for listIndex, latticeIndex in enumerate(latticeIndices):
            self.atoms.append(self.Atom(listIndex, latticeIndex, newAtomicSymbols[listIndex]))

        self.findBoundingBox()
        self.constructNeighborList()

    def findBoundingBox(self):
        minCoordinates = np.array([1e10, 1e10, 1e10])
        maxCoordinates = np.array([-1e10, -1e10, -1e10])

        for atom in self.atoms:
            curPosition = self.lattice.getCartesianPositionFromIndex(atom.latticeIndex)
            for coordinate in range(3):
                if curPosition[coordinate] < minCoordinates[coordinate]:
                    minCoordinates[coordinate] = curPosition[coordinate]
                if curPosition[coordinate] > maxCoordinates[coordinate]:
                    maxCoordinates[coordinate] = curPosition[coordinate]

        w = maxCoordinates[0] - minCoordinates[0]
        l = maxCoordinates[1] - minCoordinates[1]
        h = maxCoordinates[2] - minCoordinates[2]

        self.boundingBox = BoundingBox.BoundingBox(w, l, h, minCoordinates)

    def constructNeighborList(self):
        _, latticeIndices, _ = zip(*self.atoms)

        for atom in self.atoms:
            position = self.lattice.getLatticePositionFromIndex(atom.latticeIndex)
            neighbors = set()
            for xOffset in [-1, 0, 1]:
                for yOffset in [-1, 0, 1]:
                    for zOffset in [-1, 0, 1]:
                        if xOffset is yOffset is zOffset is 0:
                            continue
                        offset = np.array([xOffset, yOffset, zOffset])

                        if self.lattice.isValidLatticePosition(position + offset):
                            index = self.lattice.getIndexFromLatticePosition(position + offset)

                            if index in latticeIndices:
                                neighbors.add(index)

            self.neighborList.append(neighbors)

    def getInnerAtoms(self):
        innerCoordinationNumbers = [12]
        return self.getAtomsFromCoordinationNumbers(innerCoordinationNumbers)

    def getSurfaceAtoms(self):
        surfaceCoordinationNumbers = [8, 9]
        return self.getAtomsFromCoordinationNumbers(surfaceCoordinationNumbers)

    def getCornerAtoms(self):
        cornerCoordinationNumbers = [1, 2, 3, 4]
        return self.getAtomsFromCoordinationNumbers(cornerCoordinationNumbers)

    def getEdgeAtoms(self):
        edgeCoordinationNumbers = [5, 6, 7]
        return self.getAtomsFromCoordinationNumbers(edgeCoordinationNumbers)

    def getTerraceAtoms(self):
        terraceCoordinationNumbers = [8, 9, 10, 11]
        return self.getAtomsFromCoordinationNumbers(terraceCoordinationNumbers)

    def getAtomsFromCoordinationNumbers(self, coordinationNumbers):
        return [self.atoms[i] for i in range(len(self.atoms)) if self.getCoordinationNumber(self.atoms[i]) in coordinationNumbers]

    def getCoordinationNumber(self, atom):
        return len(self.neighborList[atom.listIndex])

    def getLatticePositions(self):
        positions = list()
        for atom in self.atoms:
            positions.append(self.lattice.getLatticPositionFromIndex(atom.latticeIndex))

        return positions

    def getCartesianPositions(self):
        positions = list()
        for atom in self.atoms:
            positions.append(self.lattice.getCartesianPositionFromIndex(atom.latticeIndex))

        return positions

    def getASEAtoms(self, centered=True):
        atomPositions = self.getCartesianPositions()
        _, _, atomicSymbols = zip(*self.atoms)
        if centered:
            centerOfMass = np.array([0, 0, 0])
            for position in atomPositions:
                centerOfMass = centerOfMass + position
                centerOfMass = centerOfMass / len(atomPositions)

            a, b, atomicSymbols = zip(*self.atoms)
            return Atoms(positions=(atomPositions - centerOfMass), symbols=atomicSymbols)
        else:
            return Atoms(positions=atomPositions, symbols=atomicSymbols)
