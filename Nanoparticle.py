import numpy as np
import random
import copy

from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT

import BoundingBox


class Nanoparticle:

    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = dict()
        self.boundingBox = BoundingBox.BoundingBox(0, 0, 0, np.array([0, 0, 0]))
        self.neighborList = dict()

    def rectangularPrism(self, w, l, h, symbol='X'):
        anchorPoint = self.lattice.getAnchorIndexOfCenteredBox(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    curPosition = anchorPoint + np.array([x, y, z])

                    if self.lattice.isValidLatticePosition(curPosition):
                        latticeIndex = self.lattice.getIndexFromLatticePosition(curPosition)
                        self.atoms[latticeIndex] = symbol

        self.findBoundingBox()
        self.constructNeighborList()

    def fromParticleData(self, atoms):
        self.atoms = atoms
        self.findBoundingBox()
        self.constructNeighborList()

    def splitAtomIndicesAlongPlane(self, cuttingPlane, atomIndices=None):
        if atomIndices is None:
            atomIndices = self.atoms

        atomsInPositiveSubspace = set()
        atomsInNegativeSubspace = set()

        for latticeIndex in atomIndices:
            position = self.lattice.getCartesianPositionFromIndex(latticeIndex)
            if np.dot((position - cuttingPlane.anchor), cuttingPlane.normal) >= 0.0:
                atomsInPositiveSubspace.add(latticeIndex)
            else:
                atomsInNegativeSubspace.add(latticeIndex)
        return atomsInPositiveSubspace, atomsInNegativeSubspace

    def convexShape(self, numberOfAtomsOfEachKind, atomicSymbols, w, l, h, cuttingPlaneGenerator):
        self.rectangularPrism(w, l, h)
        indicesOfCurrentAtoms = set(self.atoms.keys())

        finalNumberOfAtoms = sum(numberOfAtomsOfEachKind)
        MAX_CUTTING_ATTEMPTS = 50
        currentCuttingAttempt = 0
        cuttingPlaneGenerator.setCenter(self.boundingBox.getCenter())

        while len(indicesOfCurrentAtoms) > finalNumberOfAtoms and currentCuttingAttempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            cuttingPlane = cuttingPlaneGenerator.generateNewCuttingPlane()

            # count atoms to be removed, if new Count >= final Number remove
            atomsToBeRemoved, atomsToBeKept = self.splitAtomIndicesAlongPlane(cuttingPlane, indicesOfCurrentAtoms)
            if len(atomsToBeRemoved) != 0.0 and len(indicesOfCurrentAtoms) - len(atomsToBeRemoved) >= finalNumberOfAtoms:
                indicesOfCurrentAtoms = indicesOfCurrentAtoms.difference(atomsToBeRemoved)
                currentCuttingAttempt = 0
            else:
                currentCuttingAttempt = currentCuttingAttempt + 1

        if currentCuttingAttempt == MAX_CUTTING_ATTEMPTS:
            # place cutting plane parallel to one of the axes and at the anchor point
            cuttingPlane = cuttingPlaneGenerator.createAxisParallelCuttingPlane(self.boundingBox.position)

            # shift till too many atoms would get removed
            numberOfAtomsYetToBeRemoved = len(indicesOfCurrentAtoms) - finalNumberOfAtoms
            atomsToBeRemoved = set()
            while len(atomsToBeRemoved) < numberOfAtomsYetToBeRemoved:
                cuttingPlane = cuttingPlane._replace(anchor=cuttingPlane.anchor + cuttingPlane.normal * self.lattice.latticeConstant)
                atomsToBeKept, atomsToBeRemoved = self.splitAtomIndicesAlongPlane(cuttingPlane, indicesOfCurrentAtoms)

            # remove atoms till the final number is reached "from the ground up"

            # TODO implement sorting prioritzing the different directions in random order
            def sortByPosition(atom):
                return self.lattice.getLatticePositionFromIndex(atom)[0]

            atomsToBeRemoved = list(atomsToBeRemoved)
            atomsToBeRemoved.sort(key=sortByPosition)
            atomsToBeRemoved = atomsToBeRemoved[:numberOfAtomsYetToBeRemoved]

            atomsToBeRemoved = set(atomsToBeRemoved)
            indicesOfCurrentAtoms = indicesOfCurrentAtoms.difference(atomsToBeRemoved)

        # redistribute the different elements randomly
        newAtomicSymbols = list()
        for index, symbol in enumerate(atomicSymbols):
            for i in range(numberOfAtomsOfEachKind[index]):
                newAtomicSymbols.append(symbol)

        random.shuffle(newAtomicSymbols)

        self.atoms.clear()
        for symbolIndex, latticeIndex in enumerate(indicesOfCurrentAtoms):
            self.atoms[latticeIndex] = newAtomicSymbols[symbolIndex]

        self.findBoundingBox()
        self.constructNeighborList()

    def findBoundingBox(self):
        minCoordinates = np.array([1e10, 1e10, 1e10])
        maxCoordinates = np.array([-1e10, -1e10, -1e10])

        for latticeIndex in self.atoms:
            curPosition = self.lattice.getCartesianPositionFromIndex(latticeIndex)
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
        for latticeIndex in self.atoms:
            nearestLatticeNeighbors = self.lattice.getNearestNeighbors(latticeIndex)
            nearestNeighbors = set()
            for neighbor in nearestLatticeNeighbors:
                if neighbor in self.atoms:
                    nearestNeighbors.add(neighbor)

            self.neighborList[latticeIndex] = nearestNeighbors

    def getCornerAtomIndices(self, symbol=None):
        cornerCoordinationNumbers = [1, 2, 3, 4]
        return self.getAtomIndicesFromCoordinationNumbers(cornerCoordinationNumbers, symbol)

    def getEdgeIndices(self, symbol=None):
        edgeCoordinationNumbers = [5, 6, 7]
        return self.getAtomIndicesFromCoordinationNumbers(edgeCoordinationNumbers, symbol)

    def getSurfaceAtomIndices(self, symbol=None):
        surfaceCoordinationNumbers = [8, 9]
        return self.getAtomIndicesFromCoordinationNumbers(surfaceCoordinationNumbers, symbol)

    def getTerraceAtomIndices(self, symbol=None):
        terraceCoordinationNumbers = [10, 11]
        return self.getAtomIndicesFromCoordinationNumbers(terraceCoordinationNumbers, symbol)

    def getInnerAtomIndices(self, symbol=None):
        innerCoordinationNumbers = [12]
        return self.getAtomIndicesFromCoordinationNumbers(innerCoordinationNumbers, symbol)

    def getNumberOfHeteroatomicBonds(self, symbol1, symbol2):
        numberOfHeteroatomicBonds = 0
        for latticeIndex in self.atoms:
            neighborList = self.neighborList[latticeIndex]
            for neighbor in neighborList:
                symbolA = self.atoms[latticeIndex]
                symbolB = self.atoms[neighbor]

                if symbol1 == symbolA and symbol2 == symbolB or symbol1 == symbolB and symbol2 == symbolA:
                    numberOfHeteroatomicBonds = numberOfHeteroatomicBonds + 1

        return numberOfHeteroatomicBonds/2

    def getAtomIndicesFromCoordinationNumbers(self, coordinationNumbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers, self.atoms))
        else:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers and self.atoms[x] == symbol, self.atoms))

    def getCoordinationNumber(self, latticeIndex):
        return len(self.neighborList[latticeIndex])

    def getAtoms(self, atomIndices=None):
        if atomIndices is None:
            return dict(self.atoms)
        else:
            atoms = dict()
            for index in atomIndices:
                atoms[index] = self.atoms[index]

            return atoms.copy()

    def getASEAtoms(self, centered=True):
        atomPositions = list()
        atomicSymbols = list()
        for latticeIndex, atomicSymbol in self.atoms.items():
            atomPositions.append(self.lattice.getCartesianPositionFromIndex(latticeIndex))
            atomicSymbols.append(atomicSymbol)

        if centered:
            centerOfMass = np.array([0, 0, 0])
            for position in atomPositions:
                centerOfMass = centerOfMass + position
                centerOfMass = centerOfMass / len(atomPositions)

            return Atoms(positions=(atomPositions - centerOfMass), symbols=atomicSymbols)
        else:
            return Atoms(positions=atomPositions, symbols=atomicSymbols)

    def getPotentialEnergyPerAtom(self):
        atoms = self.getASEAtoms()
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=10)

        return atoms.get_potential_energy()/len(self.atoms)

    def getStoichiometry(self):
        stoichiometry = dict()
        for symbol in self.atoms.values():
            if symbol in stoichiometry.keys():
                stoichiometry[symbol] = stoichiometry[symbol] + 1
            else:
                stoichiometry[symbol] = 1

        return stoichiometry

    def enforceStoichiometry(self, stoichiometry):
        atomNumberDifference = len(self.atoms) - sum(stoichiometry.values())

        if atomNumberDifference > 0:
            self.removeUndercoordinatedAtoms(atomNumberDifference)

        elif atomNumberDifference < 0:
            self.fillHighCoordinatedSurfaceVacancies(-atomNumberDifference)

        if self.getStoichiometry() != stoichiometry:
            self.adjustAtomicRatios(stoichiometry)

    def removeAtoms(self, latticeIndices):
        for index in latticeIndices:
            self.atoms.pop(index)
        self.constructNeighborList()

    def addAtoms(self, atoms):
        for atom in atoms:
            self.atoms[atom[0]] = atom[1]
        self.constructNeighborList()

    def fillHighCoordinatedSurfaceVacancies(self, count):
        atomsYetToBeAdded = count
        while atomsYetToBeAdded != 0:
            notFullyCoordinatedAtoms = self.getAtomIndicesFromCoordinationNumbers(range(self.lattice.MAX_NEIGHBORS))
            surfaceVacancies = set()

            for atom in notFullyCoordinatedAtoms:
                neighborVacancies = self.lattice.getNearestNeighbors(atom).difference(self.neighborList[atom])
                surfaceVacancies = surfaceVacancies.union(neighborVacancies)

            # sort
            surfaceVacancies = list(surfaceVacancies)
            surfaceVacancies.sort(key=self.getNumberOfAtomicNeighbors)
            # add max(vacancies with max coord number, atoms yet to be removed
            atomsToBeAdded = list()
            maxCoordinationNumber = self.getNumberOfAtomicNeighbors(surfaceVacancies[-1])
            currentCoordinationNumber = maxCoordinationNumber
            index = -1
            while len(atomsToBeAdded) < atomsYetToBeAdded and currentCoordinationNumber == maxCoordinationNumber:
                atomsToBeAdded.append(surfaceVacancies[index])
                index = index - 1
                currentCoordinationNumber = self.getNumberOfAtomicNeighbors(surfaceVacancies[index])
            # choose elements according to current distribution
            symbols = list()
            for atom in atomsToBeAdded:
                symbols.append(np.random.choice(list(self.atoms.values())))

            self.addAtoms(zip(atomsToBeAdded, symbols))
            atomsYetToBeAdded = atomsYetToBeAdded - len(atomsToBeAdded)

    def removeUndercoordinatedAtoms(self, count):
        atomsYetToBeRemoved = count
        while atomsYetToBeRemoved != 0:
            mostUndercoordinatedAtoms = self.getAtomIndicesFromCoordinationNumbers(range(9))
            mostUndercoordinatedAtoms.sort(key=self.getCoordinationNumber)

            atomsToBeRemoved = list()
            minCoordinationNumber = len(self.neighborList[mostUndercoordinatedAtoms[0]])
            currentCoordinationNumber = minCoordinationNumber
            index = 0
            while len(atomsToBeRemoved) < atomsYetToBeRemoved and currentCoordinationNumber == minCoordinationNumber:
                atomsToBeRemoved.append(mostUndercoordinatedAtoms[index])
                index = index + 1
                currentCoordinationNumber = self.getCoordinationNumber(mostUndercoordinatedAtoms[index])

            self.removeAtoms(atomsToBeRemoved)
            atomsYetToBeRemoved = atomsYetToBeRemoved - len(atomsToBeRemoved)

        return

    def adjustAtomicRatios(self, stoichiometry):
        def randomlyTransfromFrom1to2(elementFrom, elementTo):
            indices = np.random.permutation(list(self.atoms.keys()))

            index = 0
            randomLatticeIndex = indices[index]
            while self.atoms[randomLatticeIndex] != elementFrom:
                index = index + 1
                randomLatticeIndex = indices[index]
            self.atoms[randomLatticeIndex] = elementTo

        # assuming bimetallic paricles
        ownStoichiometry = self.getStoichiometry()
        element1 = list(ownStoichiometry.keys())[0]
        element2 = list(ownStoichiometry.keys())[1]

        numberOfOperations = (ownStoichiometry[element1] - stoichiometry[element1])

        for sucessfulSwaps in range(np.abs(numberOfOperations)):
            if numberOfOperations < 0:
                randomlyTransfromFrom1to2(element2, element1)
            else:
                randomlyTransfromFrom1to2(element1, element2)

        return

    def getNumberOfAtomicNeighbors(self, index):
        atomicNeighbors = 0
        nearestNeighbors = self.lattice.getNearestNeighbors(index)
        for latticeIndex in nearestNeighbors:
            if latticeIndex in self.atoms:
                atomicNeighbors = atomicNeighbors + 1

        return atomicNeighbors






