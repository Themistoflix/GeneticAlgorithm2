import numpy as np
import copy

from ase import Atoms
from ase.optimize import BFGS
from asap3 import EMT

from Code import BoundingBox
from Code import IndexedAtoms


class Nanoparticle:

    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = IndexedAtoms.IndexedAtoms()
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
                        self.atoms.addAtoms([(latticeIndex, symbol)])

        self.findBoundingBox()
        self.constructNeighborList()

    def kozlovSphere(self, height, symbols, ratio):
        boundingBoxAnchor = self.lattice.getAnchorIndexOfCenteredBox(2*height, 2*height, 2*height)
        lowerTipPosition = boundingBoxAnchor + np.array([height, height, 0])

        if not self.lattice.isValidLatticePosition(lowerTipPosition):
            lowerTipPosition[2] = lowerTipPosition[2] + 1

        layerBasisVector1 = np.array([1, 1, 0])
        layerBasisVector2 = np.array([-1, 1, 0])
        for zPosition in range(height):
            layerWidth = zPosition + 1
            lowerLayerOffset = np.array([0, -zPosition, zPosition])
            upperLayerOffset = np.array([0, -zPosition, 2*height - 2 - zPosition])

            lowerLayerStartPosition = lowerTipPosition + lowerLayerOffset
            upperLayerStartPosition = lowerTipPosition + upperLayerOffset
            for width in range(layerWidth):
                for length in range(layerWidth):
                    currentPositionLowerLayer = lowerLayerStartPosition + width*layerBasisVector1 + length*layerBasisVector2
                    currentPositionUpperLayer = upperLayerStartPosition + width*layerBasisVector1 + length*layerBasisVector2

                    lowerLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionLowerLayer)
                    upperLayerIndex = self.lattice.getIndexFromLatticePosition(currentPositionUpperLayer)

                    self.atoms.addAtoms([(lowerLayerIndex, 'X'), (upperLayerIndex, 'X')])

        self.constructNeighborList()
        corners = self.getAtomIndicesFromCoordinationNumbers([4])

        self.removeAtoms(corners)
        self.findBoundingBox()

        totalNumberOfAtoms = self.atoms.getCount()
        numberOfAtomsWithSymbol1 = int(totalNumberOfAtoms*ratio)
        numberOfAtomsWithSymbol2 = totalNumberOfAtoms - numberOfAtomsWithSymbol1

        self.randomChemicalOrdering(symbols, [numberOfAtomsWithSymbol1, numberOfAtomsWithSymbol2])

    def fromParticleData(self, atoms, neighborList=None, boundingBox=None):
        self.atoms = atoms
        if boundingBox is None:
            self.findBoundingBox()
        else:
            self.boundingBox = boundingBox
        if neighborList is None:
            self.constructNeighborList()
        else:
            self.neighborList = neighborList

    def randomChemicalOrdering(self, symbols, atomsOfEachKind):
        self.atoms.randomChemicalOrdering(symbols, atomsOfEachKind)

    def splitAtomIndicesAlongPlane(self, cuttingPlane, atomIndices=None):
        if atomIndices is None:
            atomIndices = self.atoms.getIndices()

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
        indicesOfCurrentAtoms = set(self.atoms.getIndices())

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
        self.atoms.clear()
        self.atoms.addAtoms(zip(indicesOfCurrentAtoms, ['X']*len(indicesOfCurrentAtoms)))
        self.atoms.randomChemicalOrdering(atomicSymbols, numberOfAtomsOfEachKind)

        self.findBoundingBox()
        self.constructNeighborList()

    def findBoundingBox(self):
        minCoordinates = np.array([1e10, 1e10, 1e10])
        maxCoordinates = np.array([-1e10, -1e10, -1e10])

        for latticeIndex in self.atoms.getIndices():
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
        for latticeIndex in self.atoms.getIndices():
            nearestLatticeNeighbors = self.lattice.getNearestNeighbors(latticeIndex)
            nearestNeighbors = set()
            for neighbor in nearestLatticeNeighbors:
                if neighbor in self.atoms.getIndices():
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

    def getNumberOfHeteroatomicBonds(self):
        numberOfHeteroatomicBonds = 0

        if len(self.atoms.getSymbols()) == 2:
            symbol = self.atoms.getSymbols()[0]
        else:
            return 0

        for latticeIndexWithSymbol in self.atoms.getIndicesBySymbol(symbol):
            neighborList = self.neighborList[latticeIndexWithSymbol]
            for neighbor in neighborList:
                symbolOfNeighbor = self.atoms.getSymbol(neighbor)

                if symbol != symbolOfNeighbor:
                    numberOfHeteroatomicBonds = numberOfHeteroatomicBonds + 1

        return numberOfHeteroatomicBonds

    def getAtomIndicesFromCoordinationNumbers(self, coordinationNumbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers, self.atoms.getIndices()))
        else:
            return list(filter(lambda x: self.getCoordinationNumber(x) in coordinationNumbers and self.atoms.getSymbol(x) == symbol, self.atoms.getIndices()))

    def getCoordinationNumber(self, latticeIndex):
        return len(self.neighborList[latticeIndex])

    def getAtoms(self, atomIndices=None):
        if atomIndices is None:
            return self.atoms.getDeepcopy()
        else:
            # TODO: not safe yet
            atoms = IndexedAtoms.IndexedAtoms()
            symbols = [self.atoms.getSymbol(index) for index in atomIndices]

            atoms.addAtoms(zip(atomIndices, symbols))

            return copy.deepcopy(atoms)

    def getNeighborList(self):
        return self.neighborList.copy()

    def getBoundingBox(self):
        return copy.deepcopy(self.boundingBox)

    def getASEAtoms(self, centered=True):
        atomPositions = list()
        atomicSymbols = list()
        for latticeIndex in self.atoms.getIndices():
            atomPositions.append(self.lattice.getCartesianPositionFromIndex(latticeIndex))
            atomicSymbols.append(self.atoms.getSymbol(latticeIndex))

        atoms = Atoms(positions=atomPositions, symbols=atomicSymbols)
        if centered:
            COM = Atoms.get_center_of_mass()
            return Atoms(positions=[position - COM for position in atomPositions], symbols=atomicSymbols)
        else:
            return Atoms(positions=atomPositions, symbols=atomicSymbols)

    def getPotentialEnergyPerAtom(self):
        atoms = self.getASEAtoms()
        atoms.set_cell(np.array([[self.boundingBox.width, 0, 0], [0, self.boundingBox.length, 0], [0, 0, self.boundingBox.height]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=10)

        return atoms.get_potential_energy()/self.atoms.getCount()

    def getKozlovParameters(self, symbol):
        # coordination numbers from Kozlov et al. 2015
        coordinationNumberCornerAtoms = [1, 2, 3, 4, 5, 6]
        coordinationNumberEdgeAtoms = [7]
        coordinationNumberTerraceAtoms = [9]

        cornerAtoms = self.getAtomIndicesFromCoordinationNumbers(coordinationNumberCornerAtoms, symbol)
        edgeAtoms = self.getAtomIndicesFromCoordinationNumbers(coordinationNumberEdgeAtoms, symbol)
        terraceAtoms = self.getAtomIndicesFromCoordinationNumbers(coordinationNumberTerraceAtoms, symbol)

        numCornerAtoms = len(cornerAtoms)
        numEdgeAtoms = len(edgeAtoms)
        numTerraceAtoms = len(terraceAtoms)
        numHeteroatomicBonds = self.getNumberOfHeteroatomicBonds()

        return np.array([numHeteroatomicBonds, numCornerAtoms, numEdgeAtoms, numTerraceAtoms])

    def printKozlovParameters(self, symbol):
        kozlovParameters = self.getKozlovParameters(symbol)
        print("number of heteroatomic bonds: {0}".format(kozlovParameters[0]))
        print("number of {0} corner atoms: {1}".format(symbol, kozlovParameters[1]))
        print("number of {0} edge atoms: {1}".format(symbol, kozlovParameters[2]))
        print("number of {0} terrace atoms: {1}".format(symbol, kozlovParameters[3]))

    def getKozlovEnergy(self, descriptors, symbol):
        return np.dot(descriptors, self.getKozlovParameters(symbol))

    def getStoichiometry(self):
        return self.atoms.getStoichiometry()

    def enforceStoichiometry(self, stoichiometry):
        atomNumberDifference = self.atoms.getCount() - sum(stoichiometry.values())

        if atomNumberDifference > 0:
            self.removeUndercoordinatedAtoms(atomNumberDifference)

        elif atomNumberDifference < 0:
            self.fillHighCoordinatedSurfaceVacancies(-atomNumberDifference)

        if self.getStoichiometry() != stoichiometry:
            self.adjustAtomicRatios(stoichiometry)

    def removeAtoms(self, latticeIndices):
        self.atoms.removeAtoms(latticeIndices)

        self.findBoundingBox()
        self.constructNeighborList()

    def addAtoms(self, atoms):
        self.atoms.addAtoms(atoms)

        self.findBoundingBox()
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
                symbols.append(np.random.choice(self.atoms.getSymbols()))

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
            indicesSymbolFrom = self.atoms.getIndicesBySymbol(elementFrom)
            index = np.random.choice(indicesSymbolFrom)
            self.atoms.transformAtom(index, elementTo)

        # assuming bimetallic particles
        ownStoichiometry = self.getStoichiometry()
        element1 = self.atoms.getSymbols()[0]
        element2 = self.atoms.getSymbols()[1]

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
            if latticeIndex in self.atoms.getIndices():
                atomicNeighbors = atomicNeighbors + 1

        return atomicNeighbors






