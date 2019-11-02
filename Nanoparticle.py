import numpy as np
from ase import Atoms

import FCCLattice


class Nanoparticle:
    def __init__(self, lattice):
        self.lattice = lattice

        self.indicesOnLattice = list()
        self.atomicNumbers = list()

    def cube(self, w, l, h, symbol='X'):
        anchorPoint = self.lattice.getAnchorOfCenteredBox(w, l, h)
        for x in range(w + 1):
            for y in range(l + 1):
                for z in range(h + 1):
                    curPosition = anchorPoint + np.array([x, y, z])

                    if self.lattice.isValidPosition(curPosition):
                        index = self.lattice.getIndexFromPosition(curPosition)

                        self.indicesOnLattice.append(index)
                        self.atomicNumbers.append(symbol)

    def getPositionsOnLattice(self):
        positions = list()
        for atomIndex in self.indicesOnLattice:
            positions.append(self.lattice.getCartesianPositionFromIndex(atomIndex))

        return positions

    def getAtoms(self):
        atomPositions = self.getPositionsOnLattice()
        return Atoms(positions=atomPositions, symbols=self.atomicNumbers)
