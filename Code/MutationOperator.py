import numpy as np
from Code import Nanoparticle as NP

class MutationOperator:
    def __init__(self, maxExchanges):
        self.maxExchanges = maxExchanges
        self.probabilityDistribution = np.array([1./(n**(3./2.)) for n in range(1, maxExchanges + 1, 1)])
        self.probabilityDistribution = self.probabilityDistribution/np.sum(self.probabilityDistribution)

    def swapAtomicSymbols(self, atoms, atomIndexList1, atomIndexlist2):
        for index1, index2 in zip(atomIndexList1, atomIndexlist2):
            index1Symbol = atoms[index1]
            atoms[index1] = atoms[index2]
            atoms[index2] = index1Symbol

    def mutateNanoparticle(self, particle):
        commonLattice = particle.lattice
        atoms = particle.getAtoms()
        stoichiometry = particle.getStoichiometry()

        symbol1 = list(stoichiometry.keys())[0]

        symbol1List = list()
        symbol2List = list()

        for atom in atoms:
            if atoms[atom] == symbol1:
                symbol1List.append(atom)
            else:
                symbol2List.append(atom)

        numberOfExchanges = 1 + np.random.choice(self.maxExchanges, p=self.probabilityDistribution)

        symbol1Swaps = np.random.choice(symbol1List, numberOfExchanges, replace=False)
        symbol2Swaps = np.random.choice(symbol2List, numberOfExchanges, replace=False)

        self.swapAtomicSymbols(atoms, symbol1Swaps, symbol2Swaps)
        newParticle = NP.Nanoparticle(commonLattice)
        newParticle.fromParticleData(atoms, particle.getNeighborList())

        return newParticle
