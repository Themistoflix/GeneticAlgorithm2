import numpy as np
from Code import Nanoparticle as NP


class MutationOperator:
    def __init__(self, maxExchanges):
        self.maxExchanges = maxExchanges
        self.probabilityDistribution = np.array([1./(n**(3./2.)) for n in range(1, maxExchanges + 1, 1)])
        self.probabilityDistribution = self.probabilityDistribution/np.sum(self.probabilityDistribution)

    def mutateNanoparticle(self, particle):
        commonLattice = particle.lattice
        atoms = particle.getAtoms()

        symbol1 = atoms.getSymbols()[0]
        symbol2 = atoms.getSymbols()[1]

        numberOfExchanges = 1 + np.random.choice(self.maxExchanges, p=self.probabilityDistribution)

        symbol1Indices = np.random.choice(atoms.getIndicesBySymbol(symbol1), numberOfExchanges, replace=False)
        symbol2Indices = np.random.choice(atoms.getIndicesBySymbol(symbol2), numberOfExchanges, replace=False)

        atoms.swapAtoms(zip(symbol1Indices, symbol2Indices))

        newParticle = NP.Nanoparticle(commonLattice)
        newParticle.fromParticleData(atoms, particle.getNeighborList(), particle.getBoundingBox())

        return newParticle
