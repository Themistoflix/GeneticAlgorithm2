import numpy as np
from Code import Nanoparticle as NP


class MutationOperator:
    def __init__(self, maxExchanges):
        self.maxExchanges = maxExchanges
        self.probabilityDistribution = np.array([1./(n**(3./2.)) for n in range(1, maxExchanges + 1, 1)])
        self.probabilityDistribution = self.probabilityDistribution/np.sum(self.probabilityDistribution)

    def mutateNanoparticle(self, particle):
        symbol1 = particle.atoms.getSymbols()[0]
        symbol2 = particle.atoms.getSymbols()[1]

        numberOfExchanges = 1 + np.random.choice(self.maxExchanges, p=self.probabilityDistribution)

        symbol1Indices = np.random.choice(particle.atoms.getIndicesBySymbol(symbol1), numberOfExchanges, replace=False)
        symbol2Indices = np.random.choice(particle.atoms.getIndicesBySymbol(symbol2), numberOfExchanges, replace=False)

        particle.atoms.swapAtoms(zip(symbol1Indices, symbol2Indices))

        return particle, zip(symbol1Indices, symbol2Indices)

    def revertMutation(self, particle, swaps):
        symbol1Indices, symbol2Indices = zip(*swaps)
        particle.atoms.swapAtoms(zip(symbol2Indices, symbol1Indices))

        return particle
