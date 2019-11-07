import random
import Nanoparticle as NP


class CuttingPlaneOperator:
    def __init__(self, cuttingPlaneGenerator):
        self.cuttingPlaneGenerator = cuttingPlaneGenerator

    def cutAndReassembleNanoparticles(self, particle1, particle2):
        cuttingPlane = self.cuttingPlaneGenerator.generateNewCuttingPlane()
        commonLattice = particle1.lattice

        atomsInPositiveSubspace, _ = particle1.splitAtomsAlongPlane(cuttingPlane)
        _, atomsInNegativeSubspace = particle2.splitAtomsAlongPlane(cuttingPlane)

        newAtomsData = list(atomsInPositiveSubspace) + list(atomsInNegativeSubspace)
        newParticle = NP.Nanoparticle(commonLattice)
        newParticle.fromParticleData(newAtomsData)

        return newParticle

