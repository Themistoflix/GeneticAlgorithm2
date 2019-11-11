import Nanoparticle as NP


class CuttingPlaneOperator:
    def __init__(self, cuttingPlaneGenerator):
        self.cuttingPlaneGenerator = cuttingPlaneGenerator

    def cutAndReassembleNanoparticles(self, particle1, particle2):
        self.cuttingPlaneGenerator.setCenter(particle1.boundingBox.getCenter())
        cuttingPlane = self.cuttingPlaneGenerator.generateNewCuttingPlane()
        commonLattice = particle1.lattice

        atomIndicesInPositiveSubspace, _ = particle1.splitAtomIndicesAlongPlane(cuttingPlane)
        _, atomIndicesInNegativeSubspace = particle2.splitAtomIndicesAlongPlane(cuttingPlane)

        newAtomsData = {**particle1.getAtoms(atomIndicesInPositiveSubspace), **particle2.getAtoms(atomIndicesInNegativeSubspace)}
        newParticle = NP.Nanoparticle(commonLattice)
        newParticle.fromParticleData(newAtomsData)

        oldStoichiometry = particle1.getStoichiometry()
        newStoichiometry = newParticle.getStoichiometry()

        if newStoichiometry == oldStoichiometry:
            return newParticle
        else:
            newParticle.enforceStoichiometry(oldStoichiometry)
            return newParticle


