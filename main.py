import Nanoparticle as NP
import FCCLattice as FCC
import random

if __name__ == '__main__':
    random.seed(1)
    lattice = FCC.FCCLattice(11, 11, 11, 2)
    particle = NP.Nanoparticle(lattice)
    particle.rectangularPrism(5, 5, 5, 'Cu')

    coreParticle = NP.Nanoparticle(lattice)
    coreParticle.fromParticleData(particle.getInnerAtoms())


