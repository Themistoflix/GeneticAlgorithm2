import Nanoparticle as NP
import FCCLattice as FCC
import random

if __name__ == '__main__':
    random.seed(1)
    lattice = FCC.FCCLattice(23, 23, 23, 2)
    particle = NP.Nanoparticle(lattice)
    particle.convexShape([300, 300], ['Cu', 'Ag'], 15, 15, 15)

