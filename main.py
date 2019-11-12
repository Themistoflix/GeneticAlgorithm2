import Nanoparticle as NP
import FCCLattice as FCC
import CuttingPlaneGenerator as CPG
import CuttingPlaneOperator as CPO

from ase.visualize import view
import numpy as np
import random

if __name__ == '__main__':
    random.seed(23)
    lattice = FCC.FCCLattice(9, 9, 9, 2)

    particle = NP.Nanoparticle(lattice)
    particle.kozlovSphere(3, 'X', 1.0)

