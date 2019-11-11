import Nanoparticle as NP
import FCCLattice as FCC
import CuttingPlaneGenerator as CPG
import CuttingPlaneOperator as CPO

from ase.visualize import view
import numpy as np
import random

if __name__ == '__main__':
    random.seed(23)
    lattice = FCC.FCCLattice(15, 15, 15, 2)

    cuttingPlaneGenerator = CPG.SphericalCuttingPlaneGenerator(6., 9.)
    # particle.rectangularPrism(17, 17, 17, 'Cu')
    particle1 = NP.Nanoparticle(lattice)
    particle1.convexShape([100, 100], ['Ag', 'Cu'], 9, 9, 9, cuttingPlaneGenerator)

    particle2 = NP.Nanoparticle(lattice)
    particle2.convexShape([100, 100], ['Ag', 'Cu'], 9, 9, 9, cuttingPlaneGenerator)

    crossoverCuttingPlaneGenerator = CPG.SphericalCuttingPlaneGenerator(0., 3., particle2.boundingBox.getCenter())
    cuttingPlaneOperator = CPO.CuttingPlaneOperator(crossoverCuttingPlaneGenerator)

    particle3 = cuttingPlaneOperator.cutAndReassembleNanoparticles(particle1, particle2)
    print(particle3.getStoichiometry())
    print(len(particle3.atoms))
