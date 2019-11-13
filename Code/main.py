from Code import FCCLattice as FCC, Nanoparticle as NP, MutationOperator as MO

import random
import numpy as np

if __name__ == '__main__':
    np.random.seed(2348)
    lattice = FCC.FCCLattice(9, 9, 9, 2)

    startParticle = NP.Nanoparticle(lattice)
    ratio = 0.5
    startParticle.kozlovSphere(6, ['Pd', 'Au'], ratio)

    Pd_atoms = startParticle.getStoichiometry()['Pd']
    Au_atoms = startParticle.getStoichiometry()['Au']

    kozlovSymbol = 'Au'
    energies = list()
    descriptors = np.array([-13, -404, -301, -200]).transpose()
    oldEnergy = startParticle.getKozlovEnergy(descriptors, kozlovSymbol)
    energies.append(oldEnergy)

    # maxAttemptsToEscapeMinimum = 10*Pd_atoms*(totalAtoms - Pd_atoms)
    maxAttemptsToEscapeMinimum = 3
    attemptsWithoutExchange = 0
    mutationOperator = MO.MutationOperator(min(Pd_atoms, Au_atoms))

    acceptanceRate = 0
    T = 1
    beta = 8.6e-2 * T
    totalAttempts = 0

    while attemptsWithoutExchange < maxAttemptsToEscapeMinimum:
        totalAttempts = totalAttempts + 1
        if totalAttempts % 1000 == 0:
            print(totalAttempts)

        attemptsWithoutExchange = attemptsWithoutExchange + 1

        newParticle = mutationOperator.mutateNanoparticle(startParticle)
        newEnergy = newParticle.getKozlovEnergy(descriptors, kozlovSymbol)

        deltaE = newEnergy - oldEnergy
        if deltaE < 0:
            acceptanceRate = 1
        else:
            acceptanceRate = np.exp(-beta * deltaE)
            print(oldEnergy, newEnergy, deltaE)
            print(acceptanceRate)

        if np.random.random() > 1 - acceptanceRate:
            attemptsWithoutExchange = 0
            energies.append(newEnergy)
            oldEnergy = newEnergy
            startParticle = newParticle

print(energies)


