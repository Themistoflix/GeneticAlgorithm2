import numpy as np
import copy

class IndexedAtoms:
    def __init__(self):
        self.symbolByIndex = dict()
        self.indicesBySymbol = dict()

    def getIndices(self):
        return list(self.symbolByIndex)

    def getSymbols(self):
        return list(self.indicesBySymbol)

    def getSymbol(self, index):
        return self.symbolByIndex[index]

    def getIndicesBySymbol(self, symbol):
        return self.indicesBySymbol[symbol]

    def getCount(self):
        return len(self.symbolByIndex)

    def clear(self):
        self.symbolByIndex.clear()
        self.indicesBySymbol.clear()

    def addAtoms(self, atoms):
        for atom in atoms:
            index = atom[0]
            symbol = atom[1]
            self.symbolByIndex[index] = symbol

            if symbol in self.indicesBySymbol:
                allAtomsOfOneKind = self.indicesBySymbol[symbol]
                allAtomsOfOneKind.append(index)
            else:
                allAtomsOfOneKind = list()
                allAtomsOfOneKind.append(index)
                self.indicesBySymbol[symbol] = allAtomsOfOneKind

    def removeAtoms(self, indices):
        for index in indices:
            symbol = self.symbolByIndex[index]

            self.symbolByIndex.pop(index)
            self.indicesBySymbol[symbol].remove(index)

    def swapAtoms(self, pairs):
        for pair in pairs:
            index1 = pair[0]
            index2 = pair[1]

            symbol1 = self.symbolByIndex[index1]
            symbol2 = self.symbolByIndex[index2]

            self.symbolByIndex[index1] = symbol2
            self.symbolByIndex[index2] = symbol1

            self.indicesBySymbol[symbol1].remove(index1)
            self.indicesBySymbol[symbol2].append(index1)

            self.indicesBySymbol[symbol2].remove(index2)
            self.indicesBySymbol[symbol1].append(index2)

    def randomChemicalOrdering(self, symbols, numberOfAtomsOfEachKind):
        newOrdering = list()
        for index, symbol in enumerate(symbols):
            for i in range(numberOfAtomsOfEachKind[index]):
                newOrdering.append(symbol)

        np.random.shuffle(newOrdering)

        self.indicesBySymbol.clear()

        for symbolIndex, atomIndex in enumerate(self.symbolByIndex):
            newSymbol = newOrdering[symbolIndex]
            self.symbolByIndex[atomIndex] = newSymbol
            if newSymbol in self.indicesBySymbol:
                self.indicesBySymbol[newSymbol].append(atomIndex)
            else:
                allAtomsOfOneKind = list()
                allAtomsOfOneKind.append(atomIndex)
                self.indicesBySymbol[newSymbol] = allAtomsOfOneKind

    def getStoichiometry(self):
        stoichiometry = dict()
        for symbol in self.indicesBySymbol:
            stoichiometry[symbol] = len(self.indicesBySymbol[symbol])

        return stoichiometry

    def transformAtom(self, index, newSymbol):
        oldSymbol = self.symbolByIndex[index]

        self.symbolByIndex[index] = newSymbol
        self.indicesBySymbol[oldSymbol].remove(index)
        self.indicesBySymbol[newSymbol].append(index)

    def getDeepcopy(self):
        copyAtoms = IndexedAtoms()
        copyAtoms.indicesBySymbol = copy.deepcopy(self.indicesBySymbol)
        copyAtoms.symbolByIndex = dict(self.symbolByIndex)

        return copyAtoms
