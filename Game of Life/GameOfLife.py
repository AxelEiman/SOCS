import numpy as np
import matplotlib.pyplot as plt


def findAliveNeighbors(matrix):
    """
    Gives amount of alive neighbors for all cells
    :param cellPosition: 
    :return: 
    """
    matrix = np.pad(matrix, 1)
    shape = np.shape(matrix)
    nRows = shape[0]
    nCols = shape[1]
    nNeighbors = np.empty(shape)

    for row in range(nRows):
        if (row == 0) or row == nRows - 1:
            continue
        else:
            for col in range(nCols):
                if (col == 0) or col == nCols - 1:
                    continue
                else:
                    nNeighbors[row, col] = countNeighbors(row, col, matrix)
    return nNeighbors[1:-1, 1:-1]


def countNeighbors(r, c, matrix):
    rowSum = 0
    for row in [r - 1, r, r + 1]:
        rowSum += sum(matrix[row, col] for col in [c - 1, c, c + 1])

    return int(rowSum - matrix[r, c])


def findAliveNeighborsPeriodic(matrix):
    """
    Gives amount of alive neighbors for all cells
    :param cellPosition:
    :return:
    """
    # matrix = np.pad(matrix, 1)
    shape = np.shape(matrix)
    nRows = shape[0]
    nCols = shape[1]
    nNeighbors = np.empty(shape)

    for row in range(nRows):
        for col in range(nCols):
            nNeighbors[row, col] = countNeighborsPeriodic(row, col, matrix)
    return nNeighbors


def countNeighborsPeriodic(r, c, matrix):
    nSum = 0
    matshape = np.shape(matrix)
    nRows = matshape[0]
    nCols = matshape[1]
    # print(matshape)

    if r == nRows - 1:
        for row in [r - 1, r, 0]:
            if c == nCols - 1:
                for col in [c - 1, c, 0]:
                    nSum += matrix[row, col]
            else:
                for col in [c - 1, c, c + 1]:
                    nSum += matrix[row, col]
    else:
        for row in [r - 1, r, r + 1]:
            if c == nCols - 1:
                for col in [c - 1, c, 0]:
                    nSum += matrix[row, col]
            else:
                for col in [c - 1, c, c + 1]:
                    nSum += matrix[row, col]
    return int(nSum - matrix[r, c])


def generateGeneration(previousGeneration, rules="normal"):
    matshape = np.shape(previousGeneration)
    nextGeneration = np.empty(matshape)
    nNeighborMatrix = findAliveNeighbors(previousGeneration)

    if rules=="normal":
        remainsDead = [0, 1, 2, 4, 5, 6, 7, 8]
        birthNewCell = [3]
        cellDies = [0, 1, 4, 5, 6, 7, 8]
        stayingAlive = [2, 3]
    elif rules=="weird":
        remainsDead = [0, 1, 2, 4, 5, 6, 7, 8]
        birthNewCell = [3]
        cellDies = [0, 1, 4, 5, 6, 7, 8]
        stayingAlive = [2, 3]

    for rowidx, row in enumerate(nNeighborMatrix):
        for colidx, nNeighbors in enumerate(row):

            nNeighbors = int(nNeighbors)
            oldValue = previousGeneration[rowidx, colidx]

            if oldValue == 0:
                if nNeighbors in birthNewCell:
                    nextGeneration[rowidx, colidx] = 1
                elif nNeighbors in remainsDead:
                    nextGeneration[rowidx, colidx] = 0
                else:
                    print("something is wrong in generateGeneration")
                    print(nNeighbors)

            elif oldValue == 1:
                if nNeighbors in cellDies:
                    nextGeneration[rowidx, colidx] = 0
                elif nNeighbors in stayingAlive:
                    nextGeneration[rowidx, colidx] = 1
                else:
                    print("something else is wrong in gGeneration")

    return nextGeneration


def generateGenerationPB(previousGeneration, rules="normal"):
    matshape = np.shape(previousGeneration)
    nextGeneration = np.empty(matshape)
    nNeighborMatrix = findAliveNeighborsPeriodic(previousGeneration)

    if rules=="normal":
        remainsDead = [0, 1, 2, 4, 5, 6, 7, 8]
        birthNewCell = [3]
        cellDies = [0, 1, 4, 5, 6, 7, 8]
        stayingAlive = [2, 3]
    elif rules=="fixed":
        remainsDead = [0, 1, 2, 3, 5, 6, 7, 8]
        birthNewCell = [4]
        cellDies = [0, 1, 3, 5, 7]
        stayingAlive = [2, 4,6,8]
    elif rules=="extinction":
        remainsDead = [0, 1, 2, 3, 6, 7, 8]
        birthNewCell = [4, 5]
        cellDies = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        stayingAlive = []
    elif rules=="blinknoise":
        remainsDead = [0, 1, 2, 5, 6, 7, 8]
        birthNewCell = [3, 4]
        cellDies = [0, 1, 3, 5, 7]
        stayingAlive = [2, 4, 6, 8]
    elif rules=="weird":
        remainsDead = [1, 3, 4, 5, 6, 7, 8]
        birthNewCell = [0, 2]
        cellDies = [0, 1, 4, 5, 6, 7, 8]
        stayingAlive = [2, 3]
    elif rules == "majority":
        remainsDead = [0, 1, 2, 3, 4]
        birthNewCell = [5, 6, 7, 8]
        cellDies = [0, 1, 2, 3]
        stayingAlive = [4, 5, 6, 7, 8]
    elif rules=="extinction2":
        remainsDead = [3, 4, 5, 6, 7, 8]
        birthNewCell = [0, 1, 2]
        cellDies = [3, 4, 5, 6, 7, 8]
        stayingAlive = [0,1,2]

    for rowidx, row in enumerate(nNeighborMatrix):
        for colidx, nNeighbors in enumerate(row):

            nNeighbors = int(nNeighbors)
            oldValue = previousGeneration[rowidx, colidx]

            if oldValue == 0:
                if nNeighbors in birthNewCell:
                    nextGeneration[rowidx, colidx] = 1
                elif nNeighbors in remainsDead:
                    nextGeneration[rowidx, colidx] = 0
                else:
                    print("something is wrong in generateGeneration")
                    print(nNeighbors)

            elif oldValue == 1:
                if nNeighbors in cellDies:
                    nextGeneration[rowidx, colidx] = 0
                elif nNeighbors in stayingAlive:
                    nextGeneration[rowidx, colidx] = 1
                else:
                    print("something else is wrong in gGeneration")

    return nextGeneration


def runGOL(startG, generations, time=0.1, periodic=False, initialWait=0, rules="normal"):
    plt.ion()
    fig, ax = plt.subplots()
    # fig.show()

    for i in range(generations):
        if i == 0:
            ax.imshow(startG)
            plt.title("Generation {}".format(i))
            plt.pause(time + initialWait)
            if periodic:
                nextGen = generateGenerationPB(startG, rules)
            else:
                nextGen = generateGeneration(startG, rules)
            prevGen = startG
        else:
            ax.clear()
            ax.imshow(nextGen)
            plt.title("Generation {}".format(i))
            plt.pause(time)
            prevGen = nextGen
            if periodic:
                nextGen = generateGenerationPB(prevGen, rules)
            else:
                nextGen = generateGeneration(prevGen, rules)
    plt.pause(1)
    # print(nextGen.sum()/10000)


# 4.3
def SCblock():
    block = np.zeros([6, 6])
    block[2:4, 2:4] = 1
    return block


def SCbeehive():
    bh = np.zeros([6, 6])
    bh[1, 2] = 1
    bh[2:4, 1] = 1
    bh[2:4, 3] = 1
    bh[4, 2] = 1
    return bh


def SCloaf():
    loaf = np.zeros([6, 6])
    loaf[1, 2] = 1
    loaf[2:4, 1] = 1
    loaf[4, 2:4] = 1
    loaf[2, 3] = 1
    loaf[3, 4] = 1
    return loaf


def SCboat():
    boat = np.zeros([6, 6])
    boat[2, 2] = 1
    boat[3, 1] = 1
    boat[4, 2:4] = 1
    boat[3, 3] = 1
    return boat


def SCtub():
    tub = np.zeros([6, 6])
    tub[1, 3] = 1
    tub[2, 2] = 1
    tub[2, 4] = 1
    tub[3, 3] = 1
    return tub


# 4.4
def SCblinker():
    blinker = np.zeros([6, 6])
    blinker[2:5, 2] = 1
    return blinker


def SCtoad():
    toad = np.zeros([6, 6])
    toad[2, 2:5] = 1
    toad[3, 1:4] = 1
    return toad


def SCbeacon():
    beacon = np.zeros([6, 6])
    beacon[1, 1:3] = 1
    beacon[2, 1] = 1
    beacon[4, 3:5] = 1
    beacon[3, 4] = 1
    return beacon


## 4.5
def SCglider1():
    gl = np.zeros([6, 6])
    gl[3, 3:6] = 1
    gl[4, 3] = 1
    gl[5, 4] = 1
    return gl


def SCglider2():
    return np.flip(SCglider1(), axis=0)


def SCglider3():
    return np.flip(SCglider1(), axis=1)


def SCglider4():
    return np.flip(SCglider1())


# 4.6
def translate(matrix, x, y):
    return np.roll(matrix, [x, y], axis=[0, 1])


def checkIdentical(arr1, arr2):
    return np.all(arr1 == arr2)


def createCenter(size, mode):
    if mode == "random":
        return np.random.randint(0, 2, [size, size], int)
    if mode == "glider1":
        return SCglider1()
    if mode == "beacon":
        return SCbeacon()
    if mode == "block":
        return SCblock()


def createSearchGrid(size, mode="random"):
    grid = np.zeros([3 * size, 3 * size])
    center = createCenter(size, mode)
    grid[size:2 * size, size:2 * size] = center
    return grid, center


def scanGrid(grid, pattern):
    gridWidth = np.shape(grid)[0]
    gridHeight = np.shape(grid)[1]
    patternWidth = np.shape(pattern)[0]
    patternHeight = np.shape(pattern)[1]

    hSteps = gridWidth - patternWidth + 1
    vSteps = gridHeight - patternHeight + 1
    for row in range(vSteps):
        for col in range(hSteps):
            if checkIdentical(pattern, grid[row:row + patternHeight, col:col + patternWidth]):
                return True, (row, col)
    return False, (None, None)


def runSearch(maxGenerations, pSize=6, start="random", intv=0.01, startWait=0):
    startGen, pattern = createSearchGrid(pSize, start)

    fig, ax = plt.subplots()
    fig.show()

    ax.matshow(startGen)
    plt.pause(intv + startWait)
    nextGen = generateGenerationPB(startGen)

    identical, location = scanGrid(nextGen, pattern)

    k = 0
    plt.title("Generation {}".format(k))

    while not identical:
        k += 1

        ax.clear()
        ax.matshow(nextGen)
        plt.title("Generation {}".format(k))
        plt.pause(intv)

        prevGen = nextGen
        nextGen = generateGenerationPB(prevGen)
        identical, location = scanGrid(nextGen, pattern)

        if k == maxGenerations:
            break
    if identical:
        print('Spaceship or oscillator found!')
        print('Pattern moves ({}, {}) every {} generations'.format(location[0] - pSize, location[1] - pSize, k + 1))
    else:
        print('No pattern found...')

# startGeneration = np.random.randint(0, 2, [10,10], int)

## 4.3##
# startGeneration = SCblock()
# startGeneration = SCbeehive()
# startGeneration = SCloaf()
# startGeneration = SCboat()
# startGeneration = SCtub()
#
## 4.4 ##
# startGeneration = SCblinker()
# startGeneration = SCtoad()
# startGeneration = SCbeacon()

## 4.5 ##
# startGeneration = SCglider1()
# startGeneration = SCglider2()
# startGeneration = SCglider3()
# startGeneration = SCglider4()

# runGOL(startGeneration, 30, 0.2, periodic=True, initialWait=5)


## 4.6 ##
# startGeneration = createSearchGrid(6)
#
# runSearch(100, start="random", pSize=6, intv=0.1, startWait=0)

## 4.7 ##
startGeneration = np.random.randint(0, 2, [100,100], int)

runGOL(startGeneration, 30, 0.1, periodic=True, rules="extinction2", initialWait=5)

## 4.8 ##
# startGeneration = np.random.choice([0, 1], size=[100, 100], p=[0.55, 0.45])
# runGOL(startGeneration, 50, 0.1, periodic=True, rules="majority")