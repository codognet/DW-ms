from dwave_qbsolv import QBSolv
import neal
import numpy as np
import pickle
import string
import math

def errorInMagicSquare(s):
    n = len(s)
    n2 = n*n
    m = int(n*(n2+1)/2)
    sums = np.zeros((2*n+2,), dtype=int)
    #rows
    for i in range(n):
        for j in range(n):
            sums[i] += s[i,j]
        sums[i] -= m
    #columns
    for j in range(n):
        for i in range(n):
            sums[n+j] += s[i,j]
        sums[n+j] -= m
    #diag1
    for i in range(n):
        sums[2*n] += s[i,i]
    sums[2*n] -= m
    #diag2
    for i in range(n):
        sums[2*n+1] += s[i,n-i-1]
    sums[2*n+1] -= m
    res = 0
    for k in range(2*n+2):
        res += sums[k]
    return res

# QBSolv Parameter
num_repeats = 5000
seed = None
algorithm = None
verbosity = 0
timeout = 2592000
solver_limit = None
target = None
find_max = False
solver = 'tabu'
#problem size, etc
n = 5
n2 = n*n
m = int(n*(n2+1)/2)   # "mean" of magic square
penalty=n*n*10
penalty2=n*n*n*n*100
penalty3=n*n*n*n*n*n*1000
QUBO = {}

# dict to encode 3D index boolean x(i,j,k) into 1D
B = {}
index = 0
for i in range(n):
    for j in range(n):
        for k in range(n2):
            B[(i,j,k)] = index
            index +=1

def x(i,j,k):
    return int(B[(i,j,k)])

print("starting...", "n:", n, "m:", m)

#initialization of QUBO matrix
for i in range(len(B)):
    for j in range(i,len(B)):
        QUBO[i,j] = 0

# Stating the constraints...
#Remark: numbers {0,...,(n-1)*(n-1)} rather than {1,...,n*n} on the square...

# Constraint that there is exactly one value per cell
for i in range(n):
    for j in range(n):
        for k in range(n2):
            QUBO[x(i,j,k), x(i,j,k)] -= 1 * penalty3
            for k1 in range(k+1,n2):
                QUBO[x(i,j,k), x(i,j,k1)] += 2 * penalty3

# Constraint that all n2 value are assigned
for k in range(n2):
    for i in range(n):
        for j in range(n):
            QUBO[x(i,j,k), x(i,j,k)] -= 1 * penalty3
            for j1 in range(j+1,n):
                QUBO[x(i, j, k), x(i, j1, k)] += 2 * penalty3
            for i1 in range(i+1,n):
                for j1 in range(n):
                    QUBO[x(i,j,k), x(i1,j1,k)] += 2 * penalty3

# constraint: sum on each row = m - n  i.e.  sum(j,k) k*x(i,j,k) = m-n
for i in range(n):
    for j in range(n):
        for k in range(n2):
            QUBO[x(i,j,k), x(i,j,k)] += penalty2 * (k*k - 2*(m-n)*k)
            for k1 in range(k+1,n2):
                QUBO[x(i, j, k), x(i, j, k1)] += 2 * penalty2 * k * k1
            for j1 in range(j+1,n):
                for k1 in range(n2):
                    QUBO[x(i,j,k), x(i,j1,k1)] += 2 * penalty2 * k * k1

# constraint: sum on each column = m - n
for j in range(n):
    for i in range(n):
        for k in range(n2):
            QUBO[x(i,j,k), x(i,j,k)] += penalty2 * (k*k - 2*(m-n)*k)
            for k1 in range(k+1,n2):
                QUBO[x(i, j, k), x(i, j, k1)] += 2 * penalty2 * k * k1
            for i1 in range(i+1,n):
                for k1 in range(n2):
                    QUBO[x(i,j,k), x(i1,j,k1)] += 2 * penalty2 * k * k1

# constraint: sum on diagonal1 = m - n
for i in range(n):
        for k in range(n2):
            QUBO[x(i,i,k), x(i,i,k)] += penalty2 * (k*k - 2*(m-n)*k)
            for k1 in range(k+1,n2):
                QUBO[x(i, i, k), x(i, i, k1)] += 2 * penalty2 * k * k1
            for i1 in range(i+1,n):
                for k1 in range(n2):
                    QUBO[x(i,i,k), x(i1,i1,k1)] += 2 * penalty2 * k * k1

# constraint: sum on diagonal2 = m - n
for i in range(n):
        for k in range(n2):
            QUBO[x(i,n-i-1,k), x(i,n-i-1,k)] += penalty2 * (k*k - 2*(m-n)*k)
            for k1 in range(k+1,n2):
                QUBO[x(i, n-i-1, k), x(i, n-i-1, k1)] += 2 * penalty2 * k * k1
            for i1 in range(i+1,n):
                for k1 in range(n2):
                    QUBO[x(i,n-i-1,k), x(i1,n-i1-1,k1)] += 2 * penalty2 * k * k1

def test_qubo_ms(Q):

    # Call QBSolv
    #result = QBSolv().sample_qubo(Q, num_repeats=num_repeats, seed=seed, algorithm=algorithm, verbosity=verbosity,
    #                           timeout=timeout, solver_limit=solver_limit, solver=solver, target=target,
    #                           find_max=find_max).record

    #call neal
    #sampler = neal.SimulatedAnnealingSampler()

    sampler = QBSolv()
    print("sampling starting...")
    sampleset = sampler.sample_qubo(Q)
    print("sampled")
    energy = sampleset.first.energy
    print("Energy:", energy)
    sol = sampleset.first.sample

    # print magic square as permutation
    l = list(B)
    marks = 0
    ms=[]
    square = np.zeros((n,n), dtype=int)
    for s in range(len(sol)):
        if sol[s]:
            a=l[s]
            i,j,k = a
            ms.append(k+1)
            square[i,j] = k+1
    print("magic square:", ms)
    print(square)
    e = errorInMagicSquare(square)
    if e==0:
        print("is a magic square !")
    else:
        print("is NOT a magic square, TotalDev =",e)
"""
    square1 = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n2):
                if sol[n2*i + n*j + k]:
                    square1[i,j] = k+1
    print("square1:", "\n", square1)
"""
#pickle.dump(QUBO,  open( "costas" + str(n) + ".QUBO", "wb" ) )
#print("QUBO matrix saved on file")

test_qubo_ms(QUBO)

#Q = pickle.load(open( "costas" + str(n) + ".QUBO", "rb" ) )
#print("after save+read")
#test_qubo_costas(Q)

#print("sol:", sol)
# n=4: sol = [16,3,2,13,5,10,11,8,9,6,7,12,4,15,14,1]





