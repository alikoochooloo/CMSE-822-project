# we can later add optional arguments
import argparse
import random

def elements(k, center, maxw):
    alli = []
    allj = []
    half = int(k/2)
    for x in range(center[0]-half,center[0]):
        if x < 0:
            continue
        for y in range(center[1]-half, center[1]+half+1):
            if y < 0 or y >= maxw:
                continue
            alli.append(x)
            allj.append(y)
    
    for y in range(center[1]-half,center[1]):
        if y < 0:
            continue
        alli.append(center[0])
        allj.append(y)
    
    return alli,allj
    # pass

# def calculate(k, center):
#     sum = 0
#     ei, ej = elements(k, center)
#     for i in range(len(ei[0])):
#         sum = sum + ei
    # pass


if __name__ == '__main__':

    kernel = 7
    h = 1000
    w = 1000
    t1 = [[random.random() for i in range(w)] for j in range(2)]
    t2 = [[0]*w for i in range(h-2)]
    t = t1+ t2
    # starting here we must find a way to parallelize with strat 2
    for i in range(2,h):
        for j in range(w):
            sum = 0
            ei,ej = elements(kernel,(i, j),w)
            # between here
            for e in range(len(ei)):
                sum = sum + 2 * t[ei[e]][ej[e]]
            # and here can be parallelized with strat1
            t[i][j] = sum
    # print(t)