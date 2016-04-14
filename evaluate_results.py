import os
import sys
import numpy as np
import string


path = os.path.dirname(os.path.realpath(__file__))

all_results = dict()
defenders_win_counts = dict()

for filename in os.listdir(path + "/results"):
    filename_split = string.split(filename, "_")
    numDef = int(filename_split[0][3:])
    numAtk = int(filename_split[1][3:])
    maxInterceptDist = float(filename_split[2][10:])
    attackType = filename_split[3]
    meta_data = (numDef, numAtk, maxInterceptDist, attackType)
    if meta_data not in all_results.keys():
        all_results[meta_data] = list()
        defenders_win_counts[meta_data] = 0
    npzfile = np.load(path + "/results/" + filename)
    minTTA = npzfile['minTTA']
    finalTime = npzfile['finalTime']
    defenders_win = npzfile['defenders_win']
    attackHistory = npzfile['attackHistory']
    all_results[meta_data].append((defenders_win, finalTime, minTTA))
    if defenders_win:
        defenders_win_counts[meta_data] += 1

None
