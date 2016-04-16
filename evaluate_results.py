import os
import sys
import numpy as np
import string
import matplotlib.pyplot as plt


path = os.path.dirname(os.path.realpath(__file__))

all_results = dict()
defenders_win_counts = dict()
final_times = ([], [])
# TODO - static vs. dynamic flatness - add up each timestep change for
static_vs_dynamic_flatness = (0., 0.)


for filename in os.listdir(path + "/results"):
    filename_split = string.split(filename, "_")
    numDef = int(filename_split[0][3:])
    numAtk = int(filename_split[1][3:])
    maxInterceptDist = float(filename_split[2][10:])
    attackType = filename_split[3][7:]
    defenseType = filename_split[4][7:]
    meta_data = (numDef, numAtk, maxInterceptDist, attackType, defenseType)
    if meta_data not in all_results.keys():
        all_results[meta_data] = list()
        defenders_win_counts[meta_data] = 0
    npzfile = np.load(path + "/results/" + filename)
    minTTA = npzfile['minTTA']
    finalTime = npzfile['finalTime']
    defenders_win = npzfile['defenders_win']
    attackHistory = npzfile['attackHistory']
    all_results[meta_data].append({"defenders_win": defenders_win, "finalTime": finalTime, "minTTA": minTTA, "attackHistory": attackHistory})
    if defenders_win:
        defenders_win_counts[meta_data] += 1
        final_times[0].append(finalTime)
    else:
        final_times[1].append(finalTime)

defenders_win_mean_final_time = np.mean(final_times[0])
defenders_lose_mean_final_time = np.mean(final_times[1])

for md in all_results.keys():
    results = all_results[md]
    for result in results:

        result_string = ""
        if result["defenders_win"]:
            result_string += "DEFENDERS WIN: "
        else:
            result_string += "DEFENDERS LOSE: "
        result_string += str(md)

        minTTA = result["minTTA"]
        finalTime = result["finalTime"]
        finalIndex = result["minTTA"].shape[0]
        metric_Hz = finalIndex/finalTime
        attackHistory = result["attackHistory"]
        time = np.linspace(0., finalTime, finalIndex)


        minTTA_time_mean = np.max(minTTA, axis=0)
        minTTA_angle_mean = np.max(minTTA, axis=2)

        figx = 20.0
        figy = 10.0
        fig = plt.figure(figsize=(figx, figy))
        ax2 = plt.subplot(121, projection='polar')
        ax1 = plt.subplot(122)
        for r in range(minTTA.shape[1]):
            ax2.plot(np.deg2rad(np.arange(0., 360.+0.001, 1.)), minTTA_time_mean[r, :])  # time average polar plot
            ax1.plot(time, minTTA_angle_mean[:, r])
        plt.title(result_string)
        plt.show()





        print "----------------------------------------------------"
        print result_string
        print md
        print attackHistory
        print "----------------------------------------------------"

        #attackHistory[:, 0] *= metric_Hz  # change this to the row index of minTTA
        #attackHistory[:, 0] = np.round(attackHistory[:, 0], 0)
        attackHistory[:, 1] *= 180./np.pi
        attackHistory[np.where(attackHistory[:, 1] < 0), 1] += 360.
        attackHistory[:, 1] = np.floor(attackHistory[:, 1])
        # minTTA is (time index, radius index, angle index)
        # attackHistory[:, 0] - the time where an attacker goes straight at the asset
        # attackHistory[:, 1] - the angle index

        """
        radii_count = minTTA.shape[1]
        radii = [10.0, 20.0, 40.0]
        radii_colors = ['r', 'g', 'b']
        attack_count = attackHistory.shape[0]
        figx = 20.0
        figy = 10.0
        fig = plt.figure(figsize=(figx, figy))
        ax = fig.add_axes([0.05, 0.075, 0.9, 0.85])
        for a in range(attack_count):
            #plt.subplot(attack_count, 1, a+1)
            #for r in range(minTTA.shape[1]):
            r = 1
            ax.plot(time, minTTA[:, r, attackHistory[a, 1]], linewidth=2)
            ax.plot(attackHistory[a, 0], minTTA[np.ceil(attackHistory[a, 0]*metric_Hz), r, attackHistory[a, 1]], 'o', markersize=14, linewidth=3)
            ax.axis([0., finalTime, 0., 20.])
            ax.grid()
        #plt.subplot(attack_count, 1, 1)
        plt.title(result_string)
        plt.show()
        """



None
