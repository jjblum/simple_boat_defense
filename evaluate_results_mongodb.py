import numpy as np
import matplotlib.pyplot as plt
import copy
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import cPickle as cp

client = pymongo.MongoClient('localhost', 27017)
db = client.TTA
results = db.results
plt.rcParams.update({'font.size': 28})

#####results.update({}, {"$set": {"def_speed": "high"}}, upsert=False, multi=True)  # ways to add new fields to the entire collection
#results.create_index([("def_type", pymongo.ASCENDING)])  # creating an index
#results.update({}, {"$set": {"new": False}}, upsert=False, multi=True)  # ways to add new fields to the entire collection



def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    return d


def minTTA_smoothness(db_results):
    # what makes dynamic defense effective at mitigating the advantage that TTA provided the attackers?
    # Maybe smoothness - look at the how the mean (over all angles) minTTA changes over time
    # Smoothness measure (smaller is smoother)
    circularity = list()
    volatility = list()
    final_attack_circularity = list()
    final_attack_max_minTTA = list()
    for result in db_results:
        final_time = result["final_time"]
        minTTA = cp.loads(result["minTTA"])
        T = minTTA.shape[0]
        meanOverTime_minTTA = np.mean(minTTA, axis=0)
        meanOverAngle_minTTA = np.mean(minTTA, axis=2)
        volatility.append(np.sum(np.abs(np.diff(meanOverAngle_minTTA, axis=0)))/T)  # the change per step through time - like a measure of volatility
        circularity.append(np.sum(np.abs(np.diff(meanOverTime_minTTA, axis=1)))/360.)  # the change per angle - like a measure of circularity (how circular it is)
        finalIndex = minTTA.shape[0]
        metric_Hz = finalIndex/final_time
        attackHistory = np.array(cp.loads(result["attackHistory"]))
        if attackHistory.shape != (0,):
            attackHistory[:, 1] *= 180./np.pi
            attackHistory[np.where(attackHistory[:, 1] < 0), 1] += 360.
            attackHistory[:, 1] = np.floor(attackHistory[:, 1])
            final_attack_minTTA = minTTA[np.floor(attackHistory[-1, 0]*metric_Hz), :, :]
            final_attack_circularity.append(np.sum(np.abs(np.diff(final_attack_minTTA, axis=1)))/360.)
            final_attack_max_minTTA.append(np.max(minTTA[np.floor(attackHistory[-1, 0]*metric_Hz), 1, :], axis=0))
    return circularity, volatility, final_attack_circularity, final_attack_max_minTTA


def gather_results(dictionary, collection=results, smoothness=False):
    if type(dictionary) is tuple:
        winning_dict_list = list()
        losing_dict_list = list()
        general_dict_list = list()
        for dict in dictionary:
            winning_temp = copy.deepcopy(dict)
            losing_temp = copy.deepcopy(dict)
            general_temp = copy.deepcopy(dict)
            winning_temp.update({"defenders_win": True})
            losing_temp.update({"defenders_win": False})
            #winning_temp.update({"def_type": "turned"})
            #losing_temp.update({"def_type": "turned"})
            #general_temp.update({"def_type": "turned"})
            winning_dict_list.append(winning_temp)
            losing_dict_list.append(losing_temp)
            general_dict_list.append(general_temp)
        winning_dict = {"$or": winning_dict_list}
        losing_dict = {"$or": losing_dict_list}
        general_dict = {"$or": general_dict_list}
    else:
        winning_dict = copy.deepcopy(dictionary)
        losing_dict = copy.deepcopy(dictionary)
        general_dict = copy.deepcopy(dictionary)
        winning_dict.update({"defenders_win": True})
        losing_dict.update({"defenders_win": False})
        #winning_dict.update({"def_type": "turned"})
        #losing_dict.update({"def_type": "turned"})
        #general_dict.update({"def_type": "turned"})
        general_index_list = list()
        winning_index_list = list()
        losing_index_list = list()
        for key in winning_dict.keys():
            winning_index_list.append((key, pymongo.ASCENDING))
        for key in general_dict.keys():
            general_index_list.append((key, pymongo.ASCENDING))
        for key in losing_dict.keys():
            losing_index_list.append((key, pymongo.ASCENDING))
        collection.create_index(winning_index_list)
        collection.create_index(losing_index_list)
        collection.create_index(general_index_list)

    general_results = collection.find(general_dict)
    winning_results = collection.find(winning_dict)
    losing_results = collection.find(losing_dict)
    rounds = float(general_results.count())
    wins = float(winning_results.count())
    wins_ratio = wins/rounds*100.
    if smoothness:
        losing_circularity, losing_volatility, losing_final_atk_circularity, losing_final_atk_max_minTTA = minTTA_smoothness(losing_results)
        winning_circularity, winning_volatility, winning_final_atk_circularity, winning_final_atk_max_minTTA = minTTA_smoothness(winning_results)
        general_results.close()
        winning_results.close()
        losing_results.close()
        return wins_ratio, (winning_circularity, losing_circularity), (winning_volatility, losing_volatility), (winning_final_atk_circularity, losing_final_atk_circularity), (winning_final_atk_max_minTTA, losing_final_atk_max_minTTA)
    else:
        return wins_ratio

def winner_loser_histogram_plot(winners, losers, title_string, plot_type="final_atk"):

    if plot_type == "final_atk":
        bins = np.linspace(0.15, 0.5, 20)
        axes = [0.10, 0.5, -100., 100.]
    elif plot_type == "circularity":
        bins = np.linspace(0.0, 0.35, 25)
        axes = [0.0, 0.35, -100., 100.]
    elif plot_type == "volatility":
        bins = np.linspace(0., 0.25, 20)
        axes = [0.00, 0.275, -100., 100.]
    elif plot_type == "max_minTTA":
        bins = np.linspace(5., 25., 25)
        axes = [6., 25., -100., 100.]

    hist_win, _ = np.histogram(winners, bins=bins, normed=False)
    hist_lose, _ = np.histogram(losers, bins=bins, normed=False)
    hist_win = hist_win.astype(float)
    hist_lose = hist_lose.astype(float)
    max_bin_value = np.max(np.concatenate((hist_win, hist_lose)))
    hist_win_normalized = hist_win/max_bin_value*100.
    hist_lose_normalized = hist_lose/max_bin_value*100.
    net_wins = (hist_win - hist_lose) #/(float(len(winners))+float(len(losers)))
    win_over_losses = np.divide(hist_win, hist_lose)
    winning_percentage = hist_win/(hist_win + hist_lose)*100.
    #plt.bar(left=bins[1:], height=net_wins, width=bins[1]-bins[0], color='m')

    bin_left_edges = bins[:-1]
    bin_width = bins[1] - bins[0]
    bin_centers = bin_left_edges+bin_width/2.
    winner_count = len(winners)
    loser_count = len(losers)
    win_mean = np.sum(hist_win/winner_count*bin_centers)
    loser_mean = np.sum(hist_lose/loser_count*bin_centers)

    plt.bar(left=bin_left_edges, height=hist_win_normalized, width=bin_width, color='g')
    plt.bar(left=bin_left_edges, height=-hist_lose_normalized, width=bin_width, color='r')
    plt.plot(bin_centers, winning_percentage, 'ko', markersize=15)
    plt.plot(win_mean, 0., 'g.', markersize=40, markeredgecolor='lightgreen', markeredgewidth=8)
    plt.plot(loser_mean, 0., 'r.', markersize=40, markeredgecolor='pink', markeredgewidth=8)

    plt.title(title_string)
    plt.axis(axes)
    return

plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)

high_speed_results = gather_results({"def_speed": "high", "new": True})
high_speed_random_results = gather_results({"def_speed": "high", "atk_type":"random", "new": True})
high_speed_TTA_results = gather_results({"def_speed": "high", "atk_type":"TTA", "new": True})
high_speed_static_results = gather_results({"def_speed": "high", "def_type":"static", "new": True})
high_speed_turned_results = gather_results({"def_speed": "high", "def_type":"turned", "new": True})
high_speed_dynamic_results = gather_results({"def_speed": "high", "def_type":"dynamic", "new": True})
high_speed_static_random_results = gather_results({"def_speed": "high", "def_type":"static", "atk_type":"random", "new": True}, smoothness=True)
high_speed_turned_random_results = gather_results({"def_speed": "high", "def_type":"turned", "atk_type":"random", "new": True}, smoothness=True)
high_speed_dynamic_random_results = gather_results({"def_speed": "high", "def_type":"dynamic", "atk_type":"random", "new": True}, smoothness=True)
high_speed_static_TTA_results = gather_results({"def_speed": "high", "def_type":"static", "atk_type":"TTA", "new": True}, smoothness=True)
high_speed_turned_TTA_results = gather_results({"def_speed": "high", "def_type":"turned", "atk_type":"TTA", "new": True}, smoothness=True)
high_speed_dynamic_TTA_results = gather_results({"def_speed": "high", "def_type":"dynamic", "atk_type":"TTA", "new": True}, smoothness=True)


print "Full speed defenders had a {:.1f}% winning percentage".format(high_speed_results)
print "Full speed  defenders, static defense had a {:.1f}% winning percentage".format(high_speed_static_results)
print "Full speed defenders, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_results)
print "Full speed defenders, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_results)
print "Full speed defenders, random attack had a {:.1f}% winning percentage".format(high_speed_random_results)
print "Full speed defenders, TTA attack had a {:.1f}% winning percentage".format(high_speed_TTA_results)
print "Full speed defenders, random attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_random_results[0])
print "Full speed defenders, random attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_random_results[0])
print "Full speed defenders, random attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_random_results[0])
print "Full speed defenders, TTA attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_TTA_results[0])
print "Full speed defenders, TTA attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_TTA_results[0])
print "Full speed defenders, TTA attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_TTA_results[0])

plt.figure(figsize=(3.490, 1.870), dpi=275)
plt.subplot(231)
winner_loser_histogram_plot(high_speed_static_random_results[4][0], high_speed_static_random_results[4][1], plot_type="max_minTTA", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(high_speed_turned_random_results[4][0], high_speed_turned_random_results[4][1], plot_type="max_minTTA", title_string="Full speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(high_speed_dynamic_random_results[4][0], high_speed_dynamic_random_results[4][1], plot_type="max_minTTA", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(high_speed_static_TTA_results[4][0], high_speed_static_TTA_results[4][1], plot_type="max_minTTA", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(high_speed_turned_TTA_results[4][0], high_speed_turned_TTA_results[4][1], plot_type="max_minTTA", title_string="def: turned    atk: TTA")
plt.xlabel("Maximum minTTA at time of final attack", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(high_speed_dynamic_TTA_results[4][0], high_speed_dynamic_TTA_results[4][1], plot_type="max_minTTA", title_string="def: dynamic    atk: TTA")
#plt.show()
plt.savefig("fast_max_minTTA.png", bbox_inches='tight', dpi=275)


plt.subplot(231)
winner_loser_histogram_plot(high_speed_static_random_results[1][0], high_speed_static_random_results[1][1], plot_type="circularity", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(high_speed_turned_random_results[1][0], high_speed_turned_random_results[1][1], plot_type="circularity", title_string="Full speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(high_speed_dynamic_random_results[1][0], high_speed_dynamic_random_results[1][1], plot_type="circularity", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(high_speed_static_TTA_results[1][0], high_speed_static_TTA_results[1][1], plot_type="circularity", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(high_speed_turned_TTA_results[1][0], high_speed_turned_TTA_results[1][1], plot_type="circularity", title_string="def: turned    atk: TTA")
plt.xlabel("Circularity", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(high_speed_dynamic_TTA_results[1][0], high_speed_dynamic_TTA_results[1][1], plot_type="circularity", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("fast_circularity.png", bbox_inches='tight')

plt.subplot(231)
winner_loser_histogram_plot(high_speed_static_random_results[2][0], high_speed_static_random_results[2][1], plot_type="volatility", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(high_speed_turned_random_results[2][0], high_speed_turned_random_results[2][1], plot_type="volatility", title_string="Full speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(high_speed_dynamic_random_results[2][0], high_speed_dynamic_random_results[2][1], plot_type="volatility", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(high_speed_static_TTA_results[2][0], high_speed_static_TTA_results[2][1], plot_type="volatility", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(high_speed_turned_TTA_results[2][0], high_speed_turned_TTA_results[2][1], plot_type="volatility", title_string="def: turned    atk: TTA")
plt.xlabel("Volatility", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(high_speed_dynamic_TTA_results[2][0], high_speed_dynamic_TTA_results[2][1], plot_type="volatility", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("fast_volatility.png", bbox_inches='tight')

plt.subplot(231)
winner_loser_histogram_plot(high_speed_static_random_results[3][0], high_speed_static_random_results[3][1], plot_type="final_atk", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(high_speed_turned_random_results[3][0], high_speed_turned_random_results[3][1], plot_type="final_atk", title_string="Full speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(high_speed_dynamic_random_results[3][0], high_speed_dynamic_random_results[3][1], plot_type="final_atk", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(high_speed_static_TTA_results[3][0], high_speed_static_TTA_results[3][1], plot_type="final_atk", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(high_speed_turned_TTA_results[3][0], high_speed_turned_TTA_results[3][1], plot_type="final_atk", title_string="def: turned    atk: TTA")
plt.xlabel("Circularity at time of final attack", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(high_speed_dynamic_TTA_results[3][0], high_speed_dynamic_TTA_results[3][1], plot_type="final_atk", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("fast_final_atk_circularity.png", bbox_inches='tight')


low_speed_results = gather_results({"def_speed": "low", "new": True})
low_speed_random_results = gather_results({"def_speed": "low", "atk_type":"random", "new": True})
low_speed_TTA_results = gather_results({"def_speed": "low", "atk_type":"TTA", "new": True})
low_speed_static_results = gather_results({"def_speed": "low", "def_type":"static", "new": True})
low_speed_turned_results = gather_results({"def_speed": "low", "def_type":"turned", "new": True})
low_speed_dynamic_results = gather_results({"def_speed": "low", "def_type":"dynamic", "new": True})
low_speed_static_random_results = gather_results({"def_speed": "low", "def_type":"static", "atk_type":"random", "new": True}, smoothness=True)
low_speed_turned_random_results = gather_results({"def_speed": "low", "def_type":"turned", "atk_type":"random", "new": True}, smoothness=True)
low_speed_dynamic_random_results = gather_results({"def_speed": "low", "def_type":"dynamic", "atk_type":"random", "new": True}, smoothness=True)
low_speed_static_TTA_results = gather_results({"def_speed": "low", "def_type":"static", "atk_type":"TTA", "new": True}, smoothness=True)
low_speed_turned_TTA_results = gather_results({"def_speed": "low", "def_type":"turned", "atk_type":"TTA", "new": True}, smoothness=True)
low_speed_dynamic_TTA_results = gather_results({"def_speed": "low", "def_type":"dynamic", "atk_type":"TTA", "new": True}, smoothness=True)
print "Half thrust (70% top speed) defenders overall had a {:.1f}% winning percentage".format(low_speed_results)
print "Half thrust (70% top speed) defenders, static defense had a {:.1f}% winning percentage".format(low_speed_static_results)
print "Half thrust (70% top speed) defenders, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_results)
print "Half thrust (70% top speed) defenders, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_results)
print "Half thrust (70% top speed) defenders, random attack had a {:.1f}% winning percentage".format(low_speed_random_results)
print "Half thrust (70% top speed) defenders, TTA attack had a {:.1f}% winning percentage".format(low_speed_TTA_results)
print "Half thrust (70% top speed) defenders, random attack, static defense had a {:.1f}% winning percentage".format(low_speed_static_random_results[0])
print "Half thrust (70% top speed) defenders, random attack, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_random_results[0])
print "Half thrust (70% top speed) defenders, random attack, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_random_results[0])
print "Half thrust (70% top speed) defenders, TTA attack, static defense had a {:.1f}% winning percentage".format(low_speed_static_TTA_results[0])
print "Half thrust (70% top speed) defenders, TTA attack, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_TTA_results[0])
print "Half thrust (70% top speed) defenders, TTA attack, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_TTA_results[0])
plt.subplot(231)
winner_loser_histogram_plot(low_speed_static_random_results[4][0], low_speed_static_random_results[4][1], plot_type="max_minTTA", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(low_speed_turned_random_results[4][0], low_speed_turned_random_results[4][1], plot_type="max_minTTA", title_string="70% speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(low_speed_dynamic_random_results[4][0], low_speed_dynamic_random_results[4][1], plot_type="max_minTTA", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(low_speed_static_TTA_results[4][0], low_speed_static_TTA_results[4][1], plot_type="max_minTTA", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(low_speed_turned_TTA_results[4][0], low_speed_turned_TTA_results[4][1], plot_type="max_minTTA", title_string="def: turned    atk: TTA")
plt.xlabel("Maximum minTTA at time of final attack", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(low_speed_dynamic_TTA_results[4][0], low_speed_dynamic_TTA_results[4][1], plot_type="max_minTTA", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("slow_max_minTTA.png", bbox_inches='tight')


plt.subplot(231)
winner_loser_histogram_plot(low_speed_static_random_results[1][0], low_speed_static_random_results[1][1], plot_type="circularity", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(low_speed_turned_random_results[1][0], low_speed_turned_random_results[1][1], plot_type="circularity", title_string="70% speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(low_speed_dynamic_random_results[1][0], low_speed_dynamic_random_results[1][1], plot_type="circularity", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(low_speed_static_TTA_results[1][0], low_speed_static_TTA_results[1][1], plot_type="circularity", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(low_speed_turned_TTA_results[1][0], low_speed_turned_TTA_results[1][1], plot_type="circularity", title_string="def: turned    atk: TTA")
plt.xlabel("Circularity", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(low_speed_dynamic_TTA_results[1][0], low_speed_dynamic_TTA_results[1][1], plot_type="circularity", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("slow_circularity.png", bbox_inches='tight')

plt.subplot(231)
winner_loser_histogram_plot(low_speed_static_random_results[2][0], low_speed_static_random_results[2][1], plot_type="volatility", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(low_speed_turned_random_results[2][0], low_speed_turned_random_results[2][1], plot_type="volatility", title_string="70% speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(low_speed_dynamic_random_results[2][0], low_speed_dynamic_random_results[2][1], plot_type="volatility", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(low_speed_static_TTA_results[2][0], low_speed_static_TTA_results[2][1], plot_type="volatility", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(low_speed_turned_TTA_results[2][0], low_speed_turned_TTA_results[2][1], plot_type="volatility", title_string="def: turned    atk: TTA")
plt.xlabel("Volatility", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(low_speed_dynamic_TTA_results[2][0], low_speed_dynamic_TTA_results[2][1], plot_type="volatility", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("slow_volatility.png", bbox_inches='tight')

plt.subplot(231)
winner_loser_histogram_plot(low_speed_static_random_results[3][0], low_speed_static_random_results[3][1], plot_type="final_atk", title_string="def: static    atk: random")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(232)
winner_loser_histogram_plot(low_speed_turned_random_results[3][0], low_speed_turned_random_results[3][1], plot_type="final_atk", title_string="70% speed\ndef: turned    atk: random")
plt.subplot(233)
winner_loser_histogram_plot(low_speed_dynamic_random_results[3][0], low_speed_dynamic_random_results[3][1], plot_type="final_atk", title_string="def: dynamic    atk: random")
plt.subplot(234)
winner_loser_histogram_plot(low_speed_static_TTA_results[3][0], low_speed_static_TTA_results[3][1], plot_type="final_atk", title_string="def: static    atk: TTA")
plt.ylabel("Win/Loss Counts (bars)\nWinning % (black dots)\nWin/Loss Mean (colored dots)", fontsize=40)
plt.subplot(235)
winner_loser_histogram_plot(low_speed_turned_TTA_results[3][0], low_speed_turned_TTA_results[3][1], plot_type="final_atk", title_string="def: turned    atk: TTA")
plt.xlabel("Circularity at time of final attack", fontsize=50)
plt.subplot(236)
winner_loser_histogram_plot(low_speed_dynamic_TTA_results[3][0], low_speed_dynamic_TTA_results[3][1], plot_type="final_atk", title_string="def: dynamic    atk: TTA")
plt.show()
plt.savefig("slow_final_atk_circularity.png", bbox_inches='tight')


"""
print "\nNUMBER OF EXTRA DEFENDERS"
zero_results = gather_results({"num_attackers": 4, "num_defenders": 4, "def_speed": "high"}, smoothness=True)
one_results = gather_results(({"num_attackers": 4, "num_defenders": 5, "def_speed": "high"}, {"num_attackers": 3, "num_defenders": 4, "def_speed": "high"}), smoothness=True)
two_results = gather_results({"num_attackers": 3, "num_defenders": 5, "def_speed": "high"}, smoothness=True)
print "0-extra defenders had a {:.1f}% winning percentage".format(zero_results[0])
print "1-extra defender had a {:.1f}% winning percentage".format(one_results[0])
print "2-extra defenders had a {:.1f}% winning percentage".format(two_results[0])
"""

"""
plt.subplot(231)
static_circularity_histogram_distance = winner_loser_histogram_plot(zero_results[1][0], zero_results[1][1], "0-extra defenders circularity", "circularity")
plt.subplot(232)
turned_circularity_histogram_distance = winner_loser_histogram_plot(one_results[1][0], one_results[1][1], "1-extra defender circularity", "circularity")
plt.subplot(233)
dynamic_circularity_histogram_distancec = winner_loser_histogram_plot(two_results[1][0], two_results[1][1], "2-extra defenders circularity", "circularity")
plt.subplot(234)
static_volatility_histogram_distance = winner_loser_histogram_plot(zero_results[2][0], zero_results[2][1], "0-extra defenders volatility", "volatility")
plt.subplot(235)
turned_volatility_histogram_distance = winner_loser_histogram_plot(one_results[2][0], one_results[2][1], "1-extra defender volatility", "volatility")
plt.subplot(236)
dynamic_volatility_histogram_distancec = winner_loser_histogram_plot(two_results[2][0], two_results[2][1], "2-extra defenders volatility", "volatility")
plt.show()
"""
"""
plt.subplot(131)
winner_loser_histogram_plot(zero_results[4][0], zero_results[4][1], "0-extra defenders final attack max minTTA", plot_type="max_minTTA")
plt.subplot(132)
winner_loser_histogram_plot(one_results[4][0], one_results[4][1], "1-extra defender final attack max minTTA", plot_type="max_minTTA")
plt.subplot(133)
winner_loser_histogram_plot(two_results[4][0], two_results[4][1], "2-extra defenders final attack max minTTA", plot_type="max_minTTA")
plt.show()
"""

"""
high_speed_static_results = gather_results({"def_speed": "high", "def_type":"static"}, smoothness=True)
high_speed_turned_results = gather_results({"def_speed": "high", "def_type":"turned"}, smoothness=True)
high_speed_dynamic_results = gather_results({"def_speed": "high", "def_type":"dynamic"}, smoothness=True)
plt.subplot(131)
winner_loser_histogram_plot(high_speed_static_results[4][0], high_speed_static_results[4][1], "static final attack max minTTA", plot_type="max_minTTA")
plt.subplot(132)
winner_loser_histogram_plot(high_speed_turned_results[4][0], high_speed_turned_results[4][1], "turned final attack max minTTA", plot_type="max_minTTA")
plt.subplot(133)
winner_loser_histogram_plot(high_speed_dynamic_results[4][0], high_speed_dynamic_results[4][1], "dynamic final attack max minTTA", plot_type="max_minTTA")
plt.show()
"""