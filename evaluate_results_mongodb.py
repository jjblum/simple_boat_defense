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
plt.rcParams.update({'font.size': 18})

#results.update({}, {"$set": {"def_speed": "high"}}, upsert=False, multi=True)  # ways to add new fields to the entire collection



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
        bins = np.linspace(0., 0.4, 50)
        ylim = 40.
    elif plot_type == "circularity":
        bins = np.linspace(0., 0.4, 50)
        ylim = 40.
    elif plot_type == "volatility":
        bins = np.linspace(0., 0.3, 50)
        ylim = 40.
    elif plot_type == "max_minTTA":
        bins = np.linspace(5., 25., 50)
        ylim = 0.5
    n_win, bins, patches = plt.hist(winners, bins=bins, normed=True, facecolor='green', alpha=0.5)
    n_lose, bins, patches = plt.hist(losers, bins=bins, normed=True, facecolor='red', alpha=0.5)
    plt.ylim(0., ylim)
    plt.title(title_string)
    return chi2_distance(n_win, n_lose)


"""
print "\nAll RUNS:"
winning_results = results.find({"defenders_win": True})
rounds = float(results.count())
wins = float(winning_results.count())
winning_results.close()
winning_ratio = wins/rounds*100.
print "The defenders had a {:.1f}% winning percentage".format(winning_ratio)
"""

"""
print "\nRANDOM VS. TTA EXPLOIT ATTACK:"
random_results = gather_results({"atk_type": "random"})
print "Any random attack had a {:.1f}% winning percentage".format(random_results)
TTA_results = gather_results({"atk_type": "TTA"})
print "Any attack using TTA had a {:.1f}% winning percentage".format(TTA_results)


print "\nSTATIC VS. DYNAMIC DEFENSE:"
static_results = gather_results({"def_type": "static"}, smoothness=True)
print "Any static defense had a {:.1f}% winning percentage".format(static_results[0])
dynamic_results = gather_results({"def_type": "dynamic"}, smoothness=True)
print "Any dynamic defense had a {:.1f}% winning percentage".format(dynamic_results[0])

plt.subplot(121)
winner_loser_histogram_plot(static_results[3][0], static_results[3][1], "static final attack circularity", "final_atk")
plt.subplot(122)
winner_loser_histogram_plot(dynamic_results[3][0], dynamic_results[3][1], "dynamic final attack circularity", "final_atk")
plt.show()

plt.subplot(221)
static_circularity_histogram_distance = winner_loser_histogram_plot(static_results[1][0], static_results[1][1], "static circularity", "circularity")
plt.subplot(222)
dynamic_circularity_histogram_distance = winner_loser_histogram_plot(dynamic_results[1][0], dynamic_results[1][1], "dynamic circularity", "circularity")
plt.subplot(223)
static_volatility_histogram_distance = winner_loser_histogram_plot(static_results[2][0], static_results[2][1], "static volatility", "volatility")
plt.subplot(224)
dynamic_volatility_histogram_distance = winner_loser_histogram_plot(dynamic_results[2][0], dynamic_results[2][1], "dynamic volatility", "volatility")
print "\nTTA SMOOTHNESS MEASURES:"
print "Static circularity chi-squared distance between winners and losers = {:.3f}".format(static_circularity_histogram_distance)
print "Dynamic circularity chi-squared distance between winners and losers = {:.3f}".format(dynamic_circularity_histogram_distance)
print "Static voltatility chi-squared distance between winners and losers = {:.3f}".format(static_volatility_histogram_distance)
print "Dynamic voltatility chi-squared distance between winners and losers = {:.3f}".format(dynamic_volatility_histogram_distance)
plt.show()


print "\nTTA ATTACK, STATIC VS. DYNAMIC DEFENSE:"
TTA_static_results = gather_results({"atk_type": "TTA", "def_type": "static"})
print "TTA attack, static defense had a {:.1f}% winning percentage".format(TTA_static_results)
TTA_dynamic_results = gather_results({"atk_type": "TTA", "def_type": "dynamic"})
print "TTA attack, dynamic defense had a {:.1f}% winning percentage".format(TTA_dynamic_results)


print "\nRANDOM ATTACK, STATIC VS. DYNAMIC DEFENSE:"
random_static_results = gather_results({"atk_type": "random", "def_type": "static"})
print "random attack, static defense had a {:.1f}% winning percentage".format(random_static_results)
random_dynamic_results = gather_results({"atk_type": "random", "def_type": "dynamic"})
print "random attack, dynamic defense had a {:.1f}% winning percentage".format(random_dynamic_results)
"""

"""
print "\n0 EXTRA DEFENDERS VS. 1 EXTRA DEFENDER VS. 2 EXTRA DEFENDERS:"
zero_defender_results = gather_results({"num_attackers": 4, "num_defenders": 4}, smoothness=True)
print "0 extra defenders had a {:.1f}% winning percentage".format(zero_defender_results[0])
one_defender_results = gather_results(({"num_attackers": 4, "num_defenders": 5}, {"num_attackers": 3, "num_defenders": 4}), smoothness=True)
print "1 extra defender had a {:.1f}% winning percentage".format(one_defender_results[0])
two_defender_results = gather_results({"num_attackers": 3, "num_defenders": 5}, smoothness=True)
print "2 extra defenders had a {:.1f}% winning percentage".format(two_defender_results[0])
plt.subplot(231)
zero_defender_circularity_histogram_distance = winner_loser_histogram_plot(zero_defender_results[1][0], zero_defender_results[1][1], "0-extra defenders circularity", "circularity")
plt.subplot(232)
one_defender_circularity_histogram_distance = winner_loser_histogram_plot(one_defender_results[1][0], one_defender_results[1][1], "1-extra defender circularity", "circularity")
plt.subplot(233)
two_defender_circularity_histogram_distance = winner_loser_histogram_plot(two_defender_results[1][0], two_defender_results[1][1], "2-extra defenders", "circularity")
plt.subplot(234)
zero_defender_volatility_histogram_distance = winner_loser_histogram_plot(zero_defender_results[2][0], zero_defender_results[2][1], "0-extra defenders volatility", "volatility")
plt.subplot(235)
one_defender_volatility_histogram_distance = winner_loser_histogram_plot(one_defender_results[2][0], one_defender_results[2][1], "1-extra defender volatility", "volatility")
plt.subplot(236)
two_defender_volatility_histogram_distance = winner_loser_histogram_plot(two_defender_results[2][0], two_defender_results[2][1], "2-extra defenders volatility", "volatility")
plt.show()
"""

"""
print "\n30 VS. 40 MAX INTERCEPT DISTANCE:"
thirty_max_intercept_distance_results = gather_results({"max_allowable_intercept_distance": 30.}, smoothness=True)
forty_max_intercept_distance_results = gather_results({"max_allowable_intercept_distance": 40.}, smoothness=True)
print "Max intercept distance = 30 had a {:.1f}% winning percentage".format(thirty_max_intercept_distance_results[0])
print "Max intercept distance = 40 had a {:.1f}% winning percentage".format(forty_max_intercept_distance_results[0])
"""


"""
print "\nDEFENDER SPEED:"
low_speed_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low"}, smoothness=True)
high_speed_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high"}, smoothness=True)
plt.subplot(121)
winner_loser_histogram_plot(high_speed_results[4][0], high_speed_results[4][1], "fast defender final attack max minTTA", plot_type="max_minTTA")
plt.subplot(122)
winner_loser_histogram_plot(low_speed_results[4][0], low_speed_results[4][1], "slow defender final attack max minTTA", plot_type="max_minTTA")
plt.show()
"""

"""
low_speed_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "atk_type":"random"})
low_speed_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "atk_type":"TTA"})
low_speed_static_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"static"})
low_speed_turned_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"turned"})
low_speed_dynamic_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"dynamic"})
low_speed_static_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"static", "atk_type":"random"})
low_speed_turned_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"turned", "atk_type":"random"})
low_speed_dynamic_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"dynamic", "atk_type":"random"})
low_speed_static_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"static", "atk_type":"TTA"})
low_speed_turned_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"turned", "atk_type":"TTA"})
low_speed_dynamic_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "low", "def_type":"dynamic", "atk_type":"TTA"})
print "Full speed defenders had a {:.1f}% winning percentage".format(high_speed_results[0])
print "Half thrust (70% top speed) defenders overall had a {:.1f}% winning percentage".format(low_speed_results[0])
print "Half thrust (70% top speed) defenders, static defense had a {:.1f}% winning percentage".format(low_speed_static_results)
print "Half thrust (70% top speed) defenders, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_results)
print "Half thrust (70% top speed) defenders, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_results)
print "Half thrust (70% top speed) defenders, random attack had a {:.1f}% winning percentage".format(low_speed_random_results)
print "Half thrust (70% top speed) defenders, TTA attack had a {:.1f}% winning percentage".format(low_speed_TTA_results)
print "Half thrust (70% top speed) defenders, random attack, static defense had a {:.1f}% winning percentage".format(low_speed_static_random_results)
print "Half thrust (70% top speed) defenders, random attack, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_random_results)
print "Half thrust (70% top speed) defenders, random attack, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_random_results)
print "Half thrust (70% top speed) defenders, TTA attack, static defense had a {:.1f}% winning percentage".format(low_speed_static_TTA_results)
print "Half thrust (70% top speed) defenders, TTA attack, turned defense had a {:.1f}% winning percentage".format(low_speed_turned_TTA_results)
print "Half thrust (70% top speed) defenders, TTA attack, dynamic defense had a {:.1f}% winning percentage".format(low_speed_dynamic_TTA_results)

plt.subplot(221)
low_speed_circularity_histogram_distance = winner_loser_histogram_plot(low_speed_results[1][0], low_speed_results[1][1], "Fast defender circularity", "circularity")
plt.subplot(222)
high_speed_circularity_histogram_distance = winner_loser_histogram_plot(high_speed_results[1][0], high_speed_results[1][1], "Slow defender circularity", "circularity")
plt.subplot(223)
low_speed_volatility_histogram_distance = winner_loser_histogram_plot(low_speed_results[2][0], low_speed_results[2][1], "Fast defender volatility", "volatility")
plt.subplot(224)
high_speed_volatility_histogram_distance = winner_loser_histogram_plot(high_speed_results[2][0], high_speed_results[2][1], "Slow defender volatility", "volatility")
plt.show()
"""


"""
print "\nSTATIC VS DYNAMIC VS TURNED, 4 DEFENDERS AND 3 ATTACKERS"
general_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high"})
random_attack_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "atk_type": "random"})
TTA_attack_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "atk_type": "TTA"})
static_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "static", "def_speed": "high"}, smoothness=True)
turned_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "turned", "def_speed": "high"}, smoothness=True)
dynamic_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "dynamic", "def_speed": "high"}, smoothness=True)
print "All defenses, 4 defenders and 3 attackers, both attack types, had a {:.1f}% winning percentage".format(general_results)
print "All defenses, 4 defenders and 3 attackers, random attack, had a {:.1f}% winning percentage".format(random_attack_results)
print "All defenses, 4 defenders and 3 attackers, TTA attack, had a {:.1f}% winning percentage".format(TTA_attack_results)
print "Static defense, 4 defenders and 3 attackers, had a {:.1f}% winning percentage".format(static_results[0])
print "Turned defense, 4 defenders and 3 attackers, had a {:.1f}% winning percentage".format(turned_results[0])
print "Dynamic defense, 4 defenders and 3 attackers, had a {:.1f}% winning percentage".format(dynamic_results[0])
plt.subplot(231)
static_circularity_histogram_distance = winner_loser_histogram_plot(static_results[1][0], static_results[1][1], "static circularity", "circularity")
plt.subplot(232)
turned_circularity_histogram_distance = winner_loser_histogram_plot(turned_results[1][0], turned_results[1][1], "turned circularity", "circularity")
plt.subplot(233)
dynamic_circularity_histogram_distancec = winner_loser_histogram_plot(dynamic_results[1][0], dynamic_results[1][1], "dynamic circularity", "circularity")
plt.subplot(234)
static_volatility_histogram_distance = winner_loser_histogram_plot(static_results[2][0], static_results[2][1], "static volatility", "volatility")
plt.subplot(235)
turned_volatility_histogram_distance = winner_loser_histogram_plot(turned_results[2][0], turned_results[2][1], "turned volatility", "volatility")
plt.subplot(236)
dynamic_volatility_histogram_distancec = winner_loser_histogram_plot(dynamic_results[2][0], dynamic_results[2][1], "dynamic volatility", "volatility")
plt.show()


static_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "static", "atk_type": "random", "def_speed": "high"})
turned_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "turned", "atk_type": "random", "def_speed": "high"})
dynamic_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "dynamic", "atk_type": "random", "def_speed": "high"})
static_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "static", "atk_type": "TTA", "def_speed": "high"})
turned_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "turned", "atk_type": "TTA", "def_speed": "high"})
dynamic_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_type": "dynamic", "atk_type": "TTA", "def_speed": "high"})
print "Static defense, 4 defenders and 3 attackers, random attack had a {:.1f}% winning percentage".format(static_random_results)
print "Turned defense, 4 defenders and 3 attackers, random attack had a {:.1f}% winning percentage".format(turned_random_results)
print "Dynamic defense, 4 defenders and 3 attackers, random attack had a {:.1f}% winning percentage".format(dynamic_random_results)
print "Static defense, 4 defenders and 3 attackers, TTA attack had a {:.1f}% winning percentage".format(static_TTA_results)
print "Turned defense, 4 defenders and 3 attackers, TTA attack had a {:.1f}% winning percentage".format(turned_TTA_results)
print "Dynamic defense, 4 defenders and 3 attackers, TTA attack had a {:.1f}% winning percentage".format(dynamic_TTA_results)
"""

"""
high_speed_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high"})
high_speed_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "atk_type":"random"})
high_speed_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "atk_type":"TTA"})
high_speed_static_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"static"})
high_speed_turned_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"turned"})
high_speed_dynamic_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"dynamic"})
high_speed_static_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"static", "atk_type":"random"})
high_speed_turned_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"turned", "atk_type":"random"})
high_speed_dynamic_random_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"dynamic", "atk_type":"random"})
high_speed_static_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"static", "atk_type":"TTA"})
high_speed_turned_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"turned", "atk_type":"TTA"})
high_speed_dynamic_TTA_results = gather_results({"num_defenders": 4, "num_attackers": 3, "def_speed": "high", "def_type":"dynamic", "atk_type":"TTA"})
print "Full speed defenders had a {:.1f}% winning percentage".format(high_speed_results)
print "Full speed  defenders, static defense had a {:.1f}% winning percentage".format(high_speed_static_results)
print "Full speed defenders, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_results)
print "Full speed defenders, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_results)
print "Full speed defenders, random attack had a {:.1f}% winning percentage".format(high_speed_random_results)
print "Full speed defenders, TTA attack had a {:.1f}% winning percentage".format(high_speed_TTA_results)
print "Full speed defenders, random attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_random_results)
print "Full speed defenders, random attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_random_results)
print "Full speed defenders, random attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_random_results)
print "Full speed defenders, TTA attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_TTA_results)
print "Full speed defenders, TTA attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_TTA_results)
print "Full speed defenders, TTA attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_TTA_results)
"""

"""
high_speed_results = gather_results({"def_speed": "high"})
high_speed_random_results = gather_results({"def_speed": "high", "atk_type":"random"})
high_speed_TTA_results = gather_results({"def_speed": "high", "atk_type":"TTA"})
high_speed_static_results = gather_results({"def_speed": "high", "def_type":"static"}, smoothness=True)
high_speed_turned_results = gather_results({"def_speed": "high", "def_type":"turned"}, smoothness=True)
high_speed_dynamic_results = gather_results({"def_speed": "high", "def_type":"dynamic"}, smoothness=True)
high_speed_static_random_results = gather_results({"def_speed": "high", "def_type":"static", "atk_type":"random"})
high_speed_turned_random_results = gather_results({"def_speed": "high", "def_type":"turned", "atk_type":"random"})
high_speed_dynamic_random_results = gather_results({"def_speed": "high", "def_type":"dynamic", "atk_type":"random"})
high_speed_static_TTA_results = gather_results({"def_speed": "high", "def_type":"static", "atk_type":"TTA"})
high_speed_turned_TTA_results = gather_results({"def_speed": "high", "def_type":"turned", "atk_type":"TTA"})
high_speed_dynamic_TTA_results = gather_results({"def_speed": "high", "def_type":"dynamic", "atk_type":"TTA"})
print "Full speed defenders had a {:.1f}% winning percentage".format(high_speed_results)
print "Full speed  defenders, static defense had a {:.1f}% winning percentage".format(high_speed_static_results[0])
print "Full speed defenders, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_results[0])
print "Full speed defenders, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_results[0])
print "Full speed defenders, random attack had a {:.1f}% winning percentage".format(high_speed_random_results)
print "Full speed defenders, TTA attack had a {:.1f}% winning percentage".format(high_speed_TTA_results)
print "Full speed defenders, random attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_random_results)
print "Full speed defenders, random attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_random_results)
print "Full speed defenders, random attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_random_results)
print "Full speed defenders, TTA attack, static defense had a {:.1f}% winning percentage".format(high_speed_static_TTA_results)
print "Full speed defenders, TTA attack, turned defense had a {:.1f}% winning percentage".format(high_speed_turned_TTA_results)
print "Full speed defenders, TTA attack, dynamic defense had a {:.1f}% winning percentage".format(high_speed_dynamic_TTA_results)
plt.subplot(231)
static_circularity_histogram_distance = winner_loser_histogram_plot(high_speed_static_results[1][0], high_speed_static_results[1][1], "static circularity", "circularity")
plt.subplot(232)
turned_circularity_histogram_distance = winner_loser_histogram_plot(high_speed_turned_results[1][0], high_speed_turned_results[1][1], "turned circularity", "circularity")
plt.subplot(233)
dynamic_circularity_histogram_distancec = winner_loser_histogram_plot(high_speed_dynamic_results[1][0], high_speed_dynamic_results[1][1], "dynamic circularity", "circularity")
plt.subplot(234)
static_volatility_histogram_distance = winner_loser_histogram_plot(high_speed_static_results[2][0], high_speed_static_results[2][1], "static volatility", "volatility")
plt.subplot(235)
turned_volatility_histogram_distance = winner_loser_histogram_plot(high_speed_turned_results[2][0], high_speed_turned_results[2][1], "turned volatility", "volatility")
plt.subplot(236)
dynamic_volatility_histogram_distancec = winner_loser_histogram_plot(high_speed_dynamic_results[2][0], high_speed_dynamic_results[2][1], "dynamic volatility", "volatility")
plt.show()
"""

print "\nNUMBER OF EXTRA DEFENDERS"
zero_results = gather_results({"num_attackers": 4, "num_defenders": 4, "def_speed": "high"}, smoothness=True)
one_results = gather_results(({"num_attackers": 4, "num_defenders": 5, "def_speed": "high"}, {"num_attackers": 3, "num_defenders": 4, "def_speed": "high"}), smoothness=True)
two_results = gather_results({"num_attackers": 3, "num_defenders": 5, "def_speed": "high"}, smoothness=True)
print "0-extra defenders had a {:.1f}% winning percentage".format(zero_results[0])
print "1-extra defender had a {:.1f}% winning percentage".format(one_results[0])
print "2-extra defenders had a {:.1f}% winning percentage".format(two_results[0])
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

plt.subplot(131)
winner_loser_histogram_plot(zero_results[4][0], zero_results[4][1], "0-extra defenders final attack max minTTA", plot_type="max_minTTA")
plt.subplot(132)
winner_loser_histogram_plot(one_results[4][0], one_results[4][1], "1-extra defender final attack max minTTA", plot_type="max_minTTA")
plt.subplot(133)
winner_loser_histogram_plot(two_results[4][0], two_results[4][1], "2-extra defenders final attack max minTTA", plot_type="max_minTTA")
plt.show()


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