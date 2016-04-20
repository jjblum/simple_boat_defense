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
    for result in db_results:
        final_time = result["final_time"]
        minTTA = cp.loads(result["minTTA"])
        meanOverTime_minTTA = np.mean(minTTA, axis=0)
        meanOverAngle_minTTA = np.mean(minTTA, axis=2)
        volatility.append(np.sum(np.abs(np.diff(meanOverAngle_minTTA, axis=0)))/final_time)  # the change per second through time - like a measure of volatility
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
    return circularity, volatility, final_attack_circularity


def gather_results(dictionary, collection=results, smoothness=False):
    winning_dict = copy.deepcopy(dictionary)
    winning_dict.update({"defenders_win": True})
    losing_dict = copy.deepcopy(dictionary)
    losing_dict.update({"defenders_win": False})
    general_results = collection.find(dictionary)
    winning_results = collection.find(winning_dict)
    losing_results = collection.find(losing_dict)
    rounds = float(general_results.count())
    wins = float(winning_results.count())
    wins_ratio = wins/rounds*100.
    if smoothness:
        losing_circularity, losing_volatility, losing_final_atk_circularity = minTTA_smoothness(losing_results)
        winning_circularity, winning_volatility, winning_final_atk_circularity = minTTA_smoothness(winning_results)
        general_results.close()
        winning_results.close()
        losing_results.close()
        return wins_ratio, (winning_circularity, losing_circularity), (winning_volatility, losing_volatility), (winning_final_atk_circularity, losing_final_atk_circularity)
    else:
        return wins_ratio

def winner_loser_histogram_plot(winners, losers, title_string, plot_type="final_atk"):
    if plot_type == "final_atk":
        bins = np.linspace(0., 0.4, 50)
        ylim = 30.
    elif plot_type == "circularity":
        bins = np.linspace(0., 0.25, 50)
        plt.ylim(0., 35.)
    elif plot_type == "volatility":
        bins = np.linspace(0., 1.5, 50)
        plt.ylim(0., 4.5)
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


print "\nTTA VS. RANDOM ATTACK:"
TTA_results = gather_results({"atk_type": "TTA"})
print "Any attack using TTA had a {:.1f}% winning percentage".format(TTA_results)
random_results = gather_results({"atk_type": "random"})
print "Any random attack had a {:.1f}% winning percentage".format(random_results)


print "\nSTATIC VS. DYNAMIC DEFENSE:"
static_results = gather_results({"def_type": "static"}, smoothness=True)
print "Any static defense had a {:.1f}% winning percentage".format(static_results[0])
dynamic_results = gather_results({"def_type": "dynamic"}, smoothness=True)
print "Any dynamic defense had a {:.1f}% winning percentage".format(dynamic_results[0])
plt.rcParams.update({'font.size': 22})
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


print "\n***** Notice how dynamic defense mitigates the advantage that TTA knowledge provides the attackers *****"


print "\n0 EXTRA DEFENDERS VS. 1 EXTRA DEFENDER VS. 2 EXTRA DEFENDERS:"
zero_defender_results = gather_results({"num_attackers": 4, "num_defenders": 4}, smoothness=True)
print "0 extra defenders had a {:.1f}% winning percentage".format(zero_defender_results[0])
"""
one_defender_results = gather_results({"num_attackers": 4, "num_defenders": 5}, {"num_attackers": 3, "num_defenders": 4}, smoothness=True)
print "1 extra defender had a {:.1f}% winning percentage".format(one_defender_results)
two_defender_results = gather_results({"num_attackers": 3, "num_defenders": 5})
print "2 extra defenders had a {:.1f}% winning percentage".format(two_defender_results)


print "\n30 VS. 40 MAX INTERCEPT DISTANCE:"
thirty_max_intercept_distance_results = gather_results({"max_allowable_intercept_distance": 30.})
print "Max intercept distance = 30 had a {:.1f}% winning percentage".format(thirty_max_intercept_distance_results)
forty_max_intercept_distance_results = gather_results({"max_allowable_intercept_distance": 40.})
print "Max intercept distance = 40 had a {:.1f}% winning percentage".format(forty_max_intercept_distance_results)

results.close()