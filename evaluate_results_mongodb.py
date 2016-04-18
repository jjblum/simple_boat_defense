import numpy as np
import matplotlib.pyplot as plt

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
    # Smoothness measure (smaller is smoother) - sum the absolute differences in mean minTTA divided by the final time
    # (i.e. do an np.diff, np.abs, and np.sum). Sum all rings smoothnesses together, then compute the mean of this for each
    # combination of attack and defense

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

"""
print "\nAll RUNS:"
winning_results = results.find({"defenders_win": True})
rounds = float(results.count())
wins = float(winning_results.count())
winning_results.close()
winning_ratio = wins/rounds*100.
print "The defenders had a {:.1f}% winning percentage".format(winning_ratio)

print "\nTTA VS. RANDOM ATTACK:"
TTA_results = results.find({"atk_type": "TTA"})
TTA_winning_results = results.find({"atk_type": "TTA", "defenders_win": True})
TTA_rounds = float(TTA_results.count())
TTA_wins = float(TTA_winning_results.count())
TTA_results.close()
TTA_winning_results.close()
TTA_win_ratio = TTA_wins/TTA_rounds*100.
print "Any attack using TTA had a {:.1f}% winning percentage".format(TTA_win_ratio)
random_results = results.find({"atk_type": "random"})
random_winning_results = results.find({"atk_type": "random", "defenders_win": True})
random_rounds = float(random_results.count())
random_wins = float(random_winning_results.count())
random_results.close()
random_winning_results.close()
random_win_ratio = random_wins/random_rounds*100.
print "Any random attack had a {:.1f}% winning percentage".format(random_win_ratio)
"""

print "\nSTATIC VS. DYNAMIC DEFENSE:"
static_results = results.find({"def_type": "static"})
static_winning_results = results.find({"def_type": "static", "defenders_win": True})
static_losing_results = results.find({"def_type": "static", "defenders_win": False})
static_rounds = float(static_results.count())
static_wins = float(static_winning_results.count())
static_losing_circularity, static_losing_volatility, static_losing_minTTA_final_attack_circularity = minTTA_smoothness(static_losing_results)
static_winning_circularity, static_winning_volatility, static_winning_minTTA_final_attack_circularity = minTTA_smoothness(static_winning_results)
static_results.close()
static_winning_results.close()
static_losing_results.close()
static_win_ratio = static_wins/static_rounds*100.
print "Any static defense had a {:.1f}% winning percentage".format(static_win_ratio)
dynamic_results = results.find({"def_type": "dynamic"})
dynamic_winning_results = results.find({"def_type": "dynamic", "defenders_win": True})
dynamic_losing_results = results.find({"def_type": "dynamic", "defenders_win": False})
dynamic_rounds = float(dynamic_results.count())
dynamic_wins = float(dynamic_winning_results.count())
dynamic_losing_circularity, dynamic_losing_volatility, dynamic_losing_minTTA_final_attack_circularity = minTTA_smoothness(dynamic_losing_results)
dynamic_winning_circularity, dynamic_winning_volatility, dynamic_winning_minTTA_final_attack_circularity = minTTA_smoothness(dynamic_winning_results)
dynamic_results.close()
dynamic_winning_results.close()
dynamic_losing_results.close()
dynamic_win_ratio = dynamic_wins/dynamic_rounds*100.
print "Any dynamic defense had a {:.1f}% winning percentage".format(dynamic_win_ratio)

plt.rcParams.update({'font.size': 22})
final_atk_circularity_bins = np.linspace(0., 0.4, 50)
plt.subplot(121)
n_win, bins, patches = plt.hist(static_winning_minTTA_final_attack_circularity, bins=final_atk_circularity_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(static_losing_minTTA_final_attack_circularity, bins=final_atk_circularity_bins, normed=True, facecolor='red', alpha=0.5)
plt.ylim(0., 30.)
plt.title("static final attack circularity")
plt.subplot(122)
n_win, bins, patches = plt.hist(dynamic_winning_minTTA_final_attack_circularity, bins=final_atk_circularity_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(dynamic_losing_minTTA_final_attack_circularity, bins=final_atk_circularity_bins, normed=True, facecolor='red', alpha=0.5)
plt.ylim(0., 30.)
plt.title("dynamic final attack circularity")
plt.show()

volatility_bins = np.linspace(0., 1.5, 50)
circularity_bins = np.linspace(0., 0.25, 50)
plt.subplot(221)
n_win, bins, patches = plt.hist(static_winning_circularity, bins=circularity_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(static_losing_circularity, bins=circularity_bins, normed=True, facecolor='red', alpha=0.5)
static_circularity_histogram_distance = chi2_distance(n_win, n_lose)
plt.title("static circularity")
plt.ylim(0., 35.)
plt.subplot(222)
n_win, bins, patches = plt.hist(dynamic_winning_circularity, bins=circularity_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(dynamic_losing_circularity, bins=circularity_bins, normed=True, facecolor='red', alpha=0.5)
dynamic_circularity_histogram_distance = chi2_distance(n_win, n_lose)
plt.title("dynamic circularity")
plt.ylim(0., 35.0)
plt.subplot(223)
n_win, bins, patches = plt.hist(static_winning_volatility, bins=volatility_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(static_losing_volatility, bins=volatility_bins, normed=True, facecolor='red', alpha=0.5)
static_volatility_histogram_distance = chi2_distance(n_win, n_lose)
plt.title("static volatility")
plt.ylim(0., 4.5)
plt.subplot(224)
n_win, bins, patches = plt.hist(dynamic_winning_volatility, bins=volatility_bins, normed=True, facecolor='green', alpha=0.5)
n_lose, bins, patches = plt.hist(dynamic_losing_volatility, bins=volatility_bins, normed=True, facecolor='red', alpha=0.5)
dynamic_volatility_histogram_distance = chi2_distance(n_win, n_lose)
plt.title("dynamic volatility")
plt.ylim(0., 4.5)
print "\nTTA SMOOTHNESS MEASURES:"
print "Static circularity chi-squared distance between winners and losers = {:.3f}".format(static_circularity_histogram_distance)
print "Dynamic circularity chi-squared distance between winners and losers = {:.3f}".format(dynamic_circularity_histogram_distance)
print "Static voltatility chi-squared distance between winners and losers = {:.3f}".format(static_volatility_histogram_distance)
print "Dynamic voltatility chi-squared distance between winners and losers = {:.3f}".format(dynamic_volatility_histogram_distance)
plt.show()


print "\nTTA ATTACK, STATIC VS. DYNAMIC DEFENSE:"
TTA_static_results = results.find({"atk_type": "TTA", "def_type": "static"})
TTA_static_winning_results = results.find({"atk_type": "TTA", "defenders_win": True, "def_type": "static"})
TTA_static_rounds = float(TTA_static_results.count())
TTA_static_wins = float(TTA_static_winning_results.count())
TTA_static_results.close()
TTA_static_winning_results.close()
TTA_static_win_ratio = TTA_static_wins/TTA_static_rounds*100.
print "TTA attack, static defense had a {:.1f}% winning percentage".format(TTA_static_win_ratio)
TTA_dynamic_results = results.find({"atk_type": "TTA", "def_type": "dynamic"})
TTA_dynamic_winning_results = results.find({"atk_type": "TTA", "defenders_win": True, "def_type": "dynamic"})
TTA_dynamic_rounds = float(TTA_dynamic_results.count())
TTA_dynamic_wins = float(TTA_dynamic_winning_results.count())
TTA_dynamic_results.close()
TTA_dynamic_winning_results.close()
TTA_dynamic_win_ratio = TTA_dynamic_wins/TTA_dynamic_rounds*100.
print "TTA attack, dynamic defense had a {:.1f}% winning percentage".format(TTA_dynamic_win_ratio)

print "\nRANDOM ATTACK, STATIC VS. DYNAMIC DEFENSE:"
random_static_results = results.find({"atk_type": "random", "def_type": "static"})
random_static_winning_results = results.find({"atk_type": "random", "defenders_win": True, "def_type": "static"})
random_static_rounds = float(random_static_results.count())
random_static_wins = float(random_static_winning_results.count())
random_static_results.close()
random_static_winning_results.close()
random_static_win_ratio = random_static_wins/random_static_rounds*100.
print "random attack, static defense had a {:.1f}% winning percentage".format(random_static_win_ratio)
random_dynamic_results = results.find({"atk_type": "random", "def_type": "dynamic"})
random_dynamic_winning_results = results.find({"atk_type": "random", "defenders_win": True, "def_type": "dynamic"})
random_dynamic_rounds = float(random_dynamic_results.count())
random_dynamic_wins = float(random_dynamic_winning_results.count())
random_dynamic_results.close()
random_dynamic_winning_results.close()
random_dynamic_win_ratio = random_dynamic_wins/random_dynamic_rounds*100.
print "random attack, dynamic defense had a {:.1f}% winning percentage".format(random_dynamic_win_ratio)

print "\n***** Notice how dynamic defense mitigates the advantage that TTA knowledge provides the attackers *****"

print "\n0 EXTRA DEFENDERS VS. 1 EXTRA DEFENDER VS. 2 EXTRA DEFENDERS:"
zero_defender_results = results.find({"num_attackers": 4, "num_defenders": 4})
zero_defender_winning_results = results.find({"num_attackers": 4, "num_defenders": 4, "defenders_win": True})
zero_defender_rounds = float(zero_defender_results.count())
zero_defender_wins = float(zero_defender_winning_results.count())
zero_defender_results.close()
zero_defender_winning_results.close()
zero_defender_win_ratio = zero_defender_wins/zero_defender_rounds*100.
print "0 extra defenders had a {:.1f}% winning percentage".format(zero_defender_win_ratio)
one_defender_results = results.find({"num_attackers": 4, "num_defenders": 5}, {"num_attackers": 3, "num_defenders": 4})
one_defender_winning_results = results.find({"num_attackers": 4, "num_defenders": 5, "defenders_win": True}, {"num_attackers": 3, "num_defenders": 4, "defenders_win": True})
one_defender_rounds = float(one_defender_results.count())
one_defender_wins = float(one_defender_winning_results.count())
one_defender_results.close()
one_defender_winning_results.close()
one_defender_win_ratio = one_defender_wins/one_defender_rounds*100.
print "1 extra defender had a {:.1f}% winning percentage".format(one_defender_win_ratio)
two_defender_results = results.find({"num_attackers": 3, "num_defenders": 5})
two_defender_winning_results = results.find({"num_attackers": 3, "num_defenders": 5, "defenders_win": True})
two_defender_rounds = float(two_defender_results.count())
two_defender_wins = float(two_defender_winning_results.count())
two_defender_results.close()
two_defender_winning_results.close()
two_defender_win_ratio = two_defender_wins/two_defender_rounds*100.
print "2 extra defenders had a {:.1f}% winning percentage".format(two_defender_win_ratio)

print "\n30 VS. 40 MAX INTERCEPT DISTANCE:"
thirty_max_intercept_distance_results = results.find({"max_allowable_intercept_distance": 30.})
thirty_max_intercept_distance_winning_results = results.find({"max_allowable_intercept_distance": 30., "defenders_win": True})
thirty_max_intercept_distance_rounds = float(thirty_max_intercept_distance_results.count())
thirty_max_intercept_distance_wins = float(thirty_max_intercept_distance_winning_results.count())
thirty_max_intercept_distance_results.close()
thirty_max_intercept_distance_winning_results.close()
thirty_max_intercept_distance_win_ratio = thirty_max_intercept_distance_wins/thirty_max_intercept_distance_rounds*100.
print "Max intercept distance = 30 had a {:.1f}% winning percentage".format(thirty_max_intercept_distance_win_ratio)
forty_max_intercept_distance_results = results.find({"max_allowable_intercept_distance": 40.})
forty_max_intercept_distance_winning_results = results.find({"max_allowable_intercept_distance": 40., "defenders_win": True})
forty_max_intercept_distance_rounds = float(forty_max_intercept_distance_results.count())
forty_max_intercept_distance_wins = float(forty_max_intercept_distance_winning_results.count())
forty_max_intercept_distance_results.close()
forty_max_intercept_distance_winning_results.close()
forty_max_intercept_distance_win_ratio = forty_max_intercept_distance_wins/forty_max_intercept_distance_rounds*100.
print "Max intercept distance = 40 had a {:.1f}% winning percentage".format(forty_max_intercept_distance_win_ratio)

results.close()