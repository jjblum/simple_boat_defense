import numpy as np
import matplotlib.pyplot as plt

import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import cPickle as cp

client = pymongo.MongoClient('localhost', 27017)
db = client.TTA
results = db.results

"""
meta_data = dict()
NUMBER_OF_DEFENDERS = [4, 5]
NUMBER_OF_ATTACKERS = [3, 4]
ATTACK_TYPES = ["random", "TTA"]  # random or TTA
DEFENSE_TYPE = ["static", "dynamic"]  # static or dynamic
MAX_INTERCEPT_DISTANCE = [30., 40.]
for dnum in NUMBER_OF_DEFENDERS:
    for anum in NUMBER_OF_ATTACKERS:
        for atype in ATTACK_TYPES:
            for dtype in DEFENSE_TYPE:
                for mid in MAX_INTERCEPT_DISTANCE:
                    meta_data["num_defenders"] = dnum
                    meta_data["num_attackers"] = anum
                    meta_data["atk_type"] = atype
                    meta_data["def_type"] = dtype
                    meta_data["max_allowable_intercept_distance"] = mid
                    meta_data_string = "{}".format(meta_data)
                    rounds = results.find(meta_data).count()

                    meta_data.update({"defenders_win": True})
                    wins = results.find(meta_data).count()
                    print "{}: {} wins out of {} rounds".format(meta_data_string, wins, rounds)

                    meta_data.clear()
"""

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
random_win_ratio = random_wins/random_rounds*100.
random_results.close()
random_winning_results.close()
print "Any random attack had a {:.1f}% winning percentage".format(random_win_ratio)

print "\nSTATIC VS. DYNAMIC DEFENSE:"
static_results = results.find({"def_type": "static"})
static_winning_results = results.find({"def_type": "static", "defenders_win": True})
static_rounds = float(static_results.count())
static_wins = float(static_winning_results.count())
static_win_ratio = static_wins/static_rounds*100.
print "Any static defense had a {:.1f}% winning percentage".format(static_win_ratio)
dynamic_rounds = float(results.find({"def_type": "dynamic"}).count())
dynamic_wins = float(results.find({"def_type": "dynamic", "defenders_win": True}).count())
dynamic_win_ratio = dynamic_wins/dynamic_rounds*100.
print "Any dynamic defense had a {:.1f}% winning percentage".format(dynamic_win_ratio)

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