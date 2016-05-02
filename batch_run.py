import sys
import os
import subprocess

NUMBER_OF_RUNS = 79
NUMBER_PER_BATCH = 25

NUMBER_OF_DEFENDERS = [5, 4]
NUMBER_OF_ATTACKERS = [4, 3]
ATTACK_TYPES = ["TTA"]  # random or TTA
DEFENSE_TYPE = ["turned"]  # static, dynamic, or turned (static but facing along the dynamic circle)
MAX_INTERCEPT_DISTANCE = [30.]  # 30. or 40.
DEFENDER_SPEED = ["high"]  # low or high

path = os.path.dirname(os.path.realpath(__file__))


for dnum in NUMBER_OF_DEFENDERS:
    for anum in NUMBER_OF_ATTACKERS:
        for atype in ATTACK_TYPES:
            for dtype in DEFENSE_TYPE:
                for mid in MAX_INTERCEPT_DISTANCE:
                    for ds in DEFENDER_SPEED:
                        procs = list()
                        n = 0
                        while n < NUMBER_OF_RUNS:
                            for i in range(min(NUMBER_PER_BATCH, NUMBER_OF_RUNS - n)):
                                proc = subprocess.Popen([sys.executable, '{}/simulation_main.py'.format(path), str(dnum),
                                                         str(anum), str(mid), atype, dtype, ds,
                                                         "./results/def{}_atk{}_maxIntDist{}_atkType{}_defType{}_defSpeed{}_{}".format(
                                                             dnum, anum, mid, atype, dtype, ds, n)])
                                procs.append(proc)
                                n += 1
                            for proc in procs:
                                proc.wait()
                            procs = list()


