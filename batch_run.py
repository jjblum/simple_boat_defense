import sys
import os
import subprocess

NUMBER_OF_RUNS = 100
NUMBER_PER_BATCH = 20

NUMBER_OF_DEFENDERS = [4]
NUMBER_OF_ATTACKERS = [3]
ATTACK_TYPES = ["random", "TTA"]  # random or TTA
DEFENSE_TYPE = ["static", "dynamic"]  # static or dynamic
MAX_INTERCEPT_DISTANCE = [40.]
DEFENDER_MASS = ["high"]  # low or high

path = os.path.dirname(os.path.realpath(__file__))


for dnum in NUMBER_OF_DEFENDERS:
    for anum in NUMBER_OF_ATTACKERS:
        for atype in ATTACK_TYPES:
            for dtype in DEFENSE_TYPE:
                for mid in MAX_INTERCEPT_DISTANCE:
                    for dm in DEFENDER_MASS:
                        procs = list()
                        n = 0
                        while n < NUMBER_OF_RUNS:
                            for i in range(min(NUMBER_PER_BATCH, NUMBER_OF_RUNS - n)):
                                proc = subprocess.Popen([sys.executable, '{}/simulation_main.py'.format(path), str(dnum),
                                                         str(anum), str(mid), atype, dtype, dm,
                                                         "./results/def{}_atk{}_maxIntDist{}_atkType{}_defType{}_defMass{}_{}".format(
                                                             dnum, anum, mid, atype, dtype, dm, n)])
                                procs.append(proc)
                                n += 1
                            for proc in procs:
                                proc.wait()
                            procs = list()


