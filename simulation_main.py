import numpy
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.spatial as spatial
import pylab
import math
import time

import Boat
import Strategies
import Overseer

SIMULATION_TYPE = "static_ring"  # "static_ring", "convoy"
WITH_PLOTTING = True
GLOBAL_DT = 0.01  # [s]
TOTAL_TIME = 120  # [s]
BOAT_COUNT = 10
ATTACKER_COUNT = 4
MAX_DEFENDERS_PER_RING = numpy.arange(10.0, 100.0, 2.0)
RADII_OF_RINGS = numpy.arange(4.0, 600.0, 4.0)
ATTACKER_REMOVAL_DISTANCE = 1.0
ASSET_REMOVAL_DISTANCE = 1.0

if WITH_PLOTTING:
    figx = 20.0
    figy = 10.0
    fig = plt.figure(figsize=(figx, figy))
    # ax = fig.add_subplot(111)
    ax = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])
    ax.elev = 10
    ax.grid(b=False)  # no grid b/c it won't update correctly
    ax.set_xticks([])  # turn off axis labels b/c they wont update correctly
    ax.set_yticks([])  # turn off axis labels b/c they wont update correctly
    defenders_arrows = None
    attackers_arrows = None
    assets_arrows = None
    plt.ioff()
    fig.show()
    background = fig.canvas.copy_from_bbox(ax.bbox)  # must be below fig.show()!
    fig.canvas.draw()


def plotSystem(assets, defenders, attackers, title_string, plot_time):
    # gather the X and Y locations of all boats
    defenders_x = [boat.state[0] for boat in defenders]
    attackers_x = [boat.state[0] for boat in attackers]
    assets_x = [boat.state[0] for boat in assets]
    defenders_y = [boat.state[1] for boat in defenders]
    attackers_y = [boat.state[1] for boat in attackers]
    assets_y = [boat.state[1] for boat in assets]
    defenders_th = [boat.state[4] for boat in defenders]
    attackers_th = [boat.state[4] for boat in attackers]
    assets_th = [boat.state[4] for boat in assets]

    # find the bounds
    mean_x = numpy.mean(assets_x)
    mean_y = numpy.mean(assets_y)
    relative_x = numpy.asarray(defenders_x + attackers_x + assets_x) - mean_x
    relative_y = numpy.asarray(defenders_y + attackers_y + assets_y) - mean_y
    x_max = 1.05*max(abs(relative_x))
    y_max = 1.05*max(abs(relative_y))
    square_max = max(max([x_max, y_max]), 5.0)

    axes = [mean_x-square_max, mean_x+square_max, mean_y-square_max, mean_y+square_max]
    plt.plot([mean_x-square_max, mean_x+square_max], [mean_y-square_max, mean_y+square_max],
             'x', markersize=1, markerfacecolor='white', markeredgecolor='white')
    plt.axis(axes)  # requires those invisible 4 corner white markers

    fig.canvas.restore_region(background)

    defender_arrows = [pylab.arrow(defenders_x[j], defenders_y[j],
                                   0.05*math.cos(defenders_th[j]),
                                   0.05*math.sin(defenders_th[j]),
                                   fc="g", ec="g", head_width=0.5, head_length=1.0) for j in range(len(defenders_x))]
    attacker_arrows = [pylab.arrow(attackers_x[j], attackers_y[j],
                                   0.05*math.cos(attackers_th[j]),
                                   0.05*math.sin(attackers_th[j]),
                                   fc="r", ec="r", head_width=0.5, head_length=1.0) for j in range(len(attackers_x))]
    asset_arrows = [pylab.arrow(assets_x[j], assets_y[j],
                    0.05*math.cos(assets_th[j]),
                    0.05*math.sin(assets_th[j]),
                    fc="b", ec="b", head_width=1.5, head_length=2.5) for j in range(len(assets_x))]

    for boat in assets + defenders + attackers:
        if boat.plotData is not None:
            if boat.type == "asset":
                style = 'b-'
            elif boat.type == "defender":
                style = 'g-'
            elif boat.type == "attacker":
                style = 'r-'
            ax.draw_artist(ax.plot(boat.plotData[:, 0], boat.plotData[:, 1], style, linewidth=0.1)[0])

    # plt.title(title_string + " time = {tt} s".format(tt=plot_time))
    # rectangle coords
    left, width = 0.01, 0.95
    bottom, height = 0.0, 0.96
    right = left + width
    top = bottom + height
    time_text = ax.text(right, top, "time = {:.2f} s".format(plot_time),
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, size=20)
    asset_position_text = ax.text(0.5*(left + right), bottom,
                                  "asset (x,y) = {:.2f},{:.2f}".format(mean_x, mean_y),
                                  horizontalalignment='center', verticalalignment='bottom',
                                  transform=ax.transAxes, size=20)
    title_text = ax.text(left, top, "{s}".format(s=title_string),
                         horizontalalignment='left', verticalalignment='bottom',
                         transform=ax.transAxes, size=20)
    real_time_passed = time.time() - real_time_zero
    time_ratio = assets[0].time/real_time_passed
    time_ratio_text = ax.text(right, top - 0.03, "speed = {:.2f}x".format(time_ratio),
                              horizontalalignment='right', verticalalignment='bottom',
                              transform=ax.transAxes, size=20)
    ax.draw_artist(title_text)
    ax.draw_artist(asset_position_text)
    ax.draw_artist(time_text)
    ax.draw_artist(time_ratio_text)
    for defender_arrow in defender_arrows:
        ax.draw_artist(defender_arrow)
    for attacker_arrow in attacker_arrows:
        ax.draw_artist(attacker_arrow)
    for asset_arrow in asset_arrows:
        ax.draw_artist(asset_arrow)
    fig.canvas.blit(ax.bbox)
    # time.sleep(GLOBAL_DT/10)


def formDefenderRings(defenders):
        # defenders always start in rings around the asset
        defender_id = 0
        ring = 0
        while defender_id < len(defenders):
            defender_count_in_ring = min(len(defenders) - defender_id, MAX_DEFENDERS_PER_RING[ring])
            radius = RADII_OF_RINGS[ring]
            angles = numpy.arange(0.0, 2*math.pi, 2*math.pi/defender_count_in_ring)
            if len(angles) == 0:
                angles = [0.0]
            for angle in angles:
                defenders[defender_id].state[0] = radius*math.cos(angle)
                defenders[defender_id].state[1] = radius*math.sin(angle)
                defenders[defender_id].state[4] = math.atan2(defenders[defender_id].state[1],
                                                             defenders[defender_id].state[0])
                defender_id += 1
            ring += 1


def randomAttackers(attackers):
    # attackers always start at some uniform random polar location, fixed radius 30
    for b in attackers:
        radius = 20.0
        angle = numpy.random.uniform(0.0, 2*math.pi, 2)
        x = radius*math.cos(angle[0])
        y = radius*math.sin(angle[0])
        b.state[0] = x
        b.state[1] = y
        b.state[4] = angle[1]


def initialPositions(assets, defenders, attackers, type="static_ring"):
    if type == "static_ring":
        formDefenderRings(defenders)
        randomAttackers(attackers)

    elif type == "convoy":
        formDefenderRings(defenders)
        for b in defenders:
            b.state[4] = 0.0  # align with asset
        randomAttackers(attackers)


def initialStrategy(assets, defenders, attackers, type="static_ring"):
    if type == "static_ring":
        for b in boat_list:
            if b.type == "asset":
                None # asset does nothing
                b.strategy = Strategies.SingleSpline(b, [-5.0, -10.0], math.pi, surgeVelocity=0.5)
                # b.strategy = Strategies.SingleSpline(b, [100.0, 0.0], 0, surgeVelocity=1.5)
                # b.strategy = Strategies.Square(b, 1.0, 0.0, 10.0)
            if b.type == "defender":
                #b.strategy = Strategies.MoveToClosestAttacker(b)
                None
            elif b.type == "attacker":
                #b.strategy = Strategies.MoveTowardAsset(b, 1.0)
                None
    elif type == "convoy":
        for b in boat_list:
            if b.type == "asset":
                b.strategy = Strategies.HoldHeading(b, 1.0)
            elif b.type == "defender":
                if b.uniqueID % 2:
                    b.strategy = Strategies.HoldHeading(b, 1.0)
                else:
                    b.strategy = Strategies.MoveToClosestAttacker(b)
            else:
                b.strategy = Strategies.MoveTowardAsset(b, 1.0)


if __name__ == '__main__':
    # spawn boats objects

    boat_list = [Boat.Boat() for i in range(0, BOAT_COUNT)]

    # set boat types
    boat_list[0].type = "asset"
    for b in boat_list[-1 - ATTACKER_COUNT:-1]:
        b.type = "attacker"

    attackers = [b for b in boat_list if b.type == "attacker"]
    defenders = [b for b in boat_list if b.type == "defender"]
    assets = [b for b in boat_list if b.type == "asset"]
    overseer = Overseer.Overseer(assets, defenders, attackers)
    overseer.attackers = attackers
    overseer.defenders = defenders
    overseer.assets = assets
    for b in boat_list:
        b.boatList = boat_list
        b.attackers = attackers
        b.defenders = defenders
        b.assets = assets

    # set initial positions and strategies
    initialPositions(assets, defenders, attackers, SIMULATION_TYPE)
    initialStrategy(assets, defenders, attackers, SIMULATION_TYPE)

    # move asset using ODE integration
    real_time_zero = time.time()
    t = 0.0
    dt = GLOBAL_DT
    step = 1

    while t < TOTAL_TIME:
        times = numpy.linspace(t, t+dt, 2)
        for b in boat_list:
            b.time = t
            b.control()
            states = spi.odeint(Boat.ode, b.state, times, (b,))
            b.state = states[1]
            if b.type == "asset": print b.state
        t += dt
        step += 1

        if len(attackers) > 0:
            # build location arrays for defenders, attackers, and assets
            X_defenders = numpy.zeros((len(defenders), 2))
            X_attackers = numpy.zeros((len(attackers), 2))
            X_assets = numpy.zeros((len(assets), 2))
            for j in range(len(defenders)):
                boat = defenders[j]
                X_defenders[j, 0] = boat.state[0]
                X_defenders[j, 1] = boat.state[1]
            for j in range(len(attackers)):
                boat = attackers[j]
                X_attackers[j, 0] = boat.state[0]
                X_attackers[j, 1] = boat.state[1]
            for j in range(len(assets)):
                boat = assets[j]
                X_assets[j, 0] = boat.state[0]
                X_assets[j, 1] = boat.state[1]
            # scipy.spatial.distance.cdist
            atk_vs_asset_pairwise_distances = spatial.distance.cdist(X_assets, X_attackers)
            overseer.atk_vs_asset_pairwise_distances = atk_vs_asset_pairwise_distances  # inform the overseer
            atk_vs_asset_minimum_distance = numpy.min(atk_vs_asset_pairwise_distances, 1)
            if atk_vs_asset_minimum_distance < ASSET_REMOVAL_DISTANCE:
                print "Simulation ends: Asset was attacked successfully"
                break

            def_vs_atk_pairwise_distances = spatial.distance.cdist(X_defenders, X_attackers)
            overseer.def_vs_atk_pairwise_distances = def_vs_atk_pairwise_distances  # inform the overseer
            def_vs_atk_closest_distances = numpy.min(def_vs_atk_pairwise_distances, 0)
            # if any values are less than the distance threshold, remove the attacker
            attackers_to_be_removed = []
            for d in range(len(def_vs_atk_closest_distances)):
                if def_vs_atk_closest_distances[d] < ATTACKER_REMOVAL_DISTANCE:
                    attackers_to_be_removed.append(d)
                    # everything references these main lists, so this should cover all shared memory
            for attacker in reversed(attackers_to_be_removed):
                del attackers[attacker] # need to delete in backwards order to avoid index conflicts
        else:
            # end the simulation
            print "Simulation Ends: All attackers removed"
            break

        # TODO - may be easy to speed this up when an attacker is obviously outside the bounding box of all defenders
        #        with respect to the asset in the center

        if WITH_PLOTTING:
            plotSystem(assets, defenders, attackers, "ATTAAAAAACK!!!!", t)
    print "Finished {} simulated seconds in {} real-time seconds".format(t,  time.time() - real_time_zero)