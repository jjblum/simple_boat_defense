import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.integrate as spi
import scipy.spatial as spatial
import multiprocessing as mp
import pylab
import math
import time

import Boat
import Strategies
import Overseer
import Metrics

SIMULATION_TYPE = "static_ring"  # "static_ring", "convoy"
WITH_PLOTTING = True
PLOT_MAIN = True
PLOT_METRIC = False
GLOBAL_DT = 0.05  # [s]
TOTAL_TIME = 120  # [s]
BOAT_COUNT = 8
ATTACKER_COUNT = 2
MAX_DEFENDERS_PER_RING = np.arange(8.0, 100.0, 2.0)
RADII_OF_RINGS = np.arange(8.0, 600.0, 4.0)
ATTACKER_REMOVAL_DISTANCE = 1.0
ASSET_REMOVAL_DISTANCE = 1.0
# TODO - tune this probability or figure out how to treat an interaction as a single interaction (perhaps spawn an object tracking the pairwise interaction)
PROB_OF_ATK_REMOVAL_PER_TICK = 0.3  # every time step this probability is applied

if WITH_PLOTTING:
    figx = 20.0
    figy = 10.0
    fig = plt.figure(figsize=(figx, figy))
    if PLOT_METRIC:
        ax_metric = fig.add_axes([figy/figx, 0.08, 0.5, 0.8], projection='polar')
        ax_metric.grid(b=True)
    if PLOT_MAIN:
        ax_main = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])  # main ax must come last for arrows to appear on it!!!
        ax_main.elev = 10
        ax_main.grid(b=False)  # no grid b/c it won't update correctly
        ax_main.set_xticks([])  # turn off axis labels b/c they wont update correctly
        ax_main.set_yticks([])  # turn off axis labels b/c they wont update correctly
        defenders_arrows = None
        attackers_arrows = None
        assets_arrows = None
    plt.ioff()
    fig.show()
    if PLOT_METRIC:
        background_metric = fig.canvas.copy_from_bbox(ax_metric.bbox)
    if PLOT_MAIN:
        background_main = fig.canvas.copy_from_bbox(ax_main.bbox)  # must be below fig.show()!
    fig.canvas.draw()


def plotSystem(assets, defenders, attackers, defenseMetric, title_string, plot_time, real_time_zero):
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
    mean_x = np.mean(assets_x)
    mean_y = np.mean(assets_y)
    relative_x = np.asarray(defenders_x + attackers_x + assets_x) - mean_x
    relative_y = np.asarray(defenders_y + attackers_y + assets_y) - mean_y
    x_max = 1.05*max(abs(relative_x))
    y_max = 1.05*max(abs(relative_y))
    square_max = max(max([x_max, y_max]), 5.0)

    axes = [mean_x-square_max, mean_x+square_max, mean_y-square_max, mean_y+square_max]
    if PLOT_MAIN:
        ax_main.plot([mean_x-square_max, mean_x+square_max], [mean_y-square_max, mean_y+square_max],
                     'x', markersize=1, markerfacecolor='white', markeredgecolor='white')
        ax_main.axis(axes)  # requires those invisible 4 corner white markers

    if PLOT_METRIC:
        fig.canvas.restore_region(background_metric)
    if PLOT_MAIN:
        fig.canvas.restore_region(background_main)

    if PLOT_METRIC:
        # plot the metric polar plot
        max_r = np.max(defenseMetric.polarPlotData[:, 1])
        min_r = np.min(defenseMetric.polarPlotData[:, 1])
        avg_r = np.mean(defenseMetric.polarPlotData[:, 1])
        # TODO - get the polar plot to update its tickmarks or the maximum radius, because they aren't updating right now
        ax_metric.set_rmax(max_r*2.0)
        ax_metric.draw_artist(ax_metric.plot(defenseMetric.polarPlotData[:, 0],
                                         defenseMetric.polarPlotData[:, 1], 'b-', linewidth=4.0)[0])
        # TODO - convert defenders_x and defenders_y into x and y relative to the asset so they can be put into polar form
        relative_x = defenders_x - mean_x
        relative_y = defenders_y - mean_y
        defenders_r = np.sqrt(np.power(relative_x, 2.) + np.power(relative_y, 2.))  # distance from asset to defender
        defenders_phi = np.arctan2(relative_y, relative_x)  # angle from asset to defender
        ax_metric.draw_artist(ax_metric.plot(defenders_phi, defenders_r, 'go', ms=14.0)[0])

        # plot inscribed circle
        ax_metric.draw_artist(ax_metric.plot(np.linspace(-np.pi, np.pi, 100), min_r*np.ones(100,), 'm-', linewidth=6.0)[0])
        # plot mean circle
        ax_metric.draw_artist(ax_metric.plot(np.linspace(-np.pi, np.pi, 100), avg_r*np.ones(100,), 'c--', linewidth=3.0)[0])

    if PLOT_MAIN:
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
                        fc="b", ec="b", head_width=0.5, head_length=1.0) for j in range(len(assets_x))]

        # TODO - put this in the metric plot instead, of course
        for d in defenders:
            if d.TTAData is not None:
                ax_main.draw_artist(ax_main.plot(d.TTAData[:, 0], d.TTAData[:, 1], 'm-')[0])
                center = d.TTAPolygon.center()
                ax_main.draw_artist(ax_main.plot(center[0], center[1], 'ms')[0])
        for a in attackers:
            ax_main.draw_artist(ax_main.plot(a.state[0] + a.state[2]*np.cos(a.state[4])*defenseMetric.timeThreshold,
                                             a.state[1] + a.state[2]*np.sin(a.state[4])*defenseMetric.timeThreshold, 'mo', markersize=14)[0])


        for boat in assets + defenders + attackers:
            if boat.plotData is not None:
                if boat.type == "asset":
                    style = 'b-'
                elif boat.type == "defender":
                    style = 'g-'
                elif boat.type == "attacker":
                    style = 'r-'
                ax_main.draw_artist(ax_main.plot(boat.plotData[:, 0], boat.plotData[:, 1], style, linewidth=0.25)[0])

    # rectangle coords
    left, width = 0.01, 0.95
    bottom, height = 0.0, 0.96
    right = left + width
    top = bottom + height
    if PLOT_MAIN:
        time_text = ax_main.text(right, top, "time = {:.2f} s".format(plot_time),
                                 horizontalalignment='right', verticalalignment='bottom',
                                 transform=ax_main.transAxes, size=20)
        asset_position_text = ax_main.text(0.5*(left + right), bottom,
                                           "asset (x,y) = {:.2f},{:.2f}".format(mean_x, mean_y),
                                           horizontalalignment='center', verticalalignment='bottom',
                                           transform=ax_main.transAxes, size=20)
        main_title_text = ax_main.text(left, top, "{s}".format(s=title_string),
                                       horizontalalignment='left', verticalalignment='bottom',
                                       transform=ax_main.transAxes, size=20)
        real_time_passed = time.time() - real_time_zero
        time_ratio = assets[0].time/real_time_passed
        time_ratio_text = ax_main.text(right, top - 0.03, "speed = {:.2f}x".format(time_ratio),
                                       horizontalalignment='right', verticalalignment='bottom',
                                       transform=ax_main.transAxes, size=20)
        ax_main.draw_artist(main_title_text)
        ax_main.draw_artist(asset_position_text)
        ax_main.draw_artist(time_text)
        ax_main.draw_artist(time_ratio_text)
        for defender_arrow in defender_arrows:
            ax_main.draw_artist(defender_arrow)
        for attacker_arrow in attacker_arrows:
            ax_main.draw_artist(attacker_arrow)
        for asset_arrow in asset_arrows:
            ax_main.draw_artist(asset_arrow)
        fig.canvas.blit(ax_main.bbox)

    if PLOT_METRIC:
        metric_title_text = ax_metric.text(0.5, 0.95, "{:.0f}s contour".format(defenseMetric._timeThreshold),
                                           horizontalalignment='center', verticalalignment='bottom',
                                           transform=ax_metric.transAxes, size=20)
        ax_metric.draw_artist(metric_title_text)
        fig.canvas.blit(ax_metric.bbox)
    # time.sleep(GLOBAL_DT/10)


def formDefenderRings(defenders):
        # defenders always start in rings around the asset
        defender_id = 0
        ring = 0
        while defender_id < len(defenders):
            defender_count_in_ring = min(len(defenders) - defender_id, MAX_DEFENDERS_PER_RING[ring])
            radius = RADII_OF_RINGS[ring]  # + np.random.uniform(-2.0, 2.0)
            angles = np.arange(0.0, 2*math.pi, 2*math.pi/defender_count_in_ring)
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
        radius = np.random.uniform(30., 40.)
        angle = np.random.uniform(0.0, 2*math.pi, 2)
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
        for b in assets:
            None # asset does nothing
            # b.strategy = Strategies.SingleSpline(b, [np.random.uniform(-30., 20.), np.random.uniform(-30., 20.)], np.random.uniform(-np.pi, np.pi), surgeVelocity=2.5)
            # b.strategy = Strategies.SingleSpline(b, [20.0, 20.0], 0, surgeVelocity=2.5, driftDown=True)
            # b.strategy = Strategies.Square(b, 1.0, 0.0, 10.0)
            # b.strategy = Strategies.FollowWaypoints(b, np.column_stack(([1, 3, -20, -10], [0, 30, -5, -3])), surgeVelocity=2.5, closed_circuit=True)
            # b.strategy = Strategies.Circle_PID(b, [0., 25.0], 25., "ccw", surgeVelocity=0.5)
            # b.strategy = Strategies.Circle_LOS(b, [0., 1.5], 1.5, "ccw", surgeVelocity=0.6)
            # b.strategy = Strategies.SpinInPlace(b, direction="cw")
            # b.strategy = Strategies.HoldHeading(b, surgeVelocity=5.)
            # b.strategy = Strategies.StrategySequence(b, [
            #    (Strategies.ChangeHeading, (b, -np.pi/2.)),
            #    (Strategies.HoldHeading, (b, 5.0))
            #])
            # TODO - turning ccw turns FASTER than turning cw???? Figure out why. Surge velocity doesn't show this.
        for b in defenders:
            None
            #if not b.uniqueID % 4:
            #    b.strategy = Strategies.MoveToClosestAttacker(b)
            #else:
            #    None
            b.strategy = Strategies.Circle_LOS(b, [0., 0.], 9.0, surgeVelocity=2.5)
            #b.strategy = Strategies.MoveToClosestAttacker(b)
        for b in attackers:
            #b.strategy = Strategies.MoveTowardAsset(b, 1.0)
            b.strategy = Strategies.TimedStrategySequence(b, [
                (Strategies.DoNothing, (b,)),
                (Strategies.MoveTowardAsset, (b,))
            ], [6.0, 1000.0])
            None
    elif type == "convoy":
        for b in assets:
            b.strategy = Strategies.HoldHeading(b, 5.0)
        for b in defenders:
            if b.uniqueID % 4:
                b.strategy = Strategies.HoldHeading(b, 1.0)
            else:
                b.strategy = Strategies.MoveToClosestAttacker(b)
        for b in attackers:
            None
            # b.strategy = Strategies.MoveTowardAsset(b, 1.0)


def main():
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

    # set up defense metric tools
    if SIMULATION_TYPE == "static_ring":
        #defenseMetric = Metrics.StaticRingMinimumTimeToArrive(assets, defenders, attackers, resolution_th=1.*np.pi/180., resolution_r=5.0, max_r=30.0, time_threshold=5.0)
        defenseMetric = Metrics.DefenderFrameTimeToArrive(assets, defenders, attackers, time_threshold=8.0)
    elif SIMULATION_TYPE == "convoy":
        #Metrics.DefenseMetric(assets, defenders, attackers)
        defenseMetric = Metrics.StaticRingMinimumTimeToArrive(assets, defenders, attackers, resolution_th=10.*np.pi/180.)

    overseer.defenseMetric = defenseMetric

    # move asset using ODE integration
    real_time_zero = time.time()
    t = 0.0
    dt = GLOBAL_DT
    step = 1

    """
    t0 = time.time()
    pool = mp.Pool(processes=min(BOAT_COUNT, 20))
    pool.map(printUID, boat_list)
    print "Basic pool.map took {} seconds".format(time.time() - t0)

    t0 = time.time()
    results = [pool.apply_async(printUID, b) for b in boat_list]
    print [result.get(timeout=1) for result in results]
    print "pool.apply_async took {} seconds".format(time.time() - t0)

    t0 = time.time()
    for b in boat_list:
        printUID(b)
    print "Simple loop took {} seconds".format(time.time() - t0)
    """

    while t < TOTAL_TIME:
        times = np.linspace(t, t+dt, 2)
        for b in boat_list:
            b.time = t
            b.control()
            states = spi.odeint(Boat.ode, b.state, times, (b,))
            b.state = states[1]
            #if b.type == "asset":
            #    print "t = {:.2f}, thdot = {:.3f}, moment = {:.2f}, ".format(t, b.state[5], b._moment)
        t += dt
        step += 1

        # update any metrics
        defenseMetric.measureCurrentState()

        # update any overseers
        overseer.updateDefense()

        if len(attackers) > 0:
            # build location arrays for defenders, attackers, and assets
            X_defenders = np.zeros((len(defenders), 2))
            X_attackers = np.zeros((len(attackers), 2))
            X_assets = np.zeros((len(assets), 2))
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
            atk_vs_asset_pairwise_distances = spatial.distance.cdist(X_assets, X_attackers)
            overseer.atk_vs_asset_pairwise_distances = atk_vs_asset_pairwise_distances  # inform the overseer
            atk_vs_asset_minimum_distance = np.min(atk_vs_asset_pairwise_distances, 1)
            if atk_vs_asset_minimum_distance < ASSET_REMOVAL_DISTANCE:
                print "Simulation ends: Asset was attacked successfully"
                break

            def_vs_atk_pairwise_distances = spatial.distance.cdist(X_defenders, X_attackers)
            overseer.def_vs_atk_pairwise_distances = def_vs_atk_pairwise_distances  # inform the overseer
            def_vs_atk_closest_distances = np.min(def_vs_atk_pairwise_distances, 0)
            # if any values are less than the distance threshold, remove the attacker
            attackers_to_be_removed = []
            for d in range(len(def_vs_atk_closest_distances)):
                if def_vs_atk_closest_distances[d] < ATTACKER_REMOVAL_DISTANCE:
                    if np.random.uniform(0., 1.) < PROB_OF_ATK_REMOVAL_PER_TICK:
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
            plotSystem(assets, defenders, attackers, defenseMetric, SIMULATION_TYPE, t, real_time_zero)
    print "Finished {} simulated seconds in {} real-time seconds".format(t,  time.time() - real_time_zero)

if __name__ == '__main__':
    main()
