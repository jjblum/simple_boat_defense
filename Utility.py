import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import math
import PiazziSpline


def splineToEuclidean2D(spline_coeffs, u):
    """
    :param spline_coeffs: list of spline coefficients [((a0-a7), (b0-b7)),...]
    :param u: u = [0, spline_count] along the spline (chain) at which to evaluate the derivative
    :return: (x, y)
    """
    if not isinstance(spline_coeffs, list):
        spline_coeffs = [spline_coeffs]
    spline_index = max(0, int(u - 1.e-12))
    a = spline_coeffs[spline_index][0]  # subtract a tiny number to avoid a perfect integer
    b = spline_coeffs[spline_index][1]
    S = np.power(u - np.floor(u), np.arange(0., 8.))
    return np.sum(np.array(a)*S), np.sum(np.array(b)*S)


def polyInterpOfSplineToEuclidean2D(u_quad, D_quad, u):
    """
    Interpolation of what euclidean distance should be if we fit a polynomial to the distances of the s_quad values
    :param u_quad: u1, u2, u3: the three u values used as the basis for interpolation
    :param D_quad: D(u1), D(u2), D(u3): the three distances associated with the s_quad points
    :param u: u = [0, 1] along the spline at which to evaluate the derivative
    :return: P(s) - the estimated euclidean distance at s
    """
    P = 0.0
    P += (u - u_quad[1])*(u - u_quad[2])/((u_quad[0] - u_quad[1])*(u_quad[0] - u_quad[2]))*D_quad[0]
    P += (u - u_quad[0])*(u - u_quad[2])/((u_quad[1] - u_quad[0])*(u_quad[1] - u_quad[2]))*D_quad[1]
    P += (u - u_quad[0])*(u - u_quad[1])/((u_quad[2] - u_quad[0])*(u_quad[2] - u_quad[1]))*D_quad[2]
    return P


def dD_du(spline_coeffs, u, x, y):
    """
    :param spline_coeffs: list of spline coefficients [((a0-a7), (b0-b7)),...]
    :param u: u = [0, 1] along the spline at which to evaluate the derivative
    :return: (dD_ds, dD_ds2): first derivative of the sq. euclidean distance evaluated at s, 2nd deriv.
    """
    if not isinstance(spline_coeffs, list):
        spline_coeffs = [spline_coeffs]

    spline_index = max(0, int(u - 1.e-8))
    a = spline_coeffs[spline_index][0]
    b = spline_coeffs[spline_index][1]
    powers = np.arange(0., 8.)
    S = np.power(u - np.floor(u), powers)
    sx = np.sum(np.array(a)*S)
    sy = np.sum(np.array(b)*S)
    D = spatial.distance.sqeuclidean([x, y], [sx, sy])
    dx_du = np.sum(powers[1:]*a[1:]*S[:-1])
    dy_du = np.sum(powers[1:]*b[1:]*S[:-1])
    dx_du2 = np.sum(powers[1:-1]*powers[2:]*a[2:]*S[:-2])
    dy_du2 = np.sum(powers[1:-1]*powers[2:]*b[2:]*S[:-2])
    dD_du_x = 2.*(sx - x)*dx_du
    dD_du_y = 2.*(sy - y)*dy_du
    dD_du2_x = 2.*np.power(dx_du, 2.) + 2.*(sx - x)*dx_du2
    dD_du2_y = 2.*np.power(dy_du, 2.) + 2.*(sy - y)*dy_du2
    return D, dD_du_x + dD_du_y, dD_du2_x + dD_du2_y


def closestPointOnSpline2D(length, sx, sy, x, y, u, coeffs, guess=None):
    """
    Robust and Efficient Computation of the Closest Point on a Spline Curve
    Wang et. al.

    Required for LOS control.

    Two phases: 1) quadratic optimization to generate rough estimates
                2) newton's method to quickly converge

    :param length: array = [0, spline total length]
    :param sx: spline x's
    :param sy: spline y's
    :param x: point to test x
    :param y: point to test y
    :return: s* from interval [0, total length], point closest to input (x,y), distance to the spline
    """
    N = len(sx)
    min_guess_halfwidth = np.floor(N/20.0)
    u_quad = np.zeros((4,))
    D_quad = np.zeros((3,))
    P_quad = np.zeros((4,))
    X = np.atleast_2d(np.array([x, y]))
    if not isinstance(coeffs, list):
        coeffs = [coeffs]

    # TODO - address the situation where guess is not None
    #if guess is None:
    # need to do something slower b/c we have no idea where to start
    S = np.column_stack((sx, sy))
    euclidean_distances = spatial.distance.cdist(S, X)
    closest = np.argmin(euclidean_distances)
    closest_distance = euclidean_distances[closest]
    if closest == 0:
        S = np.array([sx[0], sy[0]])
        return 0.0, S, spatial.distance.euclidean(S, X)
    elif closest == len(sx) - 1:
        S = np.array([sx[-1], sy[-1]])
        return 1.0, S, spatial.distance.euclidean(S, X)
    elif min_guess_halfwidth <= closest <= N - min_guess_halfwidth - 1:
        u_quad[0] = u[closest - min_guess_halfwidth]
        u_quad[1] = u[closest]
        u_quad[2] = u[closest + min_guess_halfwidth]
    elif closest < min_guess_halfwidth:
        u_quad[0] = u[0]
        u_quad[1] = u[min_guess_halfwidth]
        u_quad[2] = u[2*min_guess_halfwidth]
    elif closest > N - min_guess_halfwidth:
        u_quad[0] = u[N - 2*min_guess_halfwidth - 1]
        u_quad[1] = u[N - min_guess_halfwidth - 1]
        u_quad[2] = u[N - 1]

    # ## QUADRATIC PHASE - 4 iterations
    u_ij = np.zeros((3, 3))
    y_ij = np.zeros((3, 3))
    for iteration in range(4):
        for i in range(3):
            for j in range(3):
                u_ij[i, j] = u_quad[i] - u_quad[j]
                y_ij[i, j] = np.power(u_quad[i], 2) - np.power(u_quad[j], 2)
        for q in range(3):
            D_quad[q] = spatial.distance.sqeuclidean(X, np.atleast_2d(splineToEuclidean2D(coeffs, u_quad[q])))

        u_quad[3] = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                        (u_ij[1, 2]*D_quad[0] + u_ij[2, 0]*D_quad[1] + u_ij[0, 1]*D_quad[2])
        for p in range(4):
            P_quad[p] = polyInterpOfSplineToEuclidean2D(u_quad, D_quad, u_quad[p])
            P_quad[np.where(P_quad < -1.e-8)] = np.inf  # do not allow negative P values
        #print "u_quad = {}, P_quad = {}".format(u_quad, P_quad)
        # first, sort based on P value so u_quad[1:3] has the 3 lowest P values
        # then, resort u_quad[0:3] so that the s values are in increasing order
        u_quad[0:3] = np.sort(u_quad[np.argsort(P_quad)[0:3]])

    u_star = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                 (u_ij[1, 2]*D_quad[0] + u_ij[2, 0]*D_quad[1] + u_ij[0, 1]*D_quad[2])

    # check that distance to u_star isn't more than closest euclidean point
    # if it is, something went terribly wrong
    # just use the closest euclidean point
    post_quadratic_S = splineToEuclidean2D(coeffs, u_star)
    post_quadratic_distance = spatial.distance.euclidean(post_quadratic_S, X)
    if post_quadratic_distance > closest_distance:
        print "WARNING: quadratic phase is incorrect, using euclidean distance guess instead"
        return u[closest], np.array([sx[closest], sy[closest]]), closest_distance


    if np.isnan(u_star) or u_star > u[-1]:
        S = np.array([sx[-1], sy[-1]])
        return u[-1], S, spatial.distance.euclidean(X, S)

    # ## NEWTONS PHASE - until convergence
    convergence = np.inf
    newtonCount = 0
    while convergence > 1.0e-8 and newtonCount < 100:
        newtonCount += 1
        D, dD_du_, dD_du2_ = dD_du(coeffs, u_star, x, y)
        u_star_old = u_star
        u_star_change = -dD_du_/dD_du2_
        if u_star + u_star_change < 0 or u_star + u_star_change > u[-1]:  # prevent Newton's method explosions
            signs = [-1, 1]
            u_star_change = signs[np.random.randint(0, 1)]*np.random.uniform(0.0005, 0.001)
        u_star = u_star + u_star_change
        convergence = np.abs(u_star - u_star_old)

    result = splineToEuclidean2D(coeffs, u_star)
    return u_star, result, spatial.distance.euclidean(X, result)


def LOS_angle(length, sx, sy, sth, u, coeffs, x, y, lookAhead=0.1):
    u_star, closest, distance = closestPointOnSpline2D(length, sx, sy, x, y, u, coeffs, guess=None)
    print "closest X = {:.3f}, {:.3f}, u* = {:.2f}".format(closest[0], closest[1], u_star)
    tangent_th = np.interp(u_star, u, sth)
    print "tangent_th = {:.2f}".format(np.rad2deg(tangent_th))
    wrapped_lookahead = np.mod(u_star + lookAhead, u[-1])  # this will make u wrap around a closed spline
    lookaheadState = splineToEuclidean2D(coeffs, wrapped_lookahead)
    dx_global = lookaheadState[0] - closest[0]
    dy_global = lookaheadState[1] - closest[1]
    dx_frenet = dx_global*math.cos(tangent_th) + dy_global*math.sin(tangent_th)
    dy_frenet = dx_global*math.sin(tangent_th) - dy_global*math.cos(tangent_th)
    # need sign of distance to spline to change
    # look at sign of cross product to determine "handed-ness"
    angle_from_closest_to_test = math.atan2(closest[1] - y, closest[0] - x)
    sign_test = np.cross([math.cos(tangent_th), math.sin(tangent_th)], [math.cos(angle_from_closest_to_test), math.sin(angle_from_closest_to_test)])
    distance *= np.sign(sign_test)
    # resulting triangle
    relative_angle = math.atan2((distance - dy_frenet), dx_frenet)
    global_angle = tangent_th + relative_angle
    return u_star, global_angle


def single_spline_main():
    x = [0, -5.]
    y = [0., 10.]
    th = [0., np.pi]
    N = 500
    sx, sy, sth, length, u, coeffs = PiazziSpline.piazziSpline(x[0], y[0], th[0], x[1], y[1], th[1], N=N)
    plt.plot(sx, sy, 'k-', linewidth=2.0)
    test_x = 1.0
    test_y = 7.0

    u_star, closest, distance = closestPointOnSpline2D(length, sx, sy, test_x, test_y, u, coeffs, guess=None)
    print "closest X = {:.3f}, {:.3f}".format(closest[0], closest[1])

    # Figure out the LOS controller using this example
    tangent_th = np.interp(u_star, u, sth)
    print "tangent th = {:.3f} deg".format(tangent_th*180.0/np.pi)
    lookAhead = 0.1  # how far forward in u
    lookaheadState = splineToEuclidean2D(coeffs, max(0.0, min(u_star + lookAhead, 1.0)))
    print "lookahead X = {:.3f}, {:.3f}".format(lookaheadState[0], lookaheadState[1])
    dx_global = lookaheadState[0] - closest[0]
    dy_global = lookaheadState[1] - closest[1]
    print "dx_global = {:.3f}, dy_global = {:.3f}".format(dx_global, dy_global)
    dx_frenet = dx_global*math.cos(tangent_th) + dy_global*math.sin(tangent_th)
    dy_frenet = dx_global*math.sin(tangent_th) - dy_global*math.cos(tangent_th)
    print "y error = {:.3f}, dx_frenet = {:.3f}, dy_frenet = {:.3f}".format(distance, dx_frenet, dy_frenet)

    # need sign of distance to spline to change
    # look at sign of cross product to determine "handed-ness"
    angle_from_closest_to_test = math.atan2(closest[1] - test_y, closest[0] - test_x)
    print "angle from closest to test = {:.3f} deg".format(angle_from_closest_to_test*180./np.pi)
    sign_test = np.cross([math.cos(tangent_th), math.sin(tangent_th)], [math.cos(angle_from_closest_to_test), math.sin(angle_from_closest_to_test)])
    distance *= np.sign(sign_test)
    print "distance to spline after sign update = {:.3f}".format(distance)

    # resulting triangle
    relative_angle = math.atan2((distance - dy_frenet), dx_frenet)
    u_star, global_angle = LOS_angle(length, sx, sy, sth, u, coeffs, test_x, test_y)
    print "relative angle = {:.3f} deg, global angle = {:.3f} deg".format(relative_angle*180./np.pi, global_angle*180./np.pi)

    # plot
    plt.plot(test_x, test_y, 'r+', markersize=12., markeredgewidth=2.0)
    plt.plot(closest[0], closest[1], 'gx', markersize=12., markeredgewidth=2.0)
    plt.plot(lookaheadState[0], lookaheadState[1], 'go', markersize=12., markeredgewidth=2.0)
    result_line = np.array([[test_x, test_y], closest])
    lookAhead_line = np.array([closest, lookaheadState])
    plt.plot(result_line[:, 0], result_line[:, 1], 'b-')
    plt.plot(lookAhead_line[:, 0], lookAhead_line[:, 1], 'g-')
    plt.arrow(closest[0], closest[1], 10.*math.cos(tangent_th), 10.*math.sin(tangent_th), linestyle='dashed')
    d = math.sqrt(math.pow(lookaheadState[0] - test_x, 2) + math.pow(lookaheadState[1] - test_y, 2))
    plt.arrow(test_x, test_y, d*math.cos(global_angle), d*math.sin(global_angle), linestyle='dotted')
    plt.axis('equal')
    plt.title("length = {:.2f}, u = {:.4f}, distance = {:.2f}".format(length[-1], u_star, distance))
    plt.show()


# TODO - perhaps move all the geometry stuff into a function, because you will use it throughout the LOS controller
def open_spline_chain_main():
    x = [0, 3, -20, -10]
    y = [0, 30, -5, -3]
    X = np.column_stack((x, y))
    th = [np.pi/2., np.pi, -np.pi/2., 0.]
    # Random desired discretization along the splines for demonstration purposes
    Ns = np.random.random_integers(100., 200., (X.shape[0]-1,))  # one less spline than waypoints.
    sx, sy, sth, length, u, coeffs = PiazziSpline.splineOpenChain(X, th, Ns)

    plt.plot(sx, sy, 'k-', linewidth=2.0)
    #test_x = 3.0
    #test_y = 30.01
    test_x = 0.013
    test_y = 0.008
    u_star, closest, distance = closestPointOnSpline2D(length, sx, sy, test_x, test_y, u, coeffs, guess=None)
    print "closest X = {:.3f}, {:.3f},  u* = {}".format(closest[0], closest[1], u_star)

    # Figure out the LOS controller using this example
    tangent_th = np.interp(u_star, u, sth)
    print "tangent th = {:.3f} deg".format(tangent_th*180.0/np.pi)
    lookAhead = 0.1  # how far forward in u
    lookaheadState = splineToEuclidean2D(coeffs, min(u_star + lookAhead, u[-1]))
    print "lookahead X = {:.3f}, {:.3f}".format(lookaheadState[0], lookaheadState[1])
    dx_global = lookaheadState[0] - closest[0]
    dy_global = lookaheadState[1] - closest[1]
    print "dx_global = {:.3f}, dy_global = {:.3f}".format(dx_global, dy_global)
    dx_frenet = dx_global*math.cos(tangent_th) + dy_global*math.sin(tangent_th)
    dy_frenet = dx_global*math.sin(tangent_th) - dy_global*math.cos(tangent_th)
    print "y error = {:.3f}, dx_frenet = {:.3f}, dy_frenet = {:.3f}".format(distance, dx_frenet, dy_frenet)

    # need sign of distance to spline to change
    # look at sign of cross product to determine "handed-ness"
    angle_from_closest_to_test = math.atan2(closest[1] - test_y, closest[0] - test_x)
    print "angle from closest to test = {:.3f} deg".format(angle_from_closest_to_test*180./np.pi)
    sign_test = np.cross([math.cos(tangent_th), math.sin(tangent_th)], [math.cos(angle_from_closest_to_test), math.sin(angle_from_closest_to_test)])
    distance *= np.sign(sign_test)
    print "distance to spline after sign update = {:.3f}".format(distance)

    # resulting triangle
    relative_angle = math.atan2((distance - dy_frenet), dx_frenet)
    global_angle = tangent_th + relative_angle
    print "relative angle = {:.3f} deg, global angle = {:.3f} deg".format(relative_angle*180./np.pi, global_angle*180./np.pi)

    # plot
    plt.plot(test_x, test_y, 'r+', markersize=12., markeredgewidth=2.0)
    plt.plot(closest[0], closest[1], 'gx', markersize=12., markeredgewidth=2.0)
    plt.plot(lookaheadState[0], lookaheadState[1], 'go', markersize=12., markeredgewidth=2.0)
    result_line = np.array([[test_x, test_y], closest])
    lookAhead_line = np.array([closest, lookaheadState])
    plt.plot(result_line[:, 0], result_line[:, 1], 'b-')
    plt.plot(lookAhead_line[:, 0], lookAhead_line[:, 1], 'g-')
    plt.arrow(closest[0], closest[1], 10.*math.cos(tangent_th), 10.*math.sin(tangent_th), linestyle='dashed')
    d = math.sqrt(math.pow(lookaheadState[0] - test_x, 2) + math.pow(lookaheadState[1] - test_y, 2))
    plt.arrow(test_x, test_y, d*math.cos(global_angle), d*math.sin(global_angle), linestyle='dotted')
    plt.axis('equal')
    plt.title("length = {:.2f}, u = {:.4f}, distance = {:.2f}".format(length[-1], u_star, distance))
    plt.show()


def closed_spline_chain_main():
    # a chain test
    x = [10, 0, -10, 0]
    y = [0, 10, 0, -10]
    X = np.column_stack((x, y))
    th = [np.pi/2.0, np.pi, -np.pi/2.0, 0]
    Ns = np.random.random_integers(100., 200., (X.shape[0],))
    sx, sy, sth, length, u, coeffs = PiazziSpline.splineClosedChain(X, th, Ns)

    plt.plot(sx, sy, 'k-', linewidth=2.0)
    test_x = 11.0
    test_y = -0.5
    u_star, closest, distance = closestPointOnSpline2D(length, sx, sy, test_x, test_y, u, coeffs, guess=None)
    print "closest X = {:.3f}, {:.3f},  u* = {}".format(closest[0], closest[1], u_star)

    # Figure out the LOS controller using this example
    tangent_th = np.interp(u_star, u, sth)
    print "tangent th = {:.3f} deg".format(tangent_th*180.0/np.pi)
    lookAhead = 0.2  # how far forward in u (remember u goes up to spline_count in a spline chain, not just up to 1)
    wrapped_lookahead = np.mod(u_star + lookAhead, u[-1])  # this will make u wrap around the closed spline
    lookaheadState = splineToEuclidean2D(coeffs, wrapped_lookahead)
    print "lookahead X = {:.3f}, {:.3f}".format(lookaheadState[0], lookaheadState[1])
    dx_global = lookaheadState[0] - closest[0]
    dy_global = lookaheadState[1] - closest[1]
    print "dx_global = {:.3f}, dy_global = {:.3f}".format(dx_global, dy_global)
    dx_frenet = dx_global*math.cos(tangent_th) + dy_global*math.sin(tangent_th)
    dy_frenet = dx_global*math.sin(tangent_th) - dy_global*math.cos(tangent_th)
    print "y error = {:.3f}, dx_frenet = {:.3f}, dy_frenet = {:.3f}".format(distance, dx_frenet, dy_frenet)

    # need sign of distance to spline to change
    # look at sign of cross product to determine "handed-ness"
    angle_from_closest_to_test = math.atan2(closest[1] - test_y, closest[0] - test_x)
    print "angle from closest to test = {:.3f} deg".format(angle_from_closest_to_test*180./np.pi)
    sign_test = np.cross([math.cos(tangent_th), math.sin(tangent_th)], [math.cos(angle_from_closest_to_test), math.sin(angle_from_closest_to_test)])
    distance *= np.sign(sign_test)
    print "distance to spline after sign update = {:.3f}".format(distance)

    # resulting triangle
    relative_angle = math.atan2((distance - dy_frenet), dx_frenet)
    global_angle = tangent_th + relative_angle
    print "relative angle = {:.3f} deg, global angle = {:.3f} deg".format(relative_angle*180./np.pi, global_angle*180./np.pi)

    # plot
    plt.plot(test_x, test_y, 'r+', markersize=12., markeredgewidth=2.0)
    plt.plot(closest[0], closest[1], 'gx', markersize=12., markeredgewidth=2.0)
    plt.plot(lookaheadState[0], lookaheadState[1], 'go', markersize=12., markeredgewidth=2.0)
    result_line = np.array([[test_x, test_y], closest])
    lookAhead_line = np.array([closest, lookaheadState])
    plt.plot(result_line[:, 0], result_line[:, 1], 'b-')
    plt.plot(lookAhead_line[:, 0], lookAhead_line[:, 1], 'g-')
    plt.arrow(closest[0], closest[1], 10.*math.cos(tangent_th), 10.*math.sin(tangent_th), linestyle='dashed')
    d = math.sqrt(math.pow(lookaheadState[0] - test_x, 2) + math.pow(lookaheadState[1] - test_y, 2))
    plt.arrow(test_x, test_y, d*math.cos(global_angle), d*math.sin(global_angle), linestyle='dotted')
    plt.axis('equal')
    plt.title("length = {:.2f}, u = {:.4f}, distance = {:.2f}".format(length[-1], u_star, distance))
    plt.show()


if __name__ == "__main__":
    # single_spline_main()
    open_spline_chain_main()
    # closed_spline_chain_main()