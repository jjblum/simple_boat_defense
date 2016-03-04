import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time
import PiazziSpline


def splineToEuclidean2D(spline_coeffs, u):
    """
    :param spline_coeffs: spline_coeffs: ((a0-a7), (b0-b7))
    :param u: u = [0, 1] along the spline at which to evaluate the derivative
    :return: (x, y)
    """
    a = spline_coeffs[0]
    b = spline_coeffs[1]
    S = np.power(u, np.arange(0., 8.))
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
    :param spline_coeffs: ((a0-a7), (b0-b7))
    :param u: u = [0, 1] along the spline at which to evaluate the derivative
    :param X: the cartesian point we are evaluating distance to
    :return: (dD_ds, dD_ds2): first derivative of the sq. euclidean distance evaluated at s, 2nd deriv.
    """
    a = spline_coeffs[0]
    b = spline_coeffs[1]
    powers = np.arange(0., 8.)
    S = np.power(u, powers)
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


def closestPointOnSpline2D(length, sx, sy, x, y, u, spline_coeffs, guess=None):
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
    total_length = length[-1]
    u_quad = np.zeros((4,))
    D_quad = np.zeros((3,))
    P_quad = np.zeros((4,))
    X = np.atleast_2d(np.array([x, y]))

    # TODO - address the situation where guess is not None
    if guess is None:
        # need to do something slower b/c we have no idea where to start
        S = np.column_stack((sx, sy))
        closest = np.argmin(spatial.distance.cdist(S, X))
        # TODO - if this returns the very first or very last index, just return that directly
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
            D_quad[q] = spatial.distance.sqeuclidean(
                    X, np.atleast_2d(splineToEuclidean2D(spline_coeffs, u_quad[q])))
        u_quad[3] = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                        (u_ij[1, 2]*D_quad[0] + u_ij[2, 0]*D_quad[1] + u_ij[0, 1]*D_quad[2])
        for p in range(4):
            P_quad[p] = polyInterpOfSplineToEuclidean2D(u_quad, D_quad, u_quad[p])
        # first, sort based on P value so u_quad[1:3] has the 3 lowest P values
        # then, resort u_quad[0:3] so that the s values are in increasing order
        u_quad[0:3] = np.sort(u_quad[np.argsort(P_quad)[0:3]])
    u_star = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                 (u_ij[1, 2]*D_quad[0] + u_ij[2, 0]*D_quad[1] + u_ij[0, 1]*D_quad[2])

    if np.isnan(u_star):
        u_star = length[closest]

    # ## NEWTONS PHASE - until convergence
    # TODO - do we even need this phase? We don't need ridiculous precision
    convergence = np.inf
    newtonCount = 0
    while convergence > 1.0e-8 and newtonCount < 100:
        newtonCount += 1
        D, dD_du_, dD_du2_ = dD_du(spline_coeffs, u_star, x, y)
        u_star_old = u_star
        u_star = u_star - dD_du_/dD_du2_
        if u_star < 0:
            raise ValueError("u_star went negative")
            return
        convergence = np.abs(u_star - u_star_old)

    result = splineToEuclidean2D(spline_coeffs, u_star)

    return u_star, result, spatial.distance.euclidean(X, result)


if __name__ == "__main__":
    x = [0, -5.]
    y = [0., 10.]
    th = [0., 0.]
    N = 100
    sx, sy, sth, length, u, coeffs = PiazziSpline.piazziSpline(x[0], y[0], th[0], x[1], y[1], th[1], N=N)
    plt.plot(sx, sy, 'k-', linewidth=2.0)
    test_x = -5.5
    test_y = 9.5
    u_star, closest, distance = closestPointOnSpline2D(length, sx, sy, test_x, test_y, u, coeffs, guess=None)
    print closest
    plt.plot(test_x, test_y, 'r+', markersize=12., markeredgewidth=2.0)
    plt.plot(closest[0], closest[1], 'gx', markersize=12., markeredgewidth=2.0)
    result_line = np.array([[test_x, test_y], closest])
    plt.plot(result_line[:, 0], result_line[:, 1], 'b-')
    plt.axis('equal')
    plt.title("length = {:.2f}, u = {:.4f}, distance = {:.2f}".format(length[-1], u_star, distance))
    plt.show()

