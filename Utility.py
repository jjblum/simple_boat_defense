import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time
import PiazziSpline


def splineToEuclidean2D(length, sx, sy, s):
    """
    :param length: array = [0, spline total length]
    :param sx: spline x's
    :param sy: spline y's
    :param s: length to convert to Euclidean point
    :return: numpy array [x, y], with shape (2,)
    """
    return np.array([np.interp(s, length, sx), np.interp(s, length, sy)])


def polyInterpOfSplineToEuclidean2D(s_quad, D_quad, s):
    """
    Interpolation of what euclidean distance should be if we fit a polynomial to the distances of the s_quad values
    :param s_quad: s1, s2, s3: the three length values used
    :param D_quad: D(s1), D(s2), D(s3): the three distances associated with the s_quad points
    :param s: the length value we are calculating euclidean distance estimation for
    :return: P(s) - the estimated euclidean distance at s
    """
    P = 0.0
    P += (s - s_quad[1])*(s - s_quad[2])/((s_quad[0] - s_quad[1])*(s_quad[0] - s_quad[2]))*D_quad[0]
    P += (s - s_quad[0])*(s - s_quad[2])/((s_quad[1] - s_quad[0])*(s_quad[1] - s_quad[2]))*D_quad[1]
    P += (s - s_quad[0])*(s - s_quad[1])/((s_quad[2] - s_quad[0])*(s_quad[2] - s_quad[1]))*D_quad[2]
    return P





def closestPointOnSpline2D(length, sx, sy, x, y, guess=None):
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
    s_star = 0.0
    N = len(sx)
    min_guess_halfwidth = np.floor(N/20.0)
    total_length = length[-1]
    s_quad = np.zeros((4,))
    D_quad = np.zeros((3,))
    P_quad = np.zeros((4,))
    X = np.atleast_2d(np.array([x, y]))

    # TODO - address the situation where guess is not None
    if guess is None:
        # need to do something slower b/c we have no idea where to start
        t = time.time()
        S = np.zeros((N, 2))
        for j in range(N):
            S[j, 0] = sx[j]
            S[j, 1] = sy[j]
        closest = np.argmin(spatial.distance.cdist(S, X))
        # TODO - if this returns the very first or very last index, just return that directly
        if closest == 0:
            S = np.array([sx[0], sy[0]])
            return 0.0, S, spatial.distance.euclidean(S, X)
        elif closest == len(sx) - 1:
            S = np.array([sx[-1], sy[-1]])
            return total_length, S, spatial.distance.euclidean(S, X)
        elif min_guess_halfwidth <= closest <= N - min_guess_halfwidth - 1:
            s_quad[0] = length[closest - min_guess_halfwidth]
            s_quad[1] = length[closest]
            s_quad[2] = length[closest + min_guess_halfwidth]
        elif closest < min_guess_halfwidth:
            s_quad[0] = length[0]
            s_quad[1] = length[min_guess_halfwidth]
            s_quad[2] = length[2*min_guess_halfwidth]
        elif closest > N - min_guess_halfwidth:
            s_quad[0] = length[N - 2*min_guess_halfwidth - 1]
            s_quad[1] = length[N - min_guess_halfwidth - 1]
            s_quad[2] = length[N - 1]

    # ## QUADRATIC PHASE - 4 iterations
    s_ij = np.zeros((3, 3))
    y_ij = np.zeros((3, 3))
    for iteration in range(4):
        for i in range(3):
            for j in range(3):
                s_ij[i, j] = s_quad[i] - s_quad[j]
                y_ij[i, j] = np.power(s_quad[i], 2) - np.power(s_quad[j], 2)
        for q in range(3):
            D_quad[q] = spatial.distance.sqeuclidean(X, np.atleast_2d(splineToEuclidean2D(length, sx, sy, s_quad[q])))

        s_quad[3] = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                        (s_ij[1, 2]*D_quad[0] + s_ij[2, 0]*D_quad[1] + s_ij[0, 1]*D_quad[2])
        for p in range(4):
            P_quad[p] = polyInterpOfSplineToEuclidean2D(s_quad, D_quad, s_quad[p])
        # first, sort based on P value so s_quad[1:3] has the 3 lowest P values
        # then, resort s_quad[0:3] so that the s values are in increasing order
        s_quad[0:3] = np.sort(s_quad[np.argsort(P_quad)[0:3]])
    s_star = 0.5*(y_ij[1, 2]*D_quad[0] + y_ij[2, 0]*D_quad[1] + y_ij[0, 1]*D_quad[2]) / \
                 (s_ij[1, 2]*D_quad[0] + s_ij[2, 0]*D_quad[1] + s_ij[0, 1]*D_quad[2])

    if np.isnan(s_star):
        s_star = length[closest]

    # ## NEWTONS PHASE - until convergence
    # TODO - do we even need this phase? We don't need ridiculous precision
    def sqEuclideanFunc(s):
        x_ = np.interp(s, length, sx)
        y_ = np.interp(s, length, sy)
        return np.power(x_ - x, 2) + np.power(y_ - y, 2)

    try:
        s_star = optimize.newton(sqEuclideanFunc, s_star)
    except:
        None

    result = splineToEuclidean2D(length, sx, sy, s_star)

    return s_star, result, spatial.distance.euclidean(X, result)


if __name__ == "__main__":
    x = [0, -5.]
    y = [0., 10.]
    th = [0., np.pi]
    N = 1000
    sx, sy, sth, total_length = PiazziSpline.piazziSpline(x[0], y[0], th[0], x[1], y[1], th[1], N=N)
    length = total_length*np.linspace(0., 1., N)
    plt.plot(sx, sy, 'k-', linewidth=2.0)
    test_x = 0.01
    test_y = 0.01
    s_star, closest, distance = closestPointOnSpline2D(length, sx, sy, test_x, test_y, guess=None)
    plt.plot(test_x, test_y, 'r+', markersize=12., markeredgewidth=2.0)
    plt.plot(closest[0], closest[1], 'gx', markersize=12., markeredgewidth=2.0)
    result_line = np.array([[test_x, test_y], closest])
    plt.plot(result_line[:, 0], result_line[:, 1], 'b-')
    plt.axis('equal')
    plt.title("length = {:.2f}, s = {:.4f}, distance = {:.2f}".format(total_length, s_star, distance))
    plt.show()
