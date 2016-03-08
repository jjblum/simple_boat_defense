import math as m
import numpy as np
import matplotlib.pyplot as plt


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# eta=[10., 10., 0., 0., 750., 750.]
def piazziSpline(x0, y0, th0, x1, y1, th1, k0=0, k0dot=0, k1=0, k1dot=0, N=100, eta=[10., 10., 0., 0., 700., 700.]):
    """
    :param x0: x coordinate, starting point
    :param y0: y coordinate, starting point
    :param th0: heading, starting point
    :param x1: x coordinate, ending point
    :param y1: y coordinate, ending point
    :param th1: heading, ending point
    :param k0: curvature, starting point
    :param k0dot: time rate of change of curvature, starting point
    :param k1: curvature, ending point
    :param k1dot: time rate of change of curvature, ending point
    :param N: number of points along the spline
    :param eta: 6 parameters that determine spline properties
    :return: (sx, sy) 2-tuple of numpy arrays for (x,y) coordinates of the spline
    """
    u = np.linspace(0.0, 1.0, N)

    # Set the coefficients for 7th degree polynomial in x
    a0 = x0
    a1 = eta[0]*m.cos(th0)
    a2 = 1./2.*eta[2]*m.cos(th0)-1./2.*m.pow(eta[0], 2)*k0*m.sin(th0)
    a3 = 1./6.*eta[4]*m.cos(th0)-1./6.*(m.pow(eta[0], 3)*k0dot+3*eta[0]*eta[2]*k0)*m.sin(th0)
    a4 = 35*(x1-x0)-(20*eta[0]+5*eta[2]+2./3.*eta[4])*m.cos(th0) \
        + (5*m.pow(eta[0], 2)*k0+2./3.*m.pow(eta[0], 3)*k0dot+2*eta[0]*eta[2]*k0)*m.sin(th0) \
        - (15*eta[1]-5./2.*eta[3]+1./6.*eta[5])*m.cos(th1) \
        - (5./2.*m.pow(eta[1], 2)*k1-1./6.*m.pow(eta[1], 3)*k1dot-1./2.*eta[1]*eta[3]*k1)*m.sin(th1)
    a5 = -84*(x1-x0)+(45*eta[0]+10*eta[2]+eta[4])*m.cos(th0) \
        - (10*m.pow(eta[0], 2)*k0+m.pow(eta[0], 3)*k0dot+3*eta[0]*eta[2]*k0)*m.sin(th0) \
        + (39*eta[1]-7*eta[3]+1./2.*eta[5])*m.cos(th1) \
        + (7*m.pow(eta[1], 2)*k1-1./2.*m.pow(eta[1], 3)*k1dot-3./2.*eta[1]*eta[3]*k1)*m.sin(th1)
    a6 = 70*(x1-x0)-(36*eta[0]+15./2.*eta[2]+2./3.*eta[4])*m.cos(th0) \
        + (15./2.*m.pow(eta[0], 2)*k0+2./3.*m.pow(eta[0], 3)*k0dot+2*eta[0]*eta[2]*k0)*m.sin(th0) \
        - (34*eta[1]-13./2.*eta[3]+1./2.*eta[5])*m.cos(th1) \
        - (13./2.*m.pow(eta[1], 2)*k1-1./2.*m.pow(eta[1], 3)*k1dot-3./2.*eta[1]*eta[3]*k1)*m.sin(th1)
    a7 = -20*(x1-x0)+(10*eta[0]+2*eta[2]+1./6.*eta[4])*m.cos(th0) \
        - (2*m.pow(eta[0], 2)*k0+1./6.*m.pow(eta[0], 3)*k0dot+1./2.*eta[0]*eta[2]*k0)*m.sin(th0) \
        + (10*eta[1]-2*eta[3]+1./6.*eta[5])*m.cos(th1) \
        + (2*m.pow(eta[1], 2)*k1-1./6.*m.pow(eta[1], 3)*k1dot-1./2.*eta[1]*eta[3]*k1)*m.sin(th1)

    # More Coefficients in y
    b0 = y0
    b1 = eta[0]*m.sin(th0)
    b2 = 1./2.*eta[2]*m.sin(th0)+1./2.*m.pow(eta[0], 2)*k0*m.cos(th0)
    b3 = 1./6.*eta[4]*m.sin(th0)+1./6.*(m.pow(eta[0], 3)*k0dot+3*eta[0]*eta[2]*k0)*m.cos(th0)
    b4 = 35*(y1-y0)-(20*eta[0]+5*eta[2]+2./3.*eta[4])*m.sin(th0) \
        - (5*m.pow(eta[0], 2)*k0+2./3.*m.pow(eta[0], 3)*k0dot+2*eta[0]*eta[2]*k0)*m.cos(th0) \
        - (15*eta[1]-5./2.*eta[3]+1./6.*eta[5])*m.sin(th1) \
        + (5./2.*m.pow(eta[1], 2)*k1-1./6.*m.pow(eta[1], 3)*k1dot-1./2.*eta[1]*eta[3]*k1)*m.cos(th1)
    b5 = -84*(y1-y0)+(45*eta[0]+10*eta[2]+eta[4])*m.sin(th0) \
        + (10*m.pow(eta[0], 2)*k0+m.pow(eta[0], 3)*k0dot+3*eta[0]*eta[2]*k0)*m.cos(th0) \
        + (39*eta[1]-7*eta[3]+1./2.*eta[5])*m.sin(th1) \
        - (7*m.pow(eta[1], 2)*k1-1./2.*m.pow(eta[1], 3)*k1dot-3./2.*eta[1]*eta[3]*k1)*m.cos(th1)
    b6 = 70*(y1-y0)-(36*eta[0]+15./2.*eta[2]+2./3.*eta[4])*m.sin(th0) \
        - (15./2.*m.pow(eta[0], 2)*k0+2./3.*m.pow(eta[0], 3)*k0dot+2*eta[0]*eta[2]*k0)*m.cos(th0) \
        - (34*eta[1]-13./2.*eta[3]+1./2.*eta[5])*m.sin(th1) \
        + (13./2.*m.pow(eta[1], 2)*k1-1./2.*m.pow(eta[1], 3)*k1dot-3./2.*eta[1]*eta[3]*k1)*m.cos(th1)
    b7 = -20*(y1-y0)+(10*eta[0]+2*eta[2]+1./6.*eta[4])*m.sin(th0) \
        + (2*m.pow(eta[0], 2)*k0+1./6.*m.pow(eta[0], 3)*k0dot+1./2.*eta[0]*eta[2]*k0)*m.cos(th0) \
        + (10*eta[1]-2*eta[3]+1./6.*eta[5])*m.sin(th1) \
        - (2*m.pow(eta[1], 2)*k1-1./6.*m.pow(eta[1], 3)*k1dot-1./2.*eta[1]*eta[3]*k1)*m.cos(th1)

    # The full polynomial
    sx = a0 + a1*u + a2*np.power(u, 2) + a3*np.power(u, 3) + a4*np.power(u, 4) + \
        a5*np.power(u, 5) + a6*np.power(u, 6) + a7 * np.power(u, 7)
    sy = b0 + b1*u + b2*np.power(u, 2) + b3*np.power(u, 3) + b4*np.power(u, 4) + \
        b5*np.power(u, 5) + b6*np.power(u, 6) + b7*np.power(u, 7)
    dx = np.r_[0.0, np.diff(sx)]
    dy = np.r_[0.0, np.diff(sy)]
    length = np.cumsum(np.sqrt(np.power(dx, 2) + np.power(dy, 2)))
    sth = np.arctan2(dy, dx)/m.pi  # multiples of pi (useful for singularity fixes)
    # need to find singularity jumps and patch them
    dth = np.r_[0.0, np.diff(sth)]
    dth[0] = dth[1];
    singularities = np.where(np.abs(np.abs(dth) - 0.5) < m.pow(10., -4))
    # erroneous sth jumps from pi/2 or -pi/2 to 0, so correct sth should be approx. equal to erroneous dth
    sth[singularities] = dth[singularities]
    return sx, sy, sth*m.pi, length, u, ((a0, a1, a2, a3, a4, a5, a6, a7), (b0, b1, b2, b3, b4, b5, b6, b7))


def splineOpenChain(waypoints, ths=None, Ns=None):
    """
    :param waypoints: numpy array with 2 columns [x, y] locations of waypoints
    :param ths: numpy array with 1 column [th] of desired orientations at the waypoints
                If not used, the tangent to the bisecting angle will be used (i.e. avg of sharp approach and exit)
    :param Ns: number of points in each part of the segment
               If not used, 100 will be used for all segments
    :return: a concatenation of the splines, with coeffs being a list of (a,b) coefficient tuples
    """
    wp_count = waypoints.shape[0]
    spline_count = wp_count - 1
    if ths is None:
        # TODO - calculate bisecting angle tangents (avg of sharp approach and exit angles)
        dX = np.diff(waypoints, axis=0)  # calculate sharp angles
    if Ns is None:
        Ns = 100.*np.ones((spline_count,))
    sx = np.zeros((sum(Ns),))
    sy = np.zeros((sum(Ns),))
    sth = np.zeros((sum(Ns),))
    length = np.zeros((sum(Ns),))
    u = np.linspace(0.0, 1.0, sum(Ns))
    coeffs = list()
    for j in range(spline_count):
        sx_, sy_, sth_, length_, u_, coeffs_ = piazziSpline(
                waypoints[j, 0], waypoints[j, 1], ths[j],
                waypoints[j+1, 0], waypoints[j+1, 1], ths[j+1], N=Ns[j])
        if j == 0:
            start_index = 0
        else:
            start_index = sum(Ns[:j])
        end_index = sum(Ns[:j+1])
        sx[start_index:end_index] = sx_
        sy[start_index:end_index] = sy_
        sth[start_index:end_index] = sth_
        length[start_index:end_index] = length_
        if j > 0:
            length[start_index:end_index] += length[start_index - 1]  # previous lengths add in
        coeffs.append(coeffs_)
    return sx, sy, sth, length, u, coeffs


# TODO - generate spline in a chain and make a strategy that uses np.mod(u, 1) to get a u that loops on itself
def splineClosedChain(waypoints, ths=None, Ns=None):
    waypoints = np.r_[waypoints, np.atleast_2d(waypoints[0, :])]
    ths = np.r_[ths, ths[0]]
    return splineOpenChain(waypoints, ths, Ns)


def main_single():
    # a simple test
    x = [0, 10.]
    y = [0., 10.]
    th = [0., 0.]
    N = 500
    sx, sy, sth, length, u, coeffs = piazziSpline(x[0], y[0], th[0], x[1], y[1], th[1], N=N)
    total_length = length[-1]
    plt.subplot(1, 2, 1)
    plt.plot(sx, sy)
    plt.axis('equal')
    plt.title("boat path, total length = {:.2f}".format(total_length))
    plt.subplot(1, 2, 2)
    plt.plot(u, sth*180.0/np.pi)
    plt.title("boat heading (deg)")
    plt.show()


def main_open_chain():
    # a chain test
    x = [10, 0, -10, 0]
    y = [0, 10, 0, -10]
    X = np.column_stack((x, y))
    th = [m.pi/2., m.pi, -m.pi/2., 0.]
    Ns = np.random.random_integers(100., 200., (X.shape[0]-1,))
    sx, sy, sth, length, u, coeffs = splineOpenChain(X, th, Ns)
    endpoint_indices = np.cumsum(Ns)-1

    plt.subplot(1, 2, 1)
    plt.plot(sx, sy)
    plt.plot(np.r_[x[0], sx[endpoint_indices]], np.r_[y[0], sy[endpoint_indices]],
             'r+', markersize=14, markeredgewidth=3)
    plt.axis('equal')
    plt.title("boat path, total length = {:.2f}".format(length[-1]))
    plt.subplot(1, 2, 2)
    plt.plot(u, sth*180.0/np.pi)

    plt.plot(np.r_[0, u[endpoint_indices]], 180.0/np.pi*np.r_[th[0], sth[endpoint_indices]],
             'r+', markersize=14, markeredgewidth=3)
    plt.title("boat heading (deg)")
    plt.show()


def main_closed_chain():
    # a chain test
    x = [10, 0, -10, 0]
    y = [0, 10, 0, -10]
    X = np.column_stack((x, y))
    th = [m.pi/2.0, m.pi, -m.pi/2.0, 0]
    Ns = np.random.random_integers(100., 200., (X.shape[0],))
    sx, sy, sth, length, u, coeffs = splineClosedChain(X, th, Ns)
    endpoint_indices = np.cumsum(Ns)-1

    plt.subplot(1, 2, 1)
    plt.plot(sx, sy)
    plt.plot(np.r_[x[0], sx[endpoint_indices]], np.r_[y[0], sy[endpoint_indices]],
             'r+', markersize=14, markeredgewidth=3)
    plt.axis('equal')
    plt.title("boat path, total length = {:.2f}".format(length[-1]))
    plt.subplot(1, 2, 2)
    plt.plot(u, sth*180.0/np.pi)

    plt.plot(np.r_[0, u[endpoint_indices]], 180.0/np.pi*np.r_[th[0], sth[endpoint_indices]],
             'r+', markersize=14, markeredgewidth=3)
    plt.title("boat heading (deg)")
    plt.show()


if __name__ == '__main__':
    main_single()
    main_open_chain()
    main_closed_chain()