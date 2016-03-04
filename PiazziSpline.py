import math as m
import numpy as np
import matplotlib.pyplot as plt


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def piazziSpline(x0, y0, th0, x1, y1, th1, k0=0, k0dot=0, k1=0, k1dot=0, N=100, eta=[5, 5, 0, 0, 750, 750]):
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
    sth = np.arctan2(dy, dx)/m.pi  # multiples of pi
    # need to find singularity jumps and patch them
    dth = np.r_[0.0, np.diff(sth)]
    dth[0] = dth[1];
    singularities = np.where(np.abs(np.abs(dth) - 0.5) < m.pow(10., -4))
    # erroneous sth jumps from pi/2 or -pi/2 to 0, so correct sth should be approx. equal to erroneous dth
    sth[singularities] = dth[singularities]
    return sx, sy, sth, length, u, ((a0, a1, a2, a3, a4, a5, a6, a7), (b0, b1, b2, b3, b4, b5, b6, b7))

if __name__ == '__main__':
    # a simple test
    x = [0, 5, 0, -5, 0, 5, 0]
    y = [0, 0, 5, 0, -5, 0, 0]
    th = [0, m.pi/2.0, m.pi, -m.pi/2.0, 0, m.pi/2.0, m.pi]
    spline_count = len(x)
    my_N = 100
    my_sx = np.empty((spline_count*my_N,))
    my_sy = np.empty((spline_count*my_N,))
    my_sth = np.empty((spline_count*my_N,))
    my_length = 0.0
    # my_dth = np.empty((spline_count*my_N,))
    l = np.linspace(0.0, 1.0, spline_count*my_N)
    for j in range(spline_count - 1):
        sx_, sy_, sth_, length_, u, coeffs = piazziSpline(x[j], y[j], th[j], x[j+1], y[j+1], th[j+1], N=my_N)
        my_sx[j*my_N:(j+1)*my_N] = sx_
        my_sy[j*my_N:(j+1)*my_N] = sy_
        my_sth[j*my_N:(j+1)*my_N] = sth_
        my_length += length_[-1]
        # my_dth[j*my_N:(j+1)*my_N] = dth_

    plt.subplot(1, 2, 1)
    plt.plot(my_sx, my_sy)
    plt.axis('equal')
    plt.title("boat path, total length = {}".format(my_length))
    plt.subplot(1, 2, 2)
    plt.plot(l, my_sth)
    plt.plot(np.arange(0., 1., 1./spline_count), my_sth[np.arange(0, spline_count*my_N, my_N)],
             'r+', markersize=14, markeredgewidth=3)
    plt.title("boat heading (multiples of pi)")
    plt.show()

