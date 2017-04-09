__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt


def autocorr( mat, min_step, max_step):
    corr = []
    count = 0
    for step in range(min_step,max_step):
        somme = 0
        for i in range(0,mat.size - step):
                somme += mat[i]*mat[i+step]
        corr.append(somme/(100))
        count += 1
    return corr




if __name__ == "__main__":
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    y0 = [5, 5, 10]

    sol = odeint(pend, y0, t, args=(sigma, rho, beta))

    def func(x,a,b,c):
        return a*np.exp(-b*x)
    from scipy.optimize import curve_fit
    corr = autocorr(sol[:,1],10,50)
    y = np.linspace(10,50,40)

    popt, pcov = curve_fit(func, y, corr)
    plt.plot(y,corr)
    plt.plot(y, func(y, *popt))
    print(popt[0],' * ',"e^(-",popt[1],") + ", popt[2])
    print("Le coeff est alors: ",1/popt[1])
    plt.show()
    print("done")