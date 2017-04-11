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
        for i in range(0,mat.size//3 - step):
                r_1 = np.sqrt(mat[i,0]**2 + mat[i,1]**2 + mat[i,2]**2)
                r_2 = np.sqrt(mat[i+step,0]**2 + mat[i+step,1]**2 + mat[i+step,2]**2)

                somme +=mat[i,0]*mat[i+step,0]
        corr.append(somme/(100))
        count += 1
    return corr

def fit(mat,length):

    from scipy.optimize import curve_fit
    corr = autocorr(mat,10,length)
    y = np.linspace(10,length,length-10)
    print(corr)

    popt, pcov = curve_fit(func, y, corr)
    return y, corr, popt, pcov

def func(x,a,b,c):
        return a*np.exp(-b*x) + c


if __name__ == "__main__":
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    y0 = [1, 1, 1]
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    #plt.plot(autocorr(sol,10,8000))
    mat = sol
    r_1 = np.sqrt(mat[:,0]**2 + mat[:,1]**2 + mat[:,2]**2)
    print(np.mean(r_1))
    plt.plot(r_1)
    plt.show()


    y, corr, popt, pcov = fit(sol,50)
    plt.plot(y,corr)
    plt.plot(y, func(y, *popt))
    print(popt[0],' * ',"e^(-",popt[1],") + ", popt[2])
    print("Le coeff est alors: ",1/popt[1])
    plt.show()
    print("done")