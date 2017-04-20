__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from utils import*

def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt

#fitting sur l'autocorrélation
def fit(mat,length):

    from scipy.optimize import curve_fit
    corr = autocorr(mat,10,length)
    y = np.linspace(10,length,length-10)

    popt, pcov = curve_fit(func, y, corr)
    return y, corr, popt, pcov

#fonction mathématique sur laquelle on fait le fit
def func(x,a,b,c):
        return a*np.exp(-b*x) + c


if __name__ == "__main__":
#initialisation de l'attracteur
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    y0 = [1, 1, 1]
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    mat = sol
#fit et autocorrélation des solutions
    longueur = 50
    y1, corr1, popt1, pcov1 = fit(sol[:,0],longueur)
    y2, corr2, popt2, pcov2 = fit(sol[:,1],longueur)
    y3, corr3, popt3, pcov3 = fit(sol[:,2],longueur -20)

    plt.plot(corr2)

    print(popt1[0],' * ',"e^(-",popt1[1],") + ", popt1[2], " avec ",1/popt1[1])
    print(popt2[0],' * ',"e^(-",popt2[1],") + ", popt2[2], " avec ",1/popt2[1])
    print(popt3[0],' * ',"e^(-",popt3[1],") + ", popt3[2], " avec ",1/popt3[1])

    coeff = (1/popt1[1] + 1/popt2[1] + 1/popt3[1])/3
#affichage des graphiques d'intéret
    print("Le coeff est alors: ",coeff)
    plt.show()
    plt.plot(corr1)
    plt.plot(t[10:(longueur-10)*100], func(t,popt1[0]/2, popt1[1], popt1[2])[10:(longueur-10)*100])
    plt.show()

    plt.plot(corr2)
    plt.plot(t[10:(longueur-10)*100], func(t,popt2[0]/2, popt2[1], popt2[2])[10:(longueur-10)*100])
    plt.show()

    plt.plot(corr3)
    plt.plot(t[10:(longueur-10)*100], func(t,popt3[0]/2, popt3[1], popt3[2])[10:(longueur-10)*100])
    plt.show()
    print("done")