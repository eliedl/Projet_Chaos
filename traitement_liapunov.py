import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a, b):
        return a*x + b


liapunov_data = np.loadtxt('liapunov_r_Rössler.txt')

xdata = np.arange(0, 31, 1)
ydata = liapunov_data[:31]
print(xdata)
print(ydata)



limit1 = 3
limit2 = 4
limit3 = 5

popt1, pcov1 = curve_fit(func, xdata[:limit1], ydata[:limit1])
popt2, pcov2 = curve_fit(func, xdata[:limit2], ydata[:limit2])
popt3, pcov3 = curve_fit(func, xdata[:limit3], ydata[:limit3])

print(popt1)
print(popt2)
print(popt3)
xdata_1 = np.arange(-0.5, 10, 1)
xdata_2 = np.arange(-1, 10, 1)
xdata_3 = np.arange(-2, 10, 1)

plt.plot(xdata_1, popt1[0]*xdata_1 + popt1[1], label=r'$\lambda_L$ = {}'.format(np.round(popt1[0], decimals=2)), ls = '--')
plt.plot(xdata_2, popt2[0]*xdata_2 + popt2[1], label=r'$\lambda_L$ = {}'.format(np.round(popt2[0], decimals=2)), ls = '--')
plt.plot(xdata_3, popt3[0]*xdata_3 + popt3[1], label=r'$\lambda_L$ = {}'.format(np.round(popt3[0], decimals=2)), ls = '--')
plt.plot(xdata, ydata, marker = 'o', markersize= 2, label= 'liapunov.txt')
plt.ylabel(r'<ln(dk)>')
plt.xlabel('k')
plt.ylim(-12.5, -4)
plt.legend()
plt.show()
