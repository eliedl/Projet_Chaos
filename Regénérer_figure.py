__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
#Ce scripte régènère les figures de recouvrement à partir des fichiers texte comporenant les données calculées.

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

thresh_matrix = np.loadtxt("-8_cossim_cover.txt")

xs = thresh_matrix[50:,0]
ys = thresh_matrix[50:,1]
zs = thresh_matrix[50:,2]
c = thresh_matrix[50:,3]
count = 0

#print(c)


p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
fig.colorbar(p)
plt.show()

calculer_max_chaque = True

if calculer_max_chaque:
    thresh_matrix = np.loadtxt("-1_cossim_cover.txt")
    max_1 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-2_cossim_cover.txt")
    max_2 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-3_cossim_cover.txt")
    max_3 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-4_cossim_cover.txt")
    max_4 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-6_cossim_cover.txt")
    max_6 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-7_cossim_cover.txt")
    max_7 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-8_cossim_cover.txt")
    max_8 = max(thresh_matrix[:,3])
    thresh_matrix = np.loadtxt("-9_cossim_cover.txt")
    max_9 = max(thresh_matrix[:,3])

    lesmax = [max_1,max_2,max_3,max_4,22.8, max_6, 27, max_8,max_9]
    print(lesmax)
    lesindices = [1,2,3,4,5, 6,7,8,9]

    plt.scatter(lesindices, lesmax)
    plt.show()