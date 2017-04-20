__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

#Scripte contenant les fonction utilitaires des programmes de test.


#Vérifie si un point se trouve dans un espace représentant les parois d'une boîte
#Retourne un booléen, vrai si le point s'y trouve faux sinon
def isinbox(position_point, centre_boite, arrete_boite, epaisseur):
    x = position_point[0]
    y = position_point[1]
    z = position_point[2]

    xb = centre_boite[0]
    yb = centre_boite[1]
    zb = centre_boite[2]

    valide = False
    if x > xb - arrete_boite and x < xb + arrete_boite:
        if y > yb - arrete_boite and y < yb + arrete_boite:
            if z > zb - arrete_boite and z < zb + arrete_boite:
                valide = True

    if x > xb - arrete_boite + epaisseur and x < xb + arrete_boite - epaisseur:
        if y > yb - arrete_boite + epaisseur and y < yb + arrete_boite - epaisseur:
            if z > zb - arrete_boite + epaisseur and z < zb + arrete_boite - epaisseur:
                valide = False

    return valide

#Vérifie si un point se trouve dans un espace représentant les parois d'une sphère
#Retourne un booléen, vrai si le point s'y trouve faux sinon
def isincircle(position_point, centre_cercle, rayon, epaisseur):
    x = position_point[0]
    y = position_point[1]
    z = position_point[2]

    xb = centre_cercle[0]
    yb = centre_cercle[1]
    zb = centre_cercle[2]
    valide = False

    if (np.abs(x-xb)**2 + np.abs(y-yb)**2 + np.abs(z-zb)**2) < rayon:
        valide = True

    if (np.abs(x-xb)**2 + np.abs(y-yb)**2 + np.abs(z-zb)**2) < rayon - epaisseur:
        valide = False

    return valide

#Isole les points d'un ensemble se trouvant  dans les parois d'une boîte de dimensions spécifiques
#Retourne une matrice contenant les points de l'ensemble initial se trouvant dans les parois de la boîte
def points_in_box(mat, centre, arrete, epaisseur):
        thresh_matrix = np.zeros((1, 3))
        for i in range(0, mat.shape[0]):
            point = [mat[i,0] , mat[i,1], mat[i,2]]
            if isinbox(point, centre, arrete, epaisseur):
                thresh_matrix = np.vstack((thresh_matrix, point))
        thresh_matrix = thresh_matrix[1:,:]

        return thresh_matrix

#Isole les points d'un ensemble se trouvant dans les parois d'une coquille sphérique de dimensions spécifiques
#Retourne une matrice contenant les points de l'ensemble initial se trouvant dans les parois de la coquille
def points_in_circle(mat, centre, rayon, epaisseur):
        thresh_matrix = np.zeros((1, 3))
        for i in range(0, mat.shape[0]):
            point = [mat[i,0] , mat[i,1], mat[i,2]]
            if isincircle(point, centre, rayon, epaisseur):
                thresh_matrix = np.vstack((thresh_matrix, point))
        thresh_matrix = thresh_matrix[1:,:]

        return thresh_matrix

#Calcule l'autocorrélation d'une trajectoire
#Retourne l'autocorrélation C(tau)
def autocorr( mat, min_step, max_step):
    corr = []
    count = 0
    for step in range(min_step,max_step):
        somme = 0
        for i in range(0,mat.size - step):
                somme +=mat[i]*mat[i+step]
        corr.append(somme/(100))
        count += 1
    return corr

#Calcule la similarité cosinus entre deux trajectoires
#Retourne une matrice contenant la similarité cosinus en fonction du temps
def cosine_similarity(mat1, mat2):
    cssim = []
    for i in range(0,mat1.size//3 -1):
        x1 = mat1[i,0]
        y1 = mat1[i,1]
        z1 = mat1[i,2]

        x2 = mat2[i,0]
        y2 = mat2[i,1]
        z2 = mat2[i,2]

        x1p = mat1[i+1,0]
        y1p = mat1[i+1,1]
        z1p = mat1[i+1,2]

        x2p = mat2[i+1,0]
        y2p = mat2[i+1,1]
        z2p = mat2[i+1,2]

        dx1 =  x1p - x1
        dy1 =  y1p - y1
        dz1 =  z1p - z1

        dx2 =  x2p - x2
        dy2 =  y2p - y2
        dz2 =  z2p - z2

        coeff = (dx1*dx2 + dy1*dy2 + dz1*dz2)/ (np.sqrt((dx1**2 + dy1**2 +dz1**2 )*(dx2**2 + dy2**2 +dz2**2 )))
        cssim.append(coeff)
    return cssim


