__author__ = 'Charles'
__author__ = 'Charles'
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


def generate_data(y0, t):
    sigma, rho, beta = 10, 28, 8/3
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))

    return sol

def cosine_similarity(mat1, mat2):
    cssim = []
    for i in range(0,mat1.size//3):
        x1 = mat1[i,0]
        y1 = mat1[i,1]
        z1 = mat1[i,2]

        x2 = mat2[i,0]
        y2 = mat2[i,1]
        z2 = mat2[i,2]

        coeff = (x1*x2 + y1*y2 + z1*z2)/ (np.sqrt((x1**2 + y1**2 +z1**2 )*(x2**2 + y2**2 +z2**2 )))
        cssim.append(coeff%1)
    return cssim

def calculate_threshhold(mat, temps):

    count = 0
    for i in mat:
        if i <= 0.5:
            return temps[count]
        count += 1
    return temps[count]

def run_one():
    t = np.linspace(1, 100, 10001)
    y0 = [-5 + 1e-2, 0.5+ 1e-2, 0.5+ 1e-2]
    y1 = [-5, 0.5, 0.5]
    sol, sol_1 = generate_data(y0,t)
    sim = np.degrees(np.arccos(cosine_similarity(sol,sol_1)))
    plt.plot(t, sim)
    #plt.title(y1)

    print("Temps de corrélation : ", calculate_threshhold(sim,t)," s")
    plt.show()
    print("done")


if __name__ == "__main__":

    sigma, rho, beta = 10, 28, 8/3

    lam_mat =lambda Y,Z:([[- sigma, (rho + sigma -Z)/2, Y/2]
                        , [(rho + sigma -Z)/2, -1, 0]
                        ,[ Y/2, 0, -beta]])
    lam_mat =lambda Y,Z:([[- sigma, (rho + sigma -Z)/2, 0]
                        , [(rho + sigma -Z)/2, -1, 0]
                        ,[ 0, 0, -beta]])
    t = np.linspace(1, 1000, 100001)
    y0 = [1.547679936204874984e+00, 2.123422213176072049e+00, 2.018664314318483122e+01]
    initial_values = generate_data(y0,t)
    thresh_matrix = np.zeros((1, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    resolution = 10000

    for i in range(0,resolution,1):
        pos = [initial_values[i,0], initial_values[i,1], initial_values[i,2]]
        Y = 0
        Z = pos[2]
        w = np.linalg.eigvalsh(np.array(lam_mat(Y,Z)))
        liapunov = max(w)
        lam3 = (-(sigma+1) + np.sqrt((rho + sigma - Z)**2 + (sigma+1)**2))/2
        liapunov = lam3
        local = np.array([pos[0], pos[1], pos[2], liapunov])
        thresh_matrix = np.vstack((thresh_matrix, local))
        print((i * 100)//resolution)


    xs = thresh_matrix[1:,0]
    ys = thresh_matrix[1:,1]
    zs = thresh_matrix[1:,2]
    c = thresh_matrix[1:,3]


    print(c)

    np.savetxt("last_liapunov", thresh_matrix)

    p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
    h = fig.colorbar(p)
    h.set_label(r'$\lambda_3(z)$')
    fig.canvas.draw()
    plt.show()

