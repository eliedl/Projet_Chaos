import sys
from PyQt4 import QtGui, QtCore
import matplotlib as mpl
import numpy as np
import scipy as sc
from unicodedata import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits.mplot3d import axes3d
from projet_core import Core
from Test3d import animate3d
from Test_scrollgraph import scrollgraph

class Projet_UI(QtGui.QWidget):
    '''
    User interface class
    '''
    def __init__(self, parent=None):
        super(Projet_UI, self).__init__()
        self.setWindowTitle("Dynamica 2017")
        self.init_UI()
        self.core = Core()

    def update_core(self):

        x = float(self.x_edit.text())
        y = float(self.y_edit.text())
        z = float(self.z_edit.text())
        x_i = float(self.x_i_edit.text())
        y_i = float(self.y_i_edit.text())
        z_i = float(self.z_i_edit.text())

        sigma = float(self.sigma_edit.text())
        rho = float(self.rho_edit.text())
        beta = float(self.beta_edit.text())

        t0 = float(self.t0_edit.text())
        tf = float(self.tf_edit.text())
        step = float(self.step_edit.text())


        self.core.attractor = self.model_combo.currentText()
        self.core.coordinates = np.array([[x, y, z], [x_i, y_i, z_i]])
        self.core.t = np.linspace(t0, tf, step)
        print(self.core.t)
        self.core.params = np.array([[sigma, rho, beta]])

    def start_simulation(self):
        self.update_core()
        self.core.solve_edo()
        self.simulationsFig.plot_simulations()

    def init_UI(self):
        char1 = lookup("GREEK SMALL LETTER SIGMA")
        char2 = lookup("GREEK SMALL LETTER RHO")
        char3 = lookup("GREEK SMALL LETTER BETA")

        #--- Buttons ---#
        start_btn = QtGui.QPushButton('Start')
        start_btn.clicked.connect(self.start_simulation)

        #--- Labels ---#
        attracteur_label = MyQLabel('Attracteur', 'left')
        x_label = MyQLabel('x', 'center')
        y_label = MyQLabel('y', 'center')
        z_label = MyQLabel('z', 'center')
        x_i_label = MyQLabel("x'", 'center')
        y_i_label = MyQLabel("y'", 'center')
        z_i_label = MyQLabel("z'", 'center')
        sigma_label = MyQLabel(char1,'center')
        rho_label = MyQLabel(char2,'center')
        beta_label =MyQLabel(char3,'center')
        parametres_label = MyQLabel('Paramètres', 'center')
        t0_label = MyQLabel('t0', 'center')
        tf_label = MyQLabel('tf', 'center')
        step_label = MyQLabel('Steps', 'center' )

        #--- Edits ---#
        self.x_edit = QtGui.QLineEdit('1')
        self.y_edit = QtGui.QLineEdit('1')
        self.z_edit = QtGui.QLineEdit('1')
        self.t0_edit = QtGui.QLineEdit('1')
        self.tf_edit = QtGui.QLineEdit('100')
        self.step_edit = QtGui.QLineEdit('10001')
        self.x_i_edit= QtGui.QLineEdit('1.00001')
        self.y_i_edit= QtGui.QLineEdit('1.00001')
        self.z_i_edit= QtGui.QLineEdit('1.00001')
        self.sigma_edit= QtGui.QLineEdit('10')
        self.rho_edit = QtGui.QLineEdit('28')
        self.beta_edit= QtGui.QLineEdit('2.67')

        self.x_edit.setFixedWidth(80)
        self.y_edit.setFixedWidth(80)
        self.z_edit.setFixedWidth(80)
        self.t0_edit.setFixedWidth(80)
        self.tf_edit.setFixedWidth(80)
        self.step_edit.setFixedWidth(80)
        self.x_i_edit.setFixedWidth(80)
        self.y_i_edit.setFixedWidth(80)
        self.z_i_edit.setFixedWidth(80)
        self.sigma_edit.setFixedWidth(80)
        self.rho_edit.setFixedWidth(80)
        self.beta_edit.setFixedWidth(80)

        self.x_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.y_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.z_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.t0_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.tf_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.step_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.x_i_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.y_i_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.z_i_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.sigma_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.rho_edit.setAlignment(QtCore.Qt.AlignHCenter)
        self.beta_edit.setAlignment(QtCore.Qt.AlignHCenter)

        #--- ComboBox ---#
        self.model_combo = QtGui.QComboBox()

        models = ['Lorentz', 'Roessler' ,'-']

        self.model_combo.addItems(models)

        # ------ Creation of the Manager for the Spectra figure -------#
        self.simulationsFig = SimulationsFig(self)
        self.simulationstool = NavigationToolbar2QT(self.simulationsFig, self)
        self.simulationsmanager = QtGui.QWidget()
        simulationsmanagergrid = QtGui.QGridLayout()
        simulationsmanagergrid.addWidget(self.simulationstool, 0, 0, 1, 6)
        simulationsmanagergrid.addWidget(self.simulationsFig, 1, 0, 1, 6)
        simulationsmanagergrid.setColumnStretch(1, 100)
        self.simulationsmanager.setLayout(simulationsmanagergrid)

        #--- Setup GroupBox ---#
        setup_groupbox = QtGui.QGroupBox('Setup')
        setup_grid = QtGui.QGridLayout()
        setup_grid.addWidget(attracteur_label, 0, 0, 1, 2)
        setup_grid.addWidget(self.model_combo, 1, 0, 1, 3)
        setup_grid.addWidget(x_label, 2, 0)
        setup_grid.addWidget(y_label, 2, 1)
        setup_grid.addWidget(z_label, 2, 2)
        setup_grid.addWidget(self.x_edit, 3, 0)
        setup_grid.addWidget(self.y_edit, 3, 1)
        setup_grid.addWidget(self.z_edit, 3, 2)
        setup_grid.addWidget(x_i_label, 4, 0)
        setup_grid.addWidget(y_i_label, 4, 1)
        setup_grid.addWidget(z_i_label, 4, 2)
        setup_grid.addWidget(self.x_i_edit, 5, 0)
        setup_grid.addWidget(self.y_i_edit, 5, 1)
        setup_grid.addWidget(self.z_i_edit, 5, 2)
        setup_grid.addWidget(t0_label, 6, 0)
        setup_grid.addWidget(tf_label, 6, 1)
        setup_grid.addWidget(step_label, 6, 2)
        setup_grid.addWidget(self.t0_edit, 7, 0)
        setup_grid.addWidget(self.tf_edit, 7, 1)
        setup_grid.addWidget(self.step_edit, 7, 2)
        setup_grid.addWidget(parametres_label, 8, 1)
        setup_grid.addWidget(sigma_label, 9, 0)
        setup_grid.addWidget(rho_label, 9, 1)
        setup_grid.addWidget(beta_label, 9, 2)
        setup_grid.addWidget(self.sigma_edit, 10, 0)
        setup_grid.addWidget(self.rho_edit, 10, 1)
        setup_grid.addWidget(self.beta_edit, 10, 2)
        setup_grid.addWidget(start_btn, 11, 0, 1, 3)
        setup_groupbox.setLayout(setup_grid)


        master_grid = QtGui.QGridLayout()
        master_grid.addWidget(setup_groupbox, 0, 0)
        master_grid.addWidget(self.simulationsmanager, 0, 1, 2, 1)
        master_grid.setColumnStretch(1, 100)
        master_grid.setRowStretch(1, 100)
        self.setLayout(master_grid)

class SimulationsFig(FigureCanvasQTAgg):
    def __init__(self, ui):
        fig = mpl.figure.Figure(facecolor='white')
        super(SimulationsFig, self).__init__(fig)
        self.ui = ui
        self.initFig()

    def initFig(self):
        self.ax1 = self.figure.add_axes([0, 0.38, 0.5, 0.65], projection='3d')
        self.ax2 = self.figure.add_axes([0.5, 0.38, 0.5, 0.65], projection = '3d')
        self.ax3 = self.figure.add_axes([0.08, 0.08, 0.4, 0.3])
        self.ax4 = self.figure.add_axes([0.58, 0.08, 0.4, 0.3])

        self.ax3.set_xlabel('Temps')
        self.ax3.set_ylabel('Correlation')

        self.ax4.set_ylabel("$|\ x - x'\ |$")
        self.ax4.set_xlabel('Temps')

        self.ax1.view_init(elev= 15)
        self.ax2.view_init(elev= 15)

    def plot_simulations(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()

        x = self.ui.core.time_series[:, 0]
        y = self.ui.core.time_series[:, 1]
        z = self.ui.core.time_series[:, 2]

        ani_1 = animate3d(self.figure, self.ax1, self.ui.core.time_series[:,:3], self.ui.core.time_series[:,3:])
        ani_3 = scrollgraph(self.figure, self.ax3, self.ui.core.time_series[:,:3], self.ui.core.time_series[:,3:])
        self.draw()
        #print(self.ui.core.time_series)
class  MyQLabel(QtGui.QLabel):
    #--- Class For Alignment ---#
    def __init__(self, label, ha='left',  parent=None):
        super(MyQLabel, self).__init__(label,parent)
        if ha == 'center':
            self.setAlignment(QtCore.Qt.AlignCenter)
        elif ha == 'right':
            self.setAlignment(QtCore.Qt.AlignRight)
        else:
            self.setAlignment(QtCore.Qt.AlignLeft)

if __name__ == '__main__':


    app = QtGui.QApplication(sys.argv)

    proj_ui = Projet_UI()
    proj_ui.showMaximized()

    sys.exit(app.exec_())