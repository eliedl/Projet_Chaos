import sys
from PyQt4 import QtGui, QtCore
import matplotlib as mpl
import numpy as np
import scipy as sc
from unicodedata import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits.mplot3d import axes3d

class Projet_UI(QtGui.QWidget):
    '''
    User interface class
    '''
    def __init__(self, parent=None):
        super(Projet_UI, self).__init__()
        self.setWindowTitle("Dynamica 2017")
        self.init_UI()


    def init_UI(self):
        char1 = lookup("GREEK SMALL LETTER SIGMA")
        char2 = lookup("GREEK SMALL LETTER RHO")
        char3 = lookup("GREEK SMALL LETTER BETA")

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

        #--- Edits ---#
        self.x_edit = QtGui.QLineEdit()
        self.y_edit = QtGui.QLineEdit()
        self.z_edit = QtGui.QLineEdit()
        self.x_i_edit= QtGui.QLineEdit()
        self.y_i_edit= QtGui.QLineEdit()
        self.z_i_edit= QtGui.QLineEdit()
        self.sigma_edit= QtGui.QLineEdit('10')
        self.rho_edit = QtGui.QLineEdit('28')
        self.beta_edit= QtGui.QLineEdit('8/3')

        self.x_edit.setFixedWidth(80)
        self.y_edit.setFixedWidth(80)
        self.z_edit.setFixedWidth(80)
        self.x_i_edit.setFixedWidth(80)
        self.y_i_edit.setFixedWidth(80)
        self.z_i_edit.setFixedWidth(80)
        self.sigma_edit.setFixedWidth(80)
        self.rho_edit.setFixedWidth(80)
        self.beta_edit.setFixedWidth(80)

        #--- ComboBox ---#
        self.model_combo = QtGui.QComboBox()

        models = ['Lorentz', '-']

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
        setup_grid.addWidget(parametres_label, 6, 1)
        setup_grid.addWidget(sigma_label, 7, 0)
        setup_grid.addWidget(rho_label, 7, 1)
        setup_grid.addWidget(beta_label, 7, 2)
        setup_grid.addWidget(self.sigma_edit, 8, 0)
        setup_grid.addWidget(self.rho_edit, 8, 1)
        setup_grid.addWidget(self.beta_edit, 8, 2)
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
        self.ax3 = self.figure.add_axes([0.05, 0.05, 0.4, 0.3])
        self.ax4 = self.figure.add_axes([0.55, 0.05, 0.4, 0.3])

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
    proj_ui.show()

    sys.exit(app.exec_())