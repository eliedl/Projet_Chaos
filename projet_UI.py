import sys
from PyQt4 import QtGui, QtCore
import matplotlib as mpl
import numpy as np
import scipy as sc


class Projet_UI(QtGui.QWidget):
    '''
    User interface class
    '''
    def __init__(self, parent=None):
        super(Projet_UI, self).__init__()
        self.setWindowTitle("Dynamica 2017")
        self.init_UI()


    def init_UI(self):


        self.model_combo = QtGui.QComboBox()

        models = ['Lorenz', 'Hénon']

        self.model_combo.addItems(models)

        master_grid = QtGui.QGridLayout()
        master_grid.addWidget(self.model_combo)

        self.setLayout(master_grid)


if __name__ == '__main__':


    app = QtGui.QApplication(sys.argv)

    proj_ui = Projet_UI()
    proj_ui.show()

    sys.exit(app.exec_())