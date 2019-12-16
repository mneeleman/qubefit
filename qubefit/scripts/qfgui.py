# first attempts in creating a GUI
import sys
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
import importlib


class ApplicationWindow(QtWidgets.QWidget):

    def __init__(self, modelfile):
        super().__init__()
        self.left = 50
        self.top = 50
        self.width = 1200
        self.height = 600
        self.initQube(modelfile)
        self.initUI()

    def initQube(self, modelfile):
        Model = importlib.import_module(modelfile)
        self.qube = Model.set_model()
        self.title = ('Looking at file: ' + self.qube.file + ' with model: ' +
                      self.qube.modelname)
        self.channel = 0
        self.nchannels = self.qube.data.shape[0]
        self.rmsval = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.vmin = -3 * self.rmsval
        self.vmax = 11 * self.rmsval
        self.contours = np.array([2, 3, 6, 9, 12])

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # get the indivudal panels
        self.get_parameter_layout()
        self.get_chanselect_layout()
        self.get_controls_layout()
        self.get_canvas_layout()

        # add to layout
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(self.canvas, 0, 0, 4, 4)
        mainLayout.addWidget(self.parameter, 0, 5, 4, 1)
        mainLayout.addWidget(self.chanselect, 4, 5, 1, 1)
        mainLayout.addWidget(self.controls, 4, 0, 1, 4)

        self.setLayout(mainLayout)

        # show the widget
        self.show()

    def get_parameter_layout(self):
        self.parameter = QtWidgets.QGroupBox('Model Parameters')

        # line edits for all of the parameters
        self.nparameters = len(self.qube.initpar)
        self.lepar_label = []
        self.lepar = []
        for idx, key in enumerate(self.qube.initpar.keys()):
            self.lepar_label.append(QtWidgets.QLabel(self))
            self.lepar_label[idx].setText(key + ': ')
            strval = str(self.qube.initpar[key]['Value'])
            self.lepar.append(QtWidgets.QLineEdit(strval, self))

        # the GO button
        self.button2 = QtWidgets.QPushButton('OK', self)
        self.button2.clicked.connect(self.update_model)

        layout = QtWidgets.QGridLayout()
        for idx in np.arange(self.nparameters):
            layout.addWidget(self.lepar_label[idx], idx, 0)
            layout.addWidget(self.lepar[idx], idx, 1)
        layout.addWidget(self.button2, idx + 1, 0, 1, 2)
        self.parameter.setLayout(layout)

    def get_chanselect_layout(self):
        self.chanselect = QtWidgets.QGroupBox('Channel Select Controls')

        # create the slider
        self.channelslider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, self)
        self.channelslider.setRange(0, self.nchannels - 1)
        self.channelslider.valueChanged.connect(self.sliderchanged)

        # define the first line edit
        self.le1_label = QtWidgets.QLabel(self)
        self.le1_label.setText('Channel: ')
        self.le1 = QtWidgets.QLineEdit(str(self.channel), self)
        self.le1.setToolTip('Type here the channel number you want to display')
        self.le1.returnPressed.connect(self.text_changed)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.channelslider, 0, 0, 1, 5)
        layout.addWidget(self.le1_label, 1, 0, 1, 1)
        layout.addWidget(self.le1, 1, 1, 1, 4)
        self.chanselect.setLayout(layout)

    def get_controls_layout(self):
        self.controls = QtWidgets.QGroupBox('Image Options')

        # contour buttons
        self.dcontour = QtWidgets.QCheckBox('Plot data contours')
        self.mcontour = QtWidgets.QCheckBox('Plot model contours')
        self.dcontour.setChecked(False)
        self.mcontour.setChecked(False)
        self.dcontour.stateChanged.connect(self.update_figures)
        self.mcontour.stateChanged.connect(self.update_figures)

        # RMS and contour levels
        self.rms = QtWidgets.QLabel(self)
        strrms = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.rms.setText('RMS value: {:10.7f}'.format(strrms))
        self.levels = QtWidgets.QLineEdit(__arrtostr__(self.contours), self)
        self.levels.setToolTip('Contours to display (in terms of RMS). ' +
                               'Shold be a list of floats seperated ' +
                               'by commas.')
        self.levels.returnPressed.connect(self.update_contours)

        # min and max values
        self.min_label = QtWidgets.QLabel(self)
        self.min_label.setText('Min: ')
        self.min = QtWidgets.QLineEdit('{:10.7f}'.format(self.vmin), self)
        self.min.setToolTip('The minimum value to display')
        self.min.returnPressed.connect(self.update_min)
        self.max_label = QtWidgets.QLabel(self)
        self.max_label.setText('Max: ')
        self.max = QtWidgets.QLineEdit('{:10.7f}'.format(self.vmax), self)
        self.max.setToolTip('The maxmimum value to display.')
        self.max.returnPressed.connect(self.update_max)

        # layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.dcontour, 0, 0)
        layout.addWidget(self.mcontour, 1, 0)
        layout.addWidget(self.rms, 0, 1)
        layout.addWidget(self.levels, 1, 1)
        layout.addWidget(self.min_label, 0, 2)
        layout.addWidget(self.max_label, 1, 2)
        layout.addWidget(self.min, 0, 3)
        layout.addWidget(self.max, 1, 3)
        self.controls.setLayout(layout)

    def get_canvas_layout(self):
        self.canvas = QtWidgets.QGroupBox('Data and Model Images')

        # the radio buttons
        self.radio1ax1 = QtWidgets.QRadioButton('Data')
        self.radio2ax1 = QtWidgets.QRadioButton('Model')
        self.radio3ax1 = QtWidgets.QRadioButton('Residual')
        self.radio1ax1.setChecked(True)
        self.radio1ax2 = QtWidgets.QRadioButton('Data')
        self.radio2ax2 = QtWidgets.QRadioButton('Model')
        self.radio3ax2 = QtWidgets.QRadioButton('Residual')
        self.radio1ax2.setChecked(True)

        # add the buttons to a group
        self.bgroupax1 = QtWidgets.QButtonGroup(self)
        self.bgroupax1.addButton(self.radio1ax1, 1)
        self.bgroupax1.addButton(self.radio2ax1, 2)
        self.bgroupax1.addButton(self.radio3ax1, 3)
        self.bgroupax2 = QtWidgets.QButtonGroup(self)
        self.bgroupax2.addButton(self.radio1ax2, 1)
        self.bgroupax2.addButton(self.radio2ax2, 2)
        self.bgroupax2.addButton(self.radio3ax2, 3)
        self.bgroupax1.buttonClicked.connect(self.update_figures)
        self.bgroupax2.buttonClicked.connect(self.update_figures)

        # the two canvases
        self.canvas1 = FigureCanvas(Figure(figsize=(5, 5)))
        self._ax1 = self.canvas1.figure.subplots()
        self.canvas2 = FigureCanvas(Figure(figsize=(5, 5)))
        self._ax2 = self.canvas2.figure.subplots()
        self.update_figures()

        # the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.radio1ax1, 0, 0)
        layout.addWidget(self.radio2ax1, 0, 1)
        layout.addWidget(self.radio3ax1, 0, 2)
        layout.addWidget(self.radio1ax2, 0, 3)
        layout.addWidget(self.radio2ax2, 0, 4)
        layout.addWidget(self.radio3ax2, 0, 5)
        layout.addWidget(self.canvas1, 1, 0, 3, 3)
        layout.addWidget(self.canvas2, 1, 3, 3, 3)
        self.canvas.setLayout(layout)

    def text_changed(self):
        try:
            value = int(self.le1.text())
            if value < self.nchannels and value >= 0:
                self.channel = value
            elif value < 0:
                self.channel = 0
            else:
                self.channel = self.nchannels
        except ValueError:
            print('Not a valid integer number')
        self.update_channel()

    def sliderchanged(self):
        self.channel = self.channelslider.value()
        self.update_channel()

    def update_channel(self):
        self.channelslider.setValue(self.channel)
        self.le1.setText(str(self.channel))
        self.rmsval = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.rms.setText('RMS value: {:10.7f}'.format(self.rmsval))
        self.update_figures()

    def update_figures(self):
        self._ax1.clear()
        self._ax2.clear()
        data1 = self.select_data(self.bgroupax1.checkedButton().text())
        data2 = self.select_data(self.bgroupax2.checkedButton().text())
        self._ax1.imshow(data1[self.channel, :, :], origin='lower',
                         cmap='RdYlBu_r', vmin=self.vmin, vmax=self.vmax)
        self._ax2.imshow(data2[self.channel, :, :], origin='lower',
                         cmap='RdYlBu_r', vmin=self.vmin, vmax=self.vmax)
        if self.dcontour.isChecked():
            self._ax1.contour(self.qube.data[self.channel, :, :],
                              colors='black',
                              levels=self.contours * self.rmsval)
            self._ax2.contour(self.qube.data[self.channel, :, :],
                              colors='black',
                              levels=self.contours * self.rmsval)
        if self.mcontour.isChecked():
            self._ax1.contour(self.qube.model[self.channel, :, :],
                              colors='black',
                              levels=self.contours * self.rmsval)
            self._ax2.contour(self.qube.model[self.channel, :, :],
                              colors='black',
                              levels=self.contours * self.rmsval)
        self._ax1.figure.canvas.draw()
        self._ax2.figure.canvas.draw()

    def update_model(self):
        for idx, key in enumerate(self.qube.initpar):
            self.update_parametervalue(key, idx)
            self.qube.load_initialparameters(self.qube.initpar)
            self.qube.create_model()
            self.update_figures()

    def update_parametervalue(self, key, idx):
        try:
            newvalue = float(self.lepar[idx].text())
            self.qube.initpar[key]['Value'] = newvalue
        except ValueError:
            print(self.lepar_label[idx].text() + 'Not a valid value!')
            self.lepar[idx].setText(str(self.qube.initpar[key]['Value']))

    def update_contours(self):
        try:
            self.contours = np.fromstring(self.levels.text(), sep=', ')
            self.update_figures()
        except ValueError:
            print('Not a valid list of contours!')
            self.levels.setText()

    def update_min(self):
        try:
            newvalue = float(self.min.text())
            self.vmin = newvalue
            self.update_figures()
        except ValueError:
            print(self.min.text() + ' is not a valid float!')
            self.min.setText(str(self.vmin))

    def update_max(self):
        try:
            newvalue = float(self.max.text())
            self.vmax = newvalue
            self.update_figures()
        except ValueError:
            print(self.max.text() + ' is not a valid float!')
            self.max.setText(str(self.vmax))

    def select_data(self, button):
        if button == 'Data':
            data = self.qube.data
        elif button == 'Model':
            data = self.qube.model
        else:
            data = self.qube.data - self.qube.model

        return data


def __arrtostr__(array):
    strlevels = ''
    for i in array:
        strlevels = strlevels + str(i) + ', '
    strlevels = strlevels[:-2]
    return strlevels


def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) != 2:
        print(sys.argv)
        print('Please supply the python script that defines the model')
        sys.exit()
    else:
        if sys.argv[1].endswith('.py'):
            sys.argv[1] = sys.argv[1][:-3]
        AppWin = ApplicationWindow(sys.argv[1])
        sys.exit(app.exec_())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) != 2:
        print('Please supply the python script that defines the model')
        sys.exit()
    else:
        if sys.argv[1].endswith('.py'):
            sys.argv[1] = sys.argv[1][:-3]
        AppWin = ApplicationWindow(sys.argv[1])
        sys.exit(app.exec_())
