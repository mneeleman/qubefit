# first attempts in creating a GUI
import sys
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt


class ApplicationWindow(QtWidgets.QWidget):

    def __init__(self, modelfile, *args):
        super().__init__()
        self.left = 20
        self.top = 50
        self.width = 1500
        self.height = 1000
        self.initQube(modelfile, *args)
        self.initUI()

    def initQube(self, modelfile, *args):
        Model = importlib.import_module(modelfile)
        self.qube = Model.set_model(*args)
        if not hasattr(self.qube, 'file'):
            self.file = 'Not specified'
        self.title = ('Looking at file: ' + self.qube.file + ' with model: ' +
                      self.qube.modelname)
        self.channel = 0
        self.nchannels = self.qube.data.shape[0]
        self.rmsval = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.vmin = -3 * self.rmsval
        self.vmax = 11 * self.rmsval
        self.cmap = 'RdYlBu_r'
        self.contours = np.array([-6, -3, -2, 2, 3, 6, 9, 12])
        self.ccolor = 'black'
        if not hasattr(self.qube, 'maskarray'):
            self.maskarray = np.ones_like(self.qube.data)
        self.chisquared = self.qube.calculate_chisquared()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # get the indivudal panels
        self.get_parameter_layout()
        self.get_chanselect_layout()
        self.get_controls_layout()
        self.get_canvas1_layout()
        self.get_canvas2_layout()

        # add to layout
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(self.canvas1, 0, 0, 3, 2)
        mainLayout.addWidget(self.canvas2, 0, 3, 3, 2)
        mainLayout.addWidget(self.parameter, 0, 6, 3, 1)
        mainLayout.addWidget(self.controls, 3, 0, 1, 6)
        mainLayout.addWidget(self.chanselect, 3, 6, 1, 1)
        self.setLayout(mainLayout)

        # update/create the figures
        self.update_figures()

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
        self.gobutton = QtWidgets.QPushButton('OK', self)
        self.gobutton.clicked.connect(self.update_model)

        # layout
        layout = QtWidgets.QGridLayout()
        for idx in np.arange(self.nparameters):
            layout.addWidget(self.lepar_label[idx], idx, 0)
            layout.addWidget(self.lepar[idx], idx, 1)
        layout.addWidget(self.gobutton, idx + 1, 0, 1, 2)
        self.parameter.setLayout(layout)

    def get_chanselect_layout(self):
        self.chanselect = QtWidgets.QGroupBox('Channel Select Controls')

        # create the channel slider
        self.channelslider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal, self)
        self.channelslider.setRange(0, self.nchannels - 1)
        self.channelslider.valueChanged.connect(self.channelslider_changed)

        # define the channel text box
        self.channeltext_label = QtWidgets.QLabel(self)
        self.channeltext_label.setText('Channel: ')
        self.channeltext = QtWidgets.QLineEdit(str(self.channel), self)
        self.channeltext.setToolTip('The channel number you want to display')
        self.channeltext.returnPressed.connect(self.channeltext_changed)

        # layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.channelslider, 0, 0, 1, 5)
        layout.addWidget(self.channeltext_label, 1, 0, 1, 1)
        layout.addWidget(self.channeltext, 1, 1, 1, 4)
        self.chanselect.setLayout(layout)

    def get_controls_layout(self):
        self.controls = QtWidgets.QGroupBox('Image controls')

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

        # RMS value
        self.rms = QtWidgets.QLabel(self)
        strrms = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.rms.setText('RMS value: {:10.7f}'.format(strrms))

        # Contour levels
        self.levels_label = QtWidgets.QLabel(self)
        self.levels_label.setText('Contour levels: ')
        self.levels = QtWidgets.QLineEdit(__arrtostr__(self.contours), self)
        self.levels.setToolTip('Contours levels to display ' +
                               '(in terms of RMS). Should be a list of ' +
                               'floats separated by commas.')
        self.levels.returnPressed.connect(self.update_contours)

        # color map
        self.colormap_label = QtWidgets.QLabel(self)
        self.colormap_label.setText('Colormap: ')
        self.colormap = QtWidgets.QLineEdit('{}'.format(self.cmap), self)
        self.colormap.setToolTip('The color map to use for the display')
        self.colormap.returnPressed.connect(self.update_cmap)

        # Contour colors
        self.levelcolor_label = QtWidgets.QLabel(self)
        self.levelcolor_label.setText('Contour Color: ')
        self.levelcolor = QtWidgets.QLineEdit('{}'.format(self.ccolor), self)
        self.levelcolor.setToolTip('The color of the contours')
        self.levelcolor.returnPressed.connect(self.update_levelcolor)

        # plot mask button
        self.mask = QtWidgets.QCheckBox('Plot mask')
        self.mask.setToolTip('Plot the mask used for the fitting.')
        self.mask.clicked.connect(self.update_figures)

        # plot chi squared value
        self.chisq = QtWidgets.QLabel(self)
        self.chisq.setText('Red. Chi-Squared: ' +
                           '{:10.7f}'.format(self.chisquared))

        # create layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.min_label, 0, 0)
        layout.addWidget(self.min, 0, 1)
        layout.addWidget(self.max_label, 1, 0)
        layout.addWidget(self.max, 1, 1)
        layout.addWidget(self.rms, 0, 2, 1, 2)
        layout.addWidget(self.levels_label, 1, 2)
        layout.addWidget(self.levels, 1, 3)
        layout.addWidget(self.colormap_label, 0, 4)
        layout.addWidget(self.colormap, 0, 5)
        layout.addWidget(self.levelcolor_label, 1, 4)
        layout.addWidget(self.levelcolor, 1, 5)
        layout.addWidget(self.mask, 0, 6)
        layout.addWidget(self.chisq, 1, 6)
        self.controls.setLayout(layout)

    def get_canvas1_layout(self):
        self.canvas1 = QtWidgets.QGroupBox('Canvas 1')

        # the image selection buttons
        self.im1_bg_label = QtWidgets.QLabel(self)
        self.im1_bg_label.setText('Image: ')
        self.im1_buttons = [QtWidgets.QRadioButton('Data'),
                            QtWidgets.QRadioButton('Model'),
                            QtWidgets.QRadioButton('Residual'),
                            QtWidgets.QRadioButton('None')]
        self.im1_buttons[0].setChecked(True)
        self.im1_bg = QtWidgets.QButtonGroup(self)
        for button in self.im1_buttons:
            self.im1_bg.addButton(button)
        self.im1_bg.buttonClicked.connect(self.update_figures)

        # the contour selection buttons
        self.co1_bg_label = QtWidgets.QLabel(self)
        self.co1_bg_label.setText('Contours: ')
        self.co1_buttons = [QtWidgets.QRadioButton('Data'),
                            QtWidgets.QRadioButton('Model'),
                            QtWidgets.QRadioButton('Residual'),
                            QtWidgets.QRadioButton('None')]
        self.co1_buttons[3].setChecked(True)
        self.co1_bg = QtWidgets.QButtonGroup(self)
        for button in self.co1_buttons:
            self.co1_bg.addButton(button)
        self.co1_bg.buttonClicked.connect(self.update_figures)

        # the canvas
        self.fig1 = FigureCanvas(Figure(figsize=(5, 5)))
        self._ax1 = self.fig1.figure.subplots()

        # the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.fig1, 0, 0, 6, 6)
        layout.addWidget(self.im1_bg_label, 6, 0)
        for idx, button in enumerate(self.im1_buttons):
            layout.addWidget(button, 6, idx + 1)
        layout.addWidget(self.co1_bg_label, 7, 0)
        for idx, button in enumerate(self.co1_buttons):
            layout.addWidget(button, 7, idx + 1)
        self.canvas1.setLayout(layout)

    def get_canvas2_layout(self):
        self.canvas2 = QtWidgets.QGroupBox('Canvas 2')

        # the image selection buttons
        self.im2_bg_label = QtWidgets.QLabel(self)
        self.im2_bg_label.setText('Image: ')
        self.im2_buttons = [QtWidgets.QRadioButton('Data'),
                            QtWidgets.QRadioButton('Model'),
                            QtWidgets.QRadioButton('Residual'),
                            QtWidgets.QRadioButton('None')]
        self.im2_buttons[1].setChecked(True)
        self.im2_bg = QtWidgets.QButtonGroup(self)
        for button in self.im2_buttons:
            self.im2_bg.addButton(button)
        self.im2_bg.buttonClicked.connect(self.update_figures)

        # the contour selection buttons
        self.co2_bg_label = QtWidgets.QLabel(self)
        self.co2_bg_label.setText('Contours: ')
        self.co2_buttons = [QtWidgets.QRadioButton('Data'),
                            QtWidgets.QRadioButton('Model'),
                            QtWidgets.QRadioButton('Residual'),
                            QtWidgets.QRadioButton('None')]
        self.co2_buttons[3].setChecked(True)
        self.co2_bg = QtWidgets.QButtonGroup(self)
        for button in self.co2_buttons:
            self.co2_bg.addButton(button)
        self.co2_bg.buttonClicked.connect(self.update_figures)

        # the canvas
        self.fig2 = FigureCanvas(Figure(figsize=(5, 5)))
        self._ax2 = self.fig2.figure.subplots()

        # the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.fig2, 0, 0, 6, 6)
        layout.addWidget(self.im2_bg_label, 6, 0)
        for idx, button in enumerate(self.im2_buttons):
            layout.addWidget(button, 6, idx + 1)
        layout.addWidget(self.co2_bg_label, 7, 0)
        for idx, button in enumerate(self.co2_buttons):
            layout.addWidget(button, 7, idx + 1)
        self.canvas2.setLayout(layout)

    def channeltext_changed(self):
        try:
            value = int(self.channeltext.text())
            if value < self.nchannels and value >= 0:
                self.channel = value
            elif value < 0:
                self.channel = 0
            else:
                self.channel = self.nchannels
        except ValueError:
            print('Not a valid integer number')
        self.update_channel()

    def channelslider_changed(self):
        self.channel = self.channelslider.value()
        self.update_channel()

    def update_channel(self):
        self.channelslider.setValue(self.channel)
        self.channeltext.setText(str(self.channel))
        self.rmsval = np.nanmedian(np.sqrt(self.qube.variance[self.channel]))
        self.rms.setText('RMS value: {:10.7f}'.format(self.rmsval))
        self.update_figures()

    def update_figures(self):
        # canvas 1
        self._ax1.clear()
        data1 = self.select_data(self.im1_bg.checkedButton().text())
        if data1 is not None:
            self._ax1.imshow(data1[self.channel, :, :], origin='lower',
                             cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        cont1 = self.select_data(self.co1_bg.checkedButton().text())
        if cont1 is not None:
            self._ax1.contour(cont1[self.channel, :, :], colors=self.ccolor,
                              levels=self.contours * self.rmsval)
        self._ax1.figure.canvas.draw()

        # canvas 2
        self._ax2.clear()
        data2 = self.select_data(self.im2_bg.checkedButton().text())
        if data2 is not None:
            self._ax2.imshow(data2[self.channel, :, :], origin='lower',
                             cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        cont2 = self.select_data(self.co2_bg.checkedButton().text())
        if cont2 is not None:
            self._ax2.contour(cont2[self.channel, :, :], colors=self.ccolor,
                              levels=self.contours * self.rmsval)
        self._ax2.figure.canvas.draw()

    def update_model(self):
        for idx, key in enumerate(self.qube.initpar):
            self.update_parametervalue(key, idx)
            self.qube.load_initialparameters(self.qube.initpar)
            self.qube.create_model()
            self.update_figures()
            self.chisquared = self.qube.calculate_chisquared()
            self.chisq.setText('Red. Chi-Squared: ' +
                               '{:10.7f}'.format(self.chisquared))

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
            self.levels.setText(__arrtostr__(self.contours))

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

    def update_cmap(self):
        if self.colormap.text() in plt.colormaps():
            self.cmap = self.colormap.text()
            self.update_figures()
        else:
            print(self.colormap.text() + ' is not a valid color map!')
            self.colormap.setText(self.cmap)

    def update_levelcolor(self):
        if self.levelcolor.text() in mpl.colors.cnames.keys():
            self.ccolor = self.levelcolor.text()
            self.update_figures()
        else:
            print(self.levelcolor.text() + ' is not a valid contour color!')
            self.levelcolor.setText(self.ccolor)

    def select_data(self, button):
        if button == 'Data':
            data = self.qube.data
        elif button == 'Model':
            data = self.qube.model
        elif button == 'Residual':
            data = self.qube.data - self.qube.model
        else:
            data = None

        if self.mask.isChecked() and data is not None:
            data = data * self.qube.maskarray

        return data


def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) <= 2:
        print(sys.argv)
        print('Please supply the python script that defines the model')
        sys.exit()
    elif len(sys.argv) == 2:
        if sys.argv[1].endswith('.py'):
            sys.argv[1] = sys.argv[1][:-3]
        AppWin = ApplicationWindow(sys.argv[1])
        AppWin.show()
        sys.exit(app.exec_())
    elif len(sys.argv) == 3:
        if sys.argv[1].endswith('.py'):
            sys.argv[1] = sys.argv[1][:-3]
        AppWin = ApplicationWindow(sys.argv[1], sys.argv[2])
        AppWin.show()
        sys.exit(app.exec_())
    else:
        print(sys.argv)
        print('Too many arguments supplied to the funcion.')
        sys.exit()


def __arrtostr__(array):
    strlevels = ''
    for i in array:
        strlevels = strlevels + str(i) + ', '
    strlevels = strlevels[:-2]
    return strlevels
