# GUI for data read in to qube
import sys
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from qubefit.qube import Qube


class ApplicationWindow(QtWidgets.QWidget):

    def __init__(self, datafile, *args):
        super().__init__()
        self.left = 20
        self.top = 50
        self.width = 1500
        self.height = 1000
        self.initQube(datafile, *args)
        self.initUI()

    def initQube(self, datafile, *args):
        self.datafile = datafile
        self.qube = Qube.from_fits(datafile)
        if args != ():
            sigcube = Qube.from_fits(args[0])
            self.rmsarr = sigcube.data[:, 0, 0]
        else:
            self.rmsarr = self.qube.calculate_sigma()
        self.title = ('Looking at file: ' + datafile)
        self.channel = 0
        self.nchannels = self.qube.data.shape[0]
        self.range = [[0, self.qube.data.shape[2]],
                      [0, self.qube.data.shape[1]],
                      [0, self.nchannels - 1]]
        self.rmsval = self.rmsarr[self.channel]
        self.vmin = -3 * np.nanmedian(self.rmsarr)
        self.vmax = 11 * np.nanmedian(self.rmsarr)
        self.cmap = 'RdYlBu_r'
        self.contours = np.array([-6, -3, -2, 2, 3, 6, 9, 12])
        self.cval = np.nanmedian(self.rmsarr)
        self.ccolor = 'black'
        self.mrange = [0, self.nchannels]
        self.mask = np.ones_like(self.qube.data)
        self.mom0rms = 0.
        self.mom = [np.ones((self.qube.data.shape[1],
                             self.qube.data.shape[2])),
                    np.ones((self.qube.data.shape[1],
                             self.qube.data.shape[2])),
                    np.ones((self.qube.data.shape[1],
                             self.qube.data.shape[2]))]
        self.mmaskval = [0, 0, 0]

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # get the individual panels
        self.get_chanselect_layout()
        self.get_controls_layout()
        self.get_dataselection_layout()
        self.get_moment_layout()
        self.get_plotselection_layout()
        self.get_canvas_layout()

        # add to layout
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(self.canvas, 0, 0, 3, 3)
        mainLayout.addWidget(self.chanselect, 3, 1)
        mainLayout.addWidget(self.controls, 3, 0)
        mainLayout.addWidget(self.dataselection, 0, 3)
        mainLayout.addWidget(self.moment, 1, 3)
        mainLayout.addWidget(self.plotselection, 3, 3)
        self.setLayout(mainLayout)

        # update/create the figures
        self.update_figure()

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

        # RMS value
        self.rms = QtWidgets.QLabel(self)
        self.rms.setText('RMS value: {:10.7f}'.format(self.rmsval))

        # layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.channeltext_label, 0, 0)
        layout.addWidget(self.channeltext, 0, 1)
        layout.addWidget(self.channelslider, 1, 0, 1, 2)
        layout.addWidget(self.rms, 2, 0, 1, 2)
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

        # Contour value for scaling
        self.contval_label = QtWidgets.QLabel(self)
        self.contval_label.setText('Contour value: ')
        self.contval = QtWidgets.QLineEdit('{:10.7f}'.format(self.cval),
                                           self)
        self.contval.setToolTip('Value used to calculate the contour levels.')
        self.contval.returnPressed.connect(self.update_contourvalue)

        # Contour levels
        self.levels_label = QtWidgets.QLabel(self)
        self.levels_label.setText('Contour levels: ')
        self.levels = QtWidgets.QLineEdit(__arrtostr__(self.contours), self)
        self.levels.setToolTip('Contours levels to display ' +
                               '(in terms of RMS). Should be a list of ' +
                               'floats separated by commas.')
        self.levels.returnPressed.connect(self.update_contourlevels)

        # color map
        self.colormap_label = QtWidgets.QLabel(self)
        self.colormap_label.setText('Colormap: ')
        self.colormap = QtWidgets.QLineEdit('{}'.format(self.cmap), self)
        self.colormap.setToolTip('The color map to use for the display')
        self.colormap.returnPressed.connect(self.update_colormap)

        # Contour colors
        self.levelcolor_label = QtWidgets.QLabel(self)
        self.levelcolor_label.setText('Contour Color: ')
        self.levelcolor = QtWidgets.QLineEdit('{}'.format(self.ccolor), self)
        self.levelcolor.setToolTip('The color of the contours')
        self.levelcolor.returnPressed.connect(self.update_contourcolor)

        # create layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.min_label, 0, 0)
        layout.addWidget(self.min, 0, 1)
        layout.addWidget(self.max_label, 1, 0)
        layout.addWidget(self.max, 1, 1)
        layout.addWidget(self.contval_label, 0, 2)
        layout.addWidget(self.contval, 0, 3)
        layout.addWidget(self.levels_label, 1, 2)
        layout.addWidget(self.levels, 1, 3)
        layout.addWidget(self.colormap_label, 0, 4)
        layout.addWidget(self.colormap, 0, 5)
        layout.addWidget(self.levelcolor_label, 1, 4)
        layout.addWidget(self.levelcolor, 1, 5)
        self.controls.setLayout(layout)

    def get_dataselection_layout(self):
        self.dataselection = QtWidgets.QGroupBox('Data selection')

        # range
        self.r = []
        for idx, label in enumerate(['X: ', 'Y: ', 'Channel: ']):
            self.r.append(QtWidgets.QLabel(label, self))
            self.r.append(QtWidgets.QLineEdit(
                    '{:4d}'.format(self.range[idx][0]), self))
            self.r.append(QtWidgets.QLabel(' to ', self))
            self.r.append(QtWidgets.QLineEdit(
                    '{:4d}'.format(self.range[idx][1]), self))

        # go button
        self.slicebutton = QtWidgets.QPushButton('Slice', self)
        self.slicebutton.clicked.connect(self.update_dataselection)

        # create layout
        layout = QtWidgets.QGridLayout()
        for idx in np.arange(3):
            layout.addWidget(self.r[4 * idx], idx, 0)
            layout.addWidget(self.r[4 * idx + 1], idx, 1)
            layout.addWidget(self.r[4 * idx + 2], idx, 2)
            layout.addWidget(self.r[4 * idx + 3], idx, 3)
        layout.addWidget(self.slicebutton, 3, 0, 1, 4)
        self.dataselection.setLayout(layout)

    def get_moment_layout(self):
        self.moment = QtWidgets.QGroupBox('Moment')

        # The moment range boxes
        self.mr = []
        self.mr.append(QtWidgets.QLabel('Channel:', self))
        self.mr.append(QtWidgets.QLineEdit(
                '{:4d}'.format(self.range[2][0]), self))
        self.mr.append(QtWidgets.QLabel(' to ', self))
        self.mr.append(QtWidgets.QLineEdit(
                '{:4d}'.format(self.range[2][1]), self))

        # The mask buttons
        self.momentmask = []
        masklabel = ['Use mask for moment-0 with values above RMS of: ',
                     'Use mask for moment-1 with values above RMS of: ',
                     'Use mask for moment-2 with values above RMS of: ']
        for label in masklabel:
            self.momentmask.append(QtWidgets.QCheckBox(label))
        self.momentmaskvalue = []
        for mmval in self.mmaskval:
            self.momentmaskvalue.append(QtWidgets.QLineEdit(
                    '{:10.7f}'.format(mmval), self))

        # Gaussian moment plot
        self.gaussianmoment = QtWidgets.QCheckBox('Gaussian moment')
        self.gaussianmoment.setChecked(False)

        # update moment button
        self.momentbutton = QtWidgets.QPushButton('Calculate Moments', self)
        self.momentbutton.clicked.connect(self.get_moments)

        # create layout
        layout = QtWidgets.QGridLayout()
        for idx in np.arange(4):
            layout.addWidget(self.mr[idx], 0, idx)
        for idx in np.arange(3):
            layout.addWidget(self.momentmask[idx], idx + 1, 0, 1, 3)
            layout.addWidget(self.momentmaskvalue[idx], idx + 1, 3)
        layout.addWidget(self.gaussianmoment, 4, 0, 1, 4)
        layout.addWidget(self.momentbutton, 5, 0, 1, 4)

        self.moment.setLayout(layout)

    def get_plotselection_layout(self):

        self.plotselection = QtWidgets.QGroupBox('Plot Selection')

        # raster image and contour button group
        self.bgdata_label = QtWidgets.QLabel('Raster:', self)
        self.bgcont_label = QtWidgets.QLabel('Contour:', self)
        self.bgdata = QtWidgets.QButtonGroup(self)
        self.bgcont = QtWidgets.QButtonGroup(self)
        self.databutton = []
        self.contbutton = []
        for button in ['Channels', 'Moment-0', 'Moment-1', 'Moment-2', 'None']:
            self.databutton.append(QtWidgets.QRadioButton(button))
            self.contbutton.append(QtWidgets.QRadioButton(button))
            self.bgdata.addButton(self.databutton[-1])
            self.bgcont.addButton(self.contbutton[-1])
        self.databutton[0].setChecked(True)
        self.contbutton[-1].setChecked(True)
        self.bgdata.buttonClicked.connect(self.initiate_dataplot)
        self.bgcont.buttonClicked.connect(self.initiate_contplot)

        # create layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.bgdata_label, 0, 0)
        layout.addWidget(self.bgcont_label, 0, 1)
        for idx, button in enumerate(self.databutton):
            layout.addWidget(button, idx + 1, 0)
        for idx, button in enumerate(self.contbutton):
            layout.addWidget(button, idx + 1, 1)
        self.plotselection.setLayout(layout)

    def get_canvas_layout(self):
        self.canvas = QtWidgets.QGroupBox('Canvas')

        # the canvas
        self.fig = FigureCanvas(Figure(figsize=(5, 5)))
        self._ax = self.fig.figure.subplots()

        # the layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.fig, 0, 0)
        self.canvas.setLayout(layout)

    # actions are defined here
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
        self.update_figure()

    def channelslider_changed(self):
        self.channel = self.channelslider.value()
        self.update_channel()
        self.update_figure()

    def update_channel(self):
        self.channelslider.setRange(0, self.nchannels-1)
        self.channelslider.setValue(self.channel)
        self.channeltext.setText(str(self.channel))
        self.rmsval = self.rmsarr[self.channel]
        self.rms.setText('RMS value: {:10.7f}'.format(self.rmsval))

    def update_min(self):
        try:
            newvalue = float(self.min.text())
            self.vmin = newvalue
            self.update_figure()
        except ValueError:
            print(self.min.text() + ' is not a valid float!')
            self.min.setText(str(self.vmin))

    def update_max(self):
        try:
            newvalue = float(self.max.text())
            self.vmax = newvalue
            self.update_figure()
        except ValueError:
            print(self.max.text() + ' is not a valid float!')
            self.max.setText(str(self.vmax))

    def update_colormap(self):
        if self.colormap.text() in plt.colormaps():
            self.cmap = self.colormap.text()
            self.update_figure()
        else:
            print(self.colormap.text() + ' is not a valid color map!')
            self.colormap.setText(self.cmap)

    def update_contourlevels(self):
        try:
            self.contours = np.fromstring(self.levels.text(), sep=', ')
            self.update_figure()
        except ValueError:
            print('Not a valid list of contours!')
            self.levels.setText(__arrtostr__(self.contours))

    def update_contourvalue(self):
        try:
            newvalue = float(self.contval.text())
            self.cval = newvalue
            self.update_figure()
        except ValueError:
            print(self.contval.text() + ' is not a valid float!')
            self.contval.setText(str(self.cval))

    def update_contourcolor(self):
        if self.levelcolor.text() in mpl.colors.cnames.keys():
            self.ccolor = self.levelcolor.text()
            self.update_figure()
        else:
            print(self.levelcolor.text() + ' is not a valid contour color!')
            self.levelcolor.setText(self.ccolor)

    def update_datarange(self):
        for idx in np.arange(3):
            try:
                newvalue = int(self.r[4 * idx + 1].text())
                self.range[idx][0] = newvalue
            except ValueError:
                print(self.r[4 * idx + 1].text() + 'Not a valid value!')
                self.r[4 * idx + 1].setText(str(self.range[idx][0]))
            try:
                newvalue = int(self.r[4 * idx + 3].text())
                self.range[idx][1] = newvalue
            except ValueError:
                print(self.r[4 * idx + 3].text() + 'Not a valid value!')
                self.r[4 * idx + 3].setText(str(self.range[idx][1]))

    def update_dataselection(self):

        self.update_datarange()
        # no consistency checks done here / could add these
        dc = self.qube.get_slice(xindex=(self.range[0][0], self.range[0][1]),
                                 yindex=(self.range[1][0], self.range[1][1]),
                                 zindex=(self.range[2][0], self.range[2][1]))
        self.rmsarr = self.rmsarr[self.range[2][0]: self.range[2][1]]
        self.qube = dc
        self.nchannels = self.qube.data.shape[0]
        for idx in np.arange(3):
            self.r[4 * idx + 1].setText(str(0))
            self.r[4 * idx + 3].setText(str(self.qube.data.shape[2 - idx] - 1))
        self.mr[1].setText('0')
        self.mr[3].setText(str(self.nchannels-1))
        self.channel = 0
        self.update_datarange()
        self.update_momentrange()
        self.update_channel()
        self.update_figure()

    def update_momentrange(self):
        try:
            newvalue = int(self.mr[1].text())
            self.mrange[0] = newvalue
        except ValueError:
            print(self.mr[1].text() + 'Not a valid value!')
            self.mr[1].setText(str(self.mrange[0]))
        try:
            newvalue = int(self.mr[3].text())
            self.mrange[1] = newvalue
        except ValueError:
            print(self.mr[3].text() + 'Not a valid value!')
            self.mr[3].setText(str(self.mrange[1]))

    def update_momentmaskvalue(self):
        for idx in np.arange(3):
            try:
                newvalue = float(self.momentmaskvalue[idx].text())
                self.mmaskval[idx] = newvalue
            except ValueError:
                print(self.momentmaskvalue[idx].text() + 'Not a valid value!')
                self.momentmaskvalue[idx].setText(str(self.mmaskval[idx]))

    def get_moments(self):
        self.update_momentrange()
        self.update_momentmaskvalue()
        channels = np.arange(self.mrange[0], self.mrange[1])
        for idx in np.arange(3):
            if self.momentmask[idx].isChecked():
                tmask = self.qube.mask_region(value=self.mmaskval[idx] *
                                              self.rmsarr)
                tmom = tmask.calculate_moment(moment=idx,
                                              channels=channels)
            else:
                tmom = self.qube.calculate_moment(moment=idx,
                                                  channels=channels)
            if idx == 0:
                self.mom0rms = tmom.calculate_sigma()
            self.mom[idx] = tmom.data
        if self.gaussianmoment.isChecked():
            self.mom[1], self.mom[2] = \
                self.qube.gaussian_moment(mom1=self.mom[1], mom2=self.mom[2])

    def update_momentmask(self):
        if True:
            self.mask = np.where(self.mom0 > 3 * self.mom0rms, 1, 0)

    def select_data(self, button):
        if button == 'Channels':
            data = self.qube.data
        elif button == 'Moment-0':
            data = self.mom[0]
        elif button == 'Moment-1':
            data = self.mom[1]
        elif button == 'Moment-2':
            data = self.mom[2]
        else:
            data = None

        return data

    def initiate_dataplot(self):
        selection = self.bgdata.checkedButton().text()
        Data = self.select_data(selection)
        if selection == 'Channels':
            self.min.setText('{:10.7f}'.format(-3 * np.nanmedian(self.rmsarr)))
            self.max.setText('{:10.7f}'.format(11 * np.nanmedian(self.rmsarr)))
        elif selection == 'None':
            self.min.setText('{:10.7f}'.format(0))
            self.max.setText('{:10.7f}'.format(1))
        else:
            minval = np.nanpercentile(Data, 5)
            maxval = np.nanpercentile(Data, 95)
            self.min.setText('{:10.7f}'.format(minval))
            self.max.setText('{:10.7f}'.format(maxval))
        self.update_min()
        self.update_max()
        self.update_figure()

    def initiate_contplot(self):
        selection = self.bgcont.checkedButton().text()
        if selection == 'Channels':
            self.contval.setText('{:10.7f}'.format(np.nanmedian(self.rmsarr)))
        elif selection == 'Moment-0':
            self.contval.setText('{:10.7f}'.format(self.mom0rms))
        elif selection == 'Moment-1':
            self.contval.setText('{:10.7f}'.format(10))
        elif selection == 'Moment-2':
            self.contval.setText('{:10.7f}'.format(10))
        else:
            self.contval.setText('{:10.7f}'.format(0))
        self.update_contourvalue()
        self.update_figure()

    def update_figure(self):

        self._ax.clear()
        Data = self.select_data(self.bgdata.checkedButton().text())

        if Data is not None:
            if Data.ndim == 2:
                PlotData = Data  # here can apply mask
                self._ax.imshow(PlotData, origin='lower', cmap=self.cmap,
                                vmin=self.vmin, vmax=self.vmax)
            elif Data.ndim == 3:
                self.update_channel()
                PlotData = Data[self.channel, :, :]
                self._ax.imshow(PlotData, origin='lower', cmap=self.cmap,
                                vmin=self.vmin, vmax=self.vmax)
            else:
                raise ValueError('Number of dimensions is not 2 or 3.')
        Cont = self.select_data(self.bgcont.checkedButton().text())
        if Cont is not None:
            if Cont.ndim == 2:
                self._ax.contour(Cont, colors=self.ccolor,
                                 levels=self.contours * self.cval)
            elif Cont.ndim == 3:
                self._ax.contour(Cont[self.channel, :, :], colors=self.ccolor,
                                 levels=self.contours * self.cval)
            else:
                raise ValueError('Number of dimensions is not 2 or 3.')
        self._ax.figure.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) != 2:
        print(sys.argv)
        print('Please supply the fits file for the data')
        sys.exit()
    else:
        AppWin = ApplicationWindow(sys.argv[1])
        AppWin.show()
        sys.exit(app.exec_())


def __arrtostr__(array):
    strlevels = ''
    for i in array:
        strlevels = strlevels + str(i) + ', '
    strlevels = strlevels[:-2]
    return strlevels
