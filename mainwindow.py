import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QButtonGroup
import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui.ui_mainwindow import Ui_MainWindow


def cv2qt(cv_image, tw=None, th=None, offset=10):
    rbg_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    h, w, ch = rbg_image.shape
    bytes_per_line = ch * w
    q_image = QImage(rbg_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    if tw is not None and th is not None:
        q_image = q_image.scaled(tw - offset, th - offset, Qt.KeepAspectRatio)

    return q_image


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QMainWindow):
    """
    Current class describe application main window
    """

    def __init__(self):
        super(MainWindow, self).__init__()  # call parent constructor

        # initialize all UI elements.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect radio buttons in a groups
        self.change_color = QButtonGroup(self)
        self.change_color.addButton(self.ui.radio_button_rgb, 1)
        self.change_color.addButton(self.ui.radio_button_gray, 2)
        self.change_color.buttonClicked[int].connect(self.on_color_mode_changed)


        # Bind button events.
        self.ui.read_image.clicked.connect(self.on_read_image_button_clicked)
        self.ui.use_filter_button.clicked.connect(self.filter_apply)
        self.ui.resize_button.clicked.connect(self.resize_image)
        self.ui.save_image_button.clicked.connect(self.save_image)

        # Variables.
        self._original_image = None

        # Create instance of Plot class and pass plot widget to it.
        # self.my_canvas = ScatterMatrix(self.ui.plot_widget_1)

    def image_show(self, image):
        w = self.ui.label_image.width()
        h = self.ui.label_image.height()
        self.ui.label_image.setPixmap(QPixmap.fromImage(cv2qt(image, w, h)))

    def add_charts(self):
        # clear grid layout
        for i in reversed(range(self.ui.content_grid_charts.count())):
            self.ui.content_grid_charts.itemAt(i).widget().setParent(None)
        # add charts
        for c, i in [['r', 2], ['g', 1], ['b', 0]]:
            hist = cv2.calcHist([self._original_image], [i], None, [256], [0, 256])

            sc = MplCanvas(self, width=5, height=4, dpi=100)
            self.ui.content_grid_charts.addWidget(sc)

            # plot the above computed histogram
            # plt.plot(hist, color='b')
            sc.axes.plot(hist, color=c)
            sc.draw()

    def image_to_gray_scale(self):
        image_gray = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        self.image_show(image_gray)

    # ------- UI EVENTS --------------------------------------------------------------------------------------------- #

    def on_read_image_button_clicked(self):
        # print("On method {0}".format("on_data_mode_page_select_file_button_clicked"))

        file, check = QFileDialog.getOpenFileName(None, "Select image file", "",
                                                  "Image Files (*.png);;Image Files (*.jpg);;All Files (*)")

        if not check:
            self._original_image = None
            return

        self._original_image = cv2.imread(file)
        ih, iw, ich = self._original_image.shape

        self.image_show(self._original_image)

        self.ui.label_w.setText("Ширина: " + str(iw))
        self.ui.label_h.setText("Висота: " + str(ih))

        self.add_charts()

    def on_color_mode_changed(self):
        is_rgb = self.ui.radio_button_rgb.isChecked()
        is_gray = self.ui.radio_button_gray.isChecked()

        if is_rgb:
            self.image_show(self._original_image)
        elif is_gray:
            self.image_to_gray_scale()

    def filter_apply(self):
        # read kernel matrix
        r_count = self.ui.kernel_table_widget.rowCount()
        c_count = self.ui.kernel_table_widget.columnCount()

        kernel = np.empty((3, 3))

        # show data
        for i in range(c_count):
            for j in range(r_count):
                kernel[i][j] = self.ui.kernel_table_widget.item(i, j).data(Qt.DisplayRole)

        self._original_image = cv2.filter2D(src=self._original_image, ddepth=-1, kernel=kernel)
        self.image_show(self._original_image)

        self.add_charts()

    def resize_image(self):
        iw = int(self.ui.width_label.toPlainText())
        ih = int(self.ui.height_label.toPlainText())
        self.ui.label_w.setText("Ширина: " + str(iw))
        self.ui.label_h.setText("Висота: " + str(ih))
        # self.ui.label_image.resize(iw, ih)
        self._original_image = cv2.resize(self._original_image, (iw, ih), interpolation=cv2.INTER_AREA)
        self.image_show(self._original_image)

        self.add_charts()

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg)")
        cv2.imwrite(file_path, self._original_image)


   
