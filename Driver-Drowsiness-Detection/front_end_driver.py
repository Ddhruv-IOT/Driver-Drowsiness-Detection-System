from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit
from PyQt5 import uic
import sys
from driver_code_main import main


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi(r".\UI\front.ui", self)

        # find the widgets in the xml file

        self.textedit = self.findChild(QTextEdit, "textEdit")
        self.button = self.findChild(QPushButton, "pushButton")
        self.button.clicked.connect(self.clickedBtn)

        self.show()

    def clickedBtn(self):
        print("Running")
        main()

app = QApplication(sys.argv)
window = UI()
app.exec_()
