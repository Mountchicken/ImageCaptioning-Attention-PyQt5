import sys
import os
import torch
import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog
from Ui_main import Ui_Form
from PIL import Image
from inferrence import *


class mywindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(mywindow, self).__init__()
        self.cwd=os.getcwd()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.initialize)
        self.pushButton_3.clicked.connect(self.predict)
        self.pushButton.setEnabled(False)
        self.pushButton_3.setEnabled(False)

    def load_image(self):
        img_path,filetype=QFileDialog.getOpenFileName(self,'open image',self.cwd,"*.JPG,*.JPEG,*.png,*.jpg,ALL Files(*)")
        if not img_path=='':
            self.image_path = img_path
            jpg = QtGui.QPixmap(img_path).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)

    def predict(self):
        predict = inferrence(self.encoder,self.decoder,self.vocab,self.image_path,plot=False)
        attentionmap_path = 'attention_map.jpg'
        predict = predict[1:-1] #filter the <sos> and <eos>
        self.textBrowser.setText(' '.join(predict))
        jpg = QtGui.QPixmap(attentionmap_path).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)

    def initialize(self):
        self.textBrowser.setText('Initializing')
        #load vocabulary
        vocab_itos = load_vocab('vocabulary/itos.txt')
        vocab_stoi = load_vocab('vocabulary/stoi.txt')
        vocab = Vocabulary(vocab_itos,vocab_stoi)

        #load model
        checkpoint = torch.load('checkpoint/my_checkpoint.pth.tar')
        encoder = checkpoint['encoder'].to(device)
        decoder = checkpoint['decoder'].to(device)
        encoder.eval()
        decoder.eval()
        self.textBrowser.setText('Initialized')
        self.pushButton.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    myshow=mywindow()
    myshow.show()
    sys.exit(app.exec_())