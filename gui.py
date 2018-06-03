#!/bin/python

import wx
from test import evaluate_one_image
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class HelloFrame(wx.Frame):

    def __init__(self,*args,**kw):
        super(HelloFrame,self).__init__(*args,**kw)

        pnl = wx.Panel(self)

        self.pnl = pnl
        st = wx.StaticText(pnl, label="花朵识别", pos=(200, 0))
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)

        # 选择图像文件按钮
        btn = wx.Button(pnl, -1, "select")
        btn.Bind(wx.EVT_BUTTON, self.OnSelect)

        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("Welcome to flower world")

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                                    "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()

        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Close(True)

    def OnHello(self, event):
        wx.MessageBox("Hello again from wxPython")

    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK | wx.ICON_INFORMATION)

    def OnSelect(self, event):
        wildcard = "image source(*.jpg)|*.jpg|" \
                   "Compile Python(*.pyc)|*.pyc|" \
                   "All file(*.*)|*.*"
        dialog = wx.FileDialog(None, "Choose a file", os.getcwd(),
                               "", wildcard, wx.ID_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            print(dialog.GetPath())
            img = Image.open(dialog.GetPath())
            imag = img.resize([64, 64])
            image = np.array(imag)
            result = evaluate_one_image(image)
            result_text = wx.StaticText(self.pnl, label=result, pos=(320, 0))
            font = result_text.GetFont()
            font.PointSize += 8
            result_text.SetFont(font)
            self.initimage(name= dialog.GetPath())

    # 生成图片控件
    def initimage(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(0,30), size=(600,400))
        return sb


if __name__ == '__main__':

    app = wx.App()
    frm = HelloFrame(None, title='flower wolrd', size=(1000,600))
    frm.Show()
    app.MainLoop()