from tkinter import *
import tkinter.messagebox as messagebox
from lab1_QUO import EmotionClassification
import time
import os


class Application(Frame):
    Ef = EmotionClassification()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.Ef.fit()
        self.pack()
        self.createWidgets()
    # create UI and add the command of button
    def createWidgets(self):
        self.nameLabel = Label(self, text='Please input a sentence here')
        self.nameLabel.pack(fill=X)
        self.nameInput = Text(self, width=60, height=6, highlightcolor='red', highlightthickness=1)
        self.nameInput.pack(fill=X, pady=5)
        self.fileButton = Button(self, width=20, text='Classify', command=self.classify_one_sentence)
        self.fileButton.pack(side=LEFT, pady=5)
        self.alertButton = Button(self, width=20, text='CLear', command=self.clear_text)
        self.alertButton.pack(side=RIGHT, pady=5)
    # create the connection between emoji and label
    # import lab1_QUO to classify on sentence and return the result
    def classify_one_sentence(self):
        newWindow = Toplevel(self)
        newWindow.title("Result")
        numIdxMap = {'joy': 3, 'anger': 8, 'sadness': 3, 'surprise': 5, "love": 30, "fear": 1}

        sentence = self.nameInput.get('0.0', 'end')

        emoji = self.Ef.classify_one_sentence(self.Ef.preprocess_sent(sentence))

        numIdx = numIdxMap[emoji]
        # gif的帧数
        # 填充numIdx帧内容到frames
        # the code below is to transfer the gif to dynamic pic
        # because python can not show the gif
        frames = [PhotoImage(file='pics/' + emoji + '.gif', format='gif -index %i' % (i)) for i in range(numIdx)]

        def update(idx):  # 定时器函数
            frame = frames[idx]
            idx += 1  # 下一帧的序号：在0,1,2,3,4,5之间循环(共6帧)
            label.configure(image=frame)  # 显示当前帧的图片
            newWindow.after(100, update, idx % numIdx)  # 0.1秒(100毫秒)之后继续执行定时器函数(update)

        label = Label(newWindow)
        label.pack(fill=X)
        name_label = Label(newWindow, text='The Emotion of input sentence is  ' + emoji)
        name_label.pack(fill=X, pady=5)
        newWindow.after(0, update, 0)

    def clear_text(self):
        self.nameInput.delete('1.0', 'end')


def main():
    app = Application()
    # 设置窗口标题:
    app.master.title('Emotion Classification')
    app.master.geometry('460x150')
    # 主消息循环:
    app.mainloop()


main()
