from tkinter import *
import time
import os
root = Tk()

numIdx = 1
# gif的帧数
# 填充numIdx帧内容到frames
frames = [PhotoImage(file='pics/fear.gif', format='gif -index %i' %(i)) for i in range(numIdx)]

def update(idx): # 定时器函数
    frame = frames[idx]
    idx += 1 # 下一帧的序号：在0,1,2,3,4,5之间循环(共6帧)
    label.configure(image=frame) # 显示当前帧的图片
    root.after(100, update, idx%numIdx) # 0.1秒(100毫秒)之后继续执行定时器函数(update)

label = Label(root)
label.pack()
root.after(0, update, 0) # 立即启动定时器函数(update)
root.mainloop()