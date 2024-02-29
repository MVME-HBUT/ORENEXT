import cv2
import mmcv
import numpy as np
import os
import torch
import torch.nn as nn
# loss 曲线
Loss_list = []  #存储每次epoch损失值
def draw_loss(Loss_list,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("./lossAndacc/Train_loss.png")
    plt.savefig("./lossAndacc/Train_loss.png")
    plt.show()

# acc曲线

def draw_fig(list,name,epoch):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig("./lossAndacc/Train_loss.png")
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig("./lossAndacc/Train _accuracy.png")
        plt.show()


if __name__ == '__main__':
    # val(model)

    with torch.no_grad():
        criterion = nn.BCEWithLogitsLoss().cuda()
        epoch = 12
        loss=[]
        acc=[]
        for i in range(1, epoch + 1):
            dir = "./work_dirs/orenext_stonemlp_sparsefc_ptsdml.py/epoch_" + str(i) + ".pth"
            model = torch.load(dir)
            model.eval()  # 需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值
            loss1,acc1=auto_val(model,criterion)
            loss.append(loss1)
            acc.append(acc1)
        draw_fig(loss,"loss",epoch)
        draw_fig(acc,"acc",epoch)


