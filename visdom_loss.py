# coding:utf8
import visdom
import time
import numpy as np
## python -m visdom.server
## 在浏览器输入：http://localhost:8097/，即可启动

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.viz = visdom.Visdom(env=env, **kwargs)
        x, y, z = 0, 0, 0
        self.win_loss = self.viz.line(
            X=np.array([x]),
            Y=np.column_stack((np.array([y]), np.array([z]))),
            opts=dict(title='Loss', xlable='Epoch', ylabel='Loss', legend=['train', 'test'], showlegend=True))
        self.win_acc = self.viz.line(
            X=np.array([x]),
            Y=np.column_stack((np.array([y]), np.array([z]))),
            opts=dict(title='Acc', xlable='Epoch', ylabel='Loss', legend=['train', 'test'], showlegend=True))

    def plot_update_loss(self,epoch,train_loss,val_loss):
        self.viz.line(
            X=np.array([epoch]),
            Y=np.column_stack((np.array([train_loss]), np.array([val_loss]))),
            opts=dict(title='Loss-source', xlable='Epoch',ylabel='Loss',legend=['train', 'test'], showlegend=True),
            win=self.win_loss,  # win要保持一致
            update='append')
    def plot_update_acc(self,epoch,train_acc,val_acc):
        self.viz.line(
            X=np.array([epoch]),
            Y=np.column_stack((np.array([train_acc]), np.array([val_acc]))),
            opts=dict(title='Acc-source',xlable='Epoch',ylabel='Acc', legend=['train', 'test'], showlegend=True),
            win=self.win_acc,  # win要保持一致
            update='append')
