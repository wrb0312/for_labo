"""
Edit by Keisuke Oyamada @2018/12/21.
For training bachelor.
"""

import chainer.functions as F
import chainer.links as L
import chainer


class sampleNet(chainer.Chain):
    def __init__(self):
        super(sampleNet, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, 392)
            self.l2 = L.Linear(392, 100)
            self.l3 = L.Linear(100, 10)

    def __call__(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
