"""
Edit by Keisuke Oyamada @2018/12/21.
For training bachelor.
"""

import chainer.functions as F
import chainer.links as L
import chainer


class generator(chainer.Chain):
    def __init__(self, z_dim, width, ch, n_cls):
        super(generator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim+n_cls, width//2*width//2*ch//2)
            self.l2 = L.Linear(width//2*width//2*ch//2, width*width*ch)
            self.c3 = L.Deconvolution2D(ch, ch//2, 4, 2, 1)
            self.c4 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1)
            self.c5 = L.Deconvolution2D(ch//4, 1, 3, 1, 1)

            self.bn1 = L.BatchNormalization(width//2*width//2*ch//2)
            self.bn2 = L.BatchNormalization(width*width*ch)
            self.bn3 = L.BatchNormalization(ch//2)
            self.bn4 = L.BatchNormalization(ch//4)

        self.width = width
        self.ch = ch

    def __call__(self, z, c):
        z = F.concat([z, c], axis=1)
        h = F.relu(self.bn1(self.l1(z)))
        h = F.relu(self.bn2(self.l2(h)))
        h = F.reshape(h, (len(z), self.ch, self.width, self.width))
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.bn4(self.c4(h)))
        x = F.tanh(self.c5(h))
        return x


class discriminator(chainer.Chain):
    def __init__(self, ch, n_cls):
        super(discriminator, self).__init__()
        with self.init_scope():
            self.c1_1 = L.Convolution2D(1+n_cls, ch//8, 3, 1, 1)
            self.c1_2 = L.Convolution2D(ch//8, ch//4, 4, 2, 1)
            self.c2_1 = L.Convolution2D(ch//4, ch//4, 3, 1, 1)
            self.c2_2 = L.Convolution2D(ch//4, ch//2, 4, 2, 1)
            self.c3_1 = L.Convolution2D(ch//2, ch//2, 3, 1, 1)
            self.c3_2 = L.Convolution2D(ch//2, ch//1, 4, 2, 1)
            self.l4 = L.Linear(None, 1)

            self.bn1_1 = L.BatchNormalization(ch//8)
            self.bn1_2 = L.BatchNormalization(ch//4)
            self.bn2_1 = L.BatchNormalization(ch//4)
            self.bn2_2 = L.BatchNormalization(ch//2)
            self.bn3_1 = L.BatchNormalization(ch//2)
            self.bn3_2 = L.BatchNormalization(ch//1)

    def __call__(self, x, c):
        x = F.concat([x, c], axis=1)
        h = F.leaky_relu(self.bn1_1(self.c1_1(x)))
        h = F.leaky_relu(self.bn1_2(self.c1_2(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn2_2(self.c2_2(h)))
        h = F.leaky_relu(self.bn3_1(self.c3_1(h)))
        h = F.leaky_relu(self.bn3_2(self.c3_2(h)))
        return self.l4(h)
