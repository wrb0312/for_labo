"""
Edit by Keisuke Oyamada @2018/12/21.
For training bachelor.
"""

import chainer.functions as F
import chainer.links as L
import chainer


class generator(chainer.Chain):
    def __init__(self, z_dim, width, ch):
        super(generator, self).__init__()
        with self.init_scope():
            self.e_c1_1 = L.Convolution2D(1, ch//4, 3, 1, 1)
            self.e_c1_2 = L.Convolution2D(ch//4, ch//4, 3, 1, 1)
            self.e_c2_1 = L.Convolution2D(ch//4, ch//2, 4, 2, 1)
            self.e_c2_2 = L.Convolution2D(ch//2, ch//2, 3, 1, 1)
            self.e_c3_1 = L.Convolution2D(ch//2, ch, 4, 2, 1)
            self.e_c3_2 = L.Convolution2D(ch, ch, 3, 1, 1)
            self.e_l4 = L.Linear(width//4*width//4*ch, z_dim)

            self.d_l1_1 = L.Linear(z_dim, width//4*width//4*ch)
            self.d_c1_2 = L.Deconvolution2D(ch, ch, 3, 1, 1)
            self.d_c2_1 = L.Deconvolution2D(ch, ch//2, 4, 2, 1)
            self.d_c2_2 = L.Deconvolution2D(ch//2, ch//2, 3, 1, 1)
            self.d_c3_1 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1)
            self.d_c3_2 = L.Deconvolution2D(ch//4, ch//4, 3, 1, 1)
            self.d_c4 = L.Deconvolution2D(ch//4, 1, 3, 1, 1)

            self.e_bn1_1 = L.BatchNormalization(ch//4)
            self.e_bn1_2 = L.BatchNormalization(ch//4)
            self.e_bn2_1 = L.BatchNormalization(ch//2)
            self.e_bn2_2 = L.BatchNormalization(ch//2)
            self.e_bn3_1 = L.BatchNormalization(ch)
            self.e_bn3_2 = L.BatchNormalization(ch)
            self.e_bn4 = L.BatchNormalization(z_dim)

            self.d_bn1_1 = L.BatchNormalization(ch)
            self.d_bn1_2 = L.BatchNormalization(ch)
            self.d_bn2_1 = L.BatchNormalization(ch//2)
            self.d_bn2_2 = L.BatchNormalization(ch//2)
            self.d_bn3_1 = L.BatchNormalization(ch//4)
            self.d_bn3_2 = L.BatchNormalization(ch//4)

        self.width = width
        self.ch = ch


    def encode(self, x):
        h = F.leaky_relu(self.e_bn1_1(self.e_c1_1(x)))
        h = F.leaky_relu(self.e_bn1_2(self.e_c1_2(h)))
        h = F.leaky_relu(self.e_bn2_1(self.e_c2_1(h)))
        h = F.leaky_relu(self.e_bn2_2(self.e_c2_2(h)))
        h = F.leaky_relu(self.e_bn3_1(self.e_c3_1(h)))
        h = F.leaky_relu(self.e_bn3_2(self.e_c3_2(h)))
        z = self.e_bn4(self.e_l4(h))

        return z

    def decode(self, z):
        batch_size = len(z)

        h = self.d_l1_1(z)
        h = F.leaky_relu(self.d_bn1_1(h.reshape(batch_size, self.ch, self.width//4, self.width//4)))
        h = F.leaky_relu(self.d_bn1_2(self.d_c1_2(h)))
        h = F.leaky_relu(self.d_bn2_1(self.d_c2_1(h)))
        h = F.leaky_relu(self.d_bn2_2(self.d_c2_2(h)))
        h = F.leaky_relu(self.d_bn3_1(self.d_c3_1(h)))
        h = F.leaky_relu(self.d_bn3_2(self.d_c3_2(h)))
        x = F.sigmoid(self.d_c4(h))

        return x

    def __call__(self, x):
        z = self.encode(x)
        x_re = self.decode(z)

        return x_re, z


class discriminator(chainer.Chain):
    def __init__(self, z_dim, ch):
        super(discriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, 1024)
            self.l2 = L.Linear(1024, 1024)
            self.l3 = L.Linear(1024, 512)
            self.l4 = L.Linear(512, 2)

            self.bn1 = L.BatchNormalization(1024)
            self.bn2 = L.BatchNormalization(1024)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, z):
        h = F.leaky_relu(self.bn1(self.l1(z)))
        h = F.dropout(h, ratio=0.2)
        h = F.leaky_relu(self.bn2(self.l2(h)))
        h = F.dropout(h, ratio=0.2)
        h = F.leaky_relu(self.bn3(self.l3(h)))
        return self.l4(h)
