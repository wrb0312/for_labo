"""
Edit by Keisuke Oyamada @2018/12/21.
For training bachelor.
"""

import chainer.functions as F
import chainer.links as L
import chainer


class encoder(chainer.Chain):
    def __init__(self, z_dim, width, ch):
        super(encoder, self).__init__()
        with self.init_scope():
            self.c1_1 = L.Convolution2D(1, ch//4, 5, 1, 2)
            self.c1_2 = L.Convolution2D(ch//4, ch//2, 4, 2, 1)
            self.c2_1 = L.Convolution2D(ch//2, ch//2, 5, 1, 2)
            self.c2_2 = L.Convolution2D(ch//2, ch, 4, 2, 1)
            self.l3 = L.Linear(None, width//2*width//2*ch//2)
            self.l4_mu = L.Linear(None, z_dim)
            self.l4_si = L.Linear(None, z_dim)

            self.bn1_1 = L.GroupNormalization(32, ch//4)
            self.bn1_2 = L.GroupNormalization(32, ch//2)
            self.bn2_1 = L.GroupNormalization(32, ch//2)
            self.bn2_2 = L.GroupNormalization(32, ch)
            self.bn3 = L.LayerNormalization(width//2*width//2*ch//2)
            self.bn4_mu = L.LayerNormalization(z_dim)
            self.bn4_si = L.LayerNormalization(z_dim)

        self.width = width
        self.ch = ch

    def __call__(self, x):
        h = F.relu(self.bn1_1(self.c1_1(x)))
        h = F.relu(self.bn1_2(self.c1_2(h)))
        h = F.relu(self.bn2_1(self.c2_1(h)))
        h = F.relu(self.bn2_2(self.c2_2(h)))
        h = F.relu(self.bn3(self.l3(h)))
        z_mu = self.bn4_mu(self.l4_mu(h))
        z_si = F.softplus(self.bn4_si(self.l4_si(h)))
        return z_mu, z_si


class decoder(chainer.Chain):
    def __init__(self, z_dim, width, ch):
        super(decoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(z_dim, width//2*width//2*ch//2)
            self.l2 = L.Linear(width//2*width//2*ch//2, width*width*ch)
            self.c3 = L.Deconvolution2D(ch, ch//2, 4, 2, 1)
            self.c4 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1)
            self.c5 = L.Deconvolution2D(ch//4, 1, 3, 1, 1)

            self.bn1 = L.LayerNormalization(width//2*width//2*ch//2)
            self.bn2 = L.LayerNormalization(width*width*ch)
            self.bn3 = L.GroupNormalization(32, ch//2)
            self.bn4 = L.GroupNormalization(32, ch//4)

        self.width = width
        self.ch = ch

    def __call__(self, z):
        h = F.relu(self.bn1(self.l1(z)))
        h = F.relu(self.bn2(self.l2(h)))
        h = F.reshape(h, (len(z), self.ch, self.width, self.width))
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.bn4(self.c4(h)))
        x = F.tanh(self.c5(h))
        return x
