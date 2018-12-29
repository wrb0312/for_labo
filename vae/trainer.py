import chainer
import numpy as np
from tqdm import tqdm
import math
import os
from chainer import serializers
import chainer.functions as F
from reporter import reporter
from chainer import dataset


def update(dec, opt_dec, enc, opt_enc, loss, is_unchain=False):
    dec.cleargrads()
    enc.cleargrads()
    loss.backward()
    opt_dec.update()
    opt_enc.update()
    if is_unchain:
        loss.unchain_backward()


class Trainer:
    def __init__(self, **kwargs):
        self.E, self.D = kwargs.pop("model")
        self.opt_E, self.opt_D = kwargs.pop("opt")
        self.epochs = kwargs.pop("epochs")
        self.save_path = kwargs.pop("save_path")
        self.iter_train, self.iter_test = kwargs.pop("iterator")
        N, N_test = kwargs.pop("data_size")
        batch_size, batch_size_test = kwargs.pop("batch_size")
        self.z_dim = kwargs.pop("z_dim")
        self.xp = self.E.xp

        self.keys_general = ["epoch"]
        self.keys_train = ["loss_AE_train",
                           "loss_KL_train"]
        self.keys_test = ["loss_AE_test",
                          "loss_KL_test"]
        self.report = reporter(self.keys_general, self.keys_train, self.keys_test, N, N_test, self.save_path)
        self.N, self.N_test = math.ceil(N / batch_size), math.ceil(N_test / batch_size_test)

    def __call__(self):
        for epoch in range(1, self.epochs+1):
            self.report.epoch = epoch

            ### train ###
            num_train = 0
            bar = tqdm(desc="Training", total=self.N, leave=False)
            for d in self.iter_train:
                x, _ = dataset.concat_examples(d)
                x = x * 2 - 1
                loss_AE, loss_KL = self.forward(x)
                self.report.loss_AE_train += float(loss_AE * len(x))
                self.report.loss_KL_train += float(loss_KL * len(x))
                num_train += len(x)

                bar.set_description("epoch: {}, loss_AE: {:.6f}, loss_KL: {:.6f}".format(
                    epoch,
                    self.report.loss_AE_train / num_train,
                    self.report.loss_KL_train / num_train),
                    refresh=False)
                bar.update()
            bar.close()

            ### test ###
            num_test = 0
            bar_test = tqdm(desc="Test", total=self.N_test, leave=False)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                for d in self.iter_test:
                    x, _ = dataset.concat_examples(d)
                    x = x * 2 - 1
                    loss_AE, loss_KL = self.forward(x)
                    self.report.loss_AE_test += float(loss_AE * len(x))
                    self.report.loss_KL_test += float(loss_KL * len(x))
                    num_test += len(x)
                    bar_test.update()
            bar_test.close()

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.report(self.keys_general, self.keys_train, self.keys_test)

            self.iter_train.reset()
            self.iter_test.reset()
            self.report.init_log()

            serializers.save_npz(os.path.join(
                self.save_path, "epoch{}.enc".format(epoch)), self.E)
            serializers.save_npz(os.path.join(
                self.save_path, "epoch{}.dec".format(epoch)), self.D)
            serializers.save_npz(os.path.join(
                self.save_path, "epoch{}.opt_enc".format(epoch)), self.opt_E)
            serializers.save_npz(os.path.join(
                self.save_path, "epoch{}.opt_dec".format(epoch)), self.opt_D)

    def kld(self, mean, log_sigma):
        kl = -0.5 * (1+log_sigma - F.square(mean) - F.exp(log_sigma))
        return F.mean(F.sum(kl, 1))

    def forward(self, x):
        batch_size = len(x)
        x = x.reshape(batch_size, 1, 28, 28)
        x_real = chainer.Variable(self.xp.asarray(x))
        z_mean, z_log_sig = self.E(x_real)
        eps = chainer.Variable(self.xp.random.random(z_mean.shape).astype(self.xp.float32))
        z = z_mean + F.square(F.exp(z_log_sig)) * eps
        x_fake = self.D(z)

        loss_AE = F.sum(F.square(x_real - x_fake)) / batch_size
        loss_KL = self.kld(z_mean, z_log_sig)

        if chainer.config.train:
            loss = loss_AE + loss_KL
            update(self.D, self.opt_D, self.E, self.opt_E, loss, True)

        return loss_AE.data, loss_KL.data
