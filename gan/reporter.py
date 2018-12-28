from collections import OrderedDict
import os
import json


class reporter:
    def __init__(self, keys_general, keys_train, keys_test, N, N_test, save_path):
        self.keys_general = keys_general
        self.keys_train = keys_train
        self.keys_test = keys_test

        log_txt = ""

        for v in keys_general:
            log_txt += (v + "\t")
            setattr(self, v, 0.)
        for v in keys_train:
            log_txt += (v + "\t")
            setattr(self, v, 0.)
        for v in keys_test:
            log_txt += (v + "\t")
            setattr(self, v, 0.)

        self.N = N
        self.N_test = N_test
        self.save_path = os.path.join(save_path, "log.json")

        if os.path.exists(self.save_path):
            os.remove(self.save_path)

        print(log_txt)

    def __call__(self, keys_general, keys_train, keys_test):
        self.log_txt = ""
        self.log = OrderedDict()

        self.output_log(keys_general, None)
        self.output_log(keys_train, self.N)
        self.output_log(keys_test, self.N_test)

        print(self.log_txt)
        with open(self.save_path, "a") as f:
            # json.dump(self.log, f, indent=2)
            json.dump(self.log, f)
            f.write("\n")

    def output_log(self, keys, num):
        for k in keys:
            if num is not None:
                v = getattr(self, k) / num
            else:
                v = getattr(self, k)

            if isinstance(v, float):
                self.log_txt += ("{:.6f}\t".format(v))
            else:
                self.log_txt += ("{}\t".format(v))

            self.log[k] = v

    def init_log(self):
        for k in self.keys_train:
            setattr(self, k, 0)
        for k in self.keys_test:
            setattr(self, k, 0)
