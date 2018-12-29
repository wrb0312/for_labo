"""
Edit by Keisuke Oyamada @2018/12/25.
For training bachelor.
"""

import json
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="MNIST example for test")
    parser.add_argument("--json_path", "-j", help="model path", required=True, type=str)
    parser.add_argument("--save_path", "-s", help="data save path", required=True, type=str)

    args = parser.parse_args()

    json_path = args.json_path
    save_path = args.save_path

    print("VAE MNIST example for log")
    print("json path: {}".format(json_path))
    print("save path: {}".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_AE_train = []
    loss_KL_train = []
    loss_AE_test = []
    loss_KL_test = []
    epoch = []
    with open(json_path) as f:
        for i in f:
            j = json.loads(i)
            loss_AE_train.append(j["loss_AE_train"])
            loss_KL_train.append(j["loss_KL_train"])
            loss_AE_test.append(j["loss_AE_test"])
            loss_KL_test.append(j["loss_KL_test"])
            epoch.append(j["epoch"])

    plt.plot(epoch, loss_AE_train, label="loss_AE_train", marker="*")
    plt.plot(epoch, loss_KL_train, label="loss_KL_train", marker="*")
    plt.plot(epoch, loss_AE_test, label="loss_AE_test", marker="*")
    plt.plot(epoch, loss_KL_test, label="loss_KL_test", marker="*")
    plt.xticks(epoch[::3])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(os.path.join(save_path, "loss.jpg"))
    plt.close()

if __name__ == "__main__":
    main()
