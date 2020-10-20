from os.path import join as osj
import numpy as np
import matplotlib.pyplot as plt

logfile = "results/spade/11_22_ft_log.txt"
lines = open(logfile).readlines()

def plot_dic(dic, file):
    for k, v in dic.items():
        plt.plot(v)
    plt.legend(list(dic.keys()))
    plt.savefig(file)
    plt.close()

dic = {}
for line in lines:
    items = line.strip().split(" ")
    if len(items) < 10 or items[2] != "maskrcnn_benchmark.trainer":
        continue
    for i in range(8, len(items)):
        if ":" not in items[i]:
            continue
        keyname = items[i][:-1]
        if keyname not in dic.keys():
            dic[keyname] = []
        dic[keyname].append(float(items[i+1]))

plot_dic({k:v for k,v in dic.items() if "loss" in k}, "loss.png")
plot_dic({k:v for k,v in dic.items() if "lr" in k}, "lr.png")