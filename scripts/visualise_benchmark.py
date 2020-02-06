#!/bin/python3
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import cm, colorbar, colors
from mpl_toolkits.mplot3d import Axes3D  # dla projekcji 3d
import numpy as np
import sys
import csv
import re

from bisect import bisect_left
def to_si(number):
    idx = max(bisect_left(to_si.low_bound, number+1e-15) - 1, 0)
    return "{: >3.1f}{}".format(number/to_si.low_bound[idx], to_si.unit[idx])
to_si.low_bound = [10**(x) for x in range(-15, 16, 3)]
to_si.unit = ['f', 'p', 'n', 'u', 'm', '', 'k' ,'M', 'G', 'T', 'P']

if __name__ == "__main__":
    infiles = sys.argv[1:]
    methods = []
    scales = dict()
    values = []
    for infile in infiles:
        with open(infile, 'r') as csvfile:
            benchmark_reader = csv.reader(csvfile, delimiter=',')
            for _ in range(9):
                next(benchmark_reader, None)
                
            # row[0] - __name__
            # row[2] - real_time
            method = infile.split('/')[-1].split('.')[0]
            
            data_line = []
            for i, row in enumerate(benchmark_reader):
                time = float(row[2])
                
                name = row[0].split('/')
                
                function = row[0].split('/')[0].split('_')[-1]
                
                dimX = int(name[1])
                dimY = int(name[2])
                dimK = int(name[3])
                
                scales[2*(dimX+dimK-1)*(dimY+dimK-1)*dimK*dimK] = i
                print(function+'_'+method, dimX*dimY, dimK, time)
                
                data_line.append(time)
            values.append(data_line)
            methods.append(function+'_'+method)
    print(methods, sorted(scales.items()))
    # reordering
    new_order = [pair[1] for pair in sorted(scales.items())]
    values = [[entry[i] for i in reversed(new_order)] for entry in values]
    values = np.mat(values)
            
    ax = plt.subplot(projection='3d')
    #mns, avr, tcx, met, _ = prepare_averages(simulations)

    top = np.ravel(values)
    X = len(scales)
    Y = len(methods)
    xx = [[i] * X for i in range(Y)]
    yy = [i for i in range(X)] * Y
    xx = np.ravel(xx)
    #top = np.log10(top)*10
    mx = np.max(values)
    mn = np.min(values)
    alpha = 0.7
    color_mapping = [cm.RdYlGn_r((i-mn)/(mx-mn)) for i in top]
    color_mapping = [(r,g,b,alpha) for r,g,b,a in color_mapping]

    ax.bar3d(xx, yy, np.zeros_like(top), 0.9, 0.9, top, color=color_mapping)
    for i in range(len(top)):
        text = to_si(top[i]*1e-9)
        ax.text3D(xx[i]+0.5, yy[i]+0.5, top[i]+1, text, horizontalalignment='center')

    # ax.set_zscale('log')
    ax.set_xlabel("")
    #ax.set_xlim(0,max(len(methods), len(scales)))
    ax.set_xticks([i+0.5 for i in range(len(methods))])
    ax.yaxis.set_tick_params(labelrotation=-20)
    ax.set_xticklabels(methods)
    ax.set_ylabel("FLOPS")
    #ax.set_ylim(0,max(len(scales), len(methods)))
    ax.set_yticks([i+0.5 for i in range(len(scales))])
    ax.set_yticklabels([to_si(m) for m in reversed(scales)], rotation_mode='anchor')
    ax.set_zlabel("CPU time [ns]")
    ax.set_title("CPU/GPU real time consumption comparison")
    ax.view_init(elev=79., azim=-42.0)
    
    plt.tight_layout()
    # ax.autoscale(tight=True)
    plt.show()
