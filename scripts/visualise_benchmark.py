
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import cm, colorbar, colors
from mpl_toolkits.mplot3d import Axes3D  # dla projekcji 3d
import numpy as np
def plot_results_3d(simulations):
    ax = plt.subplot(projection='3d')
    mns, avr, var, tcx, met, _ = prepare_averages(simulations)

    vr = np.ravel(var)
    top = np.ravel(mns)
    xx = [[i] * len(met) for i in range(len(tcx))]
    yy = [i for i in range(len(met))] * len(tcx)
    xx = np.ravel(xx)
    top = np.log10(top)*10
    mx = max(top)
    mn = min(top)
    alpha = 0.7
    color_mapping = [cm.RdYlGn_r((i-mn)/(mx-mn)) for i in top]
    color_mapping = [(r,g,b,alpha) for r,g,b,a in color_mapping]

    # print("x\n", xx, "y\n", yy, "top\n", top)
    ax.bar3d(xx, yy, np.zeros_like(top), 0.9, 0.9, top, color=color_mapping)
    if not config.nolabels:
        for i in range(len(top)):
            #text = '{:.1f}±{:.1f}'.format(10**(top[i]/10), vr[i]**0.5)
            text = to_si(10**(top[i]/10)*1e-6)#'{:.1e}'.format(10**(top[i]/10))
            ax.text3D(xx[i]+0.5, yy[i]+0.5, top[i]+1, text, horizontalalignment='center')

    # ax.set_zscale('log')
    ax.set_xlabel("Task complexity")
    ax.set_xlim(0,max(len(met), len(tcx)))
    ax.set_xticks([i+0.5 for i in range(len(tcx))])
    ax.yaxis.set_tick_params(labelrotation=-20)
    ax.set_xticklabels(tcx)
    ax.set_ylabel("Method")
    ax.set_ylim(0,max(len(met), len(tcx)))
    ax.set_yticks([i+0.5 for i in range(len(met))])
    ax.set_yticklabels([re.sub('[a-z]', '', m) for m in met], rotation_mode='anchor')
    ax.set_zlabel("CPU time [10log_10 us]")
    ax.set_title("CPU time consumption comparison 2 dimensions")
    ax.view_init(elev=45., azim=-45.0)
    # ax.autoscale(tight=True)
    plt.show()
