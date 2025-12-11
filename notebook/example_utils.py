import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from WaveTorch import SimRunner, SimSetting

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['axes.edgecolor'] = '0.3'
plt.rcParams['grid.color'] = '0.85'
plt.rcParams['axes.grid'] = True


def plot_fields(data_ana, data_arl, data_pbl, y_slice=128):

    ana = data_ana.cpu().numpy() if torch.is_tensor(data_ana) else data_ana
    arl = data_arl.cpu().numpy() if torch.is_tensor(data_arl) else data_arl
    pbl = data_pbl.cpu().numpy() if torch.is_tensor(data_pbl) else data_pbl

    cmap = 'RdGy'

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        2,
        4,
        wspace=0.5,
        hspace=0.28,
        width_ratios=[1, 1, 1, 1.25],
        height_ratios=[1, 1],
    )

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        ana, vmin=-1, vmax=1, cmap=cmap, aspect='equal', interpolation='bicubic'
    )
    ax0.set_title('Analytical', pad=8)
    ax0.axis('off')
    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes("right", size="5%", pad=0.08)
    cb0 = fig.colorbar(im0, cax=cax0, ticks=[-1, 0, 1])
    fmt0 = ScalarFormatter(useMathText=True)
    fmt0.set_powerlimits((-3, 3))
    cb0.ax.yaxis.set_major_formatter(fmt0)
    cb0.ax.tick_params(labelsize=9)
    cb0.update_ticks()

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        arl, vmin=-1, vmax=1, cmap=cmap, aspect='equal', interpolation='bicubic'
    )
    ax1.set_title('WaveTorch-ARL', pad=8)
    ax1.axis('off')
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.08)
    cb1 = fig.colorbar(im1, cax=cax1, ticks=[-1, 0, 1])
    fmt1 = ScalarFormatter(useMathText=True)
    fmt1.set_powerlimits((-3, 3))
    cb1.ax.yaxis.set_major_formatter(fmt1)
    cb1.ax.tick_params(labelsize=9)
    cb1.update_ticks()

    ax2 = fig.add_subplot(gs[0, 2])
    diff_arl = ana - arl
    im2 = ax2.imshow(
        diff_arl,
        vmin=-3e-4,
        vmax=3e-4,
        cmap=cmap,
        aspect='equal',
        interpolation='bicubic',
    )
    ax2.set_title('Diff (Ana − ARL)', pad=8)
    ax2.axis('off')
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.08)
    cb2 = fig.colorbar(im2, cax=cax2, ticks=[-0.0003, 0, 0.0003])

    cb2.ax.set_yticklabels([r"$-3\times10^{-4}$", r"$0$", r"$3\times10^{-4}$"])
    cb2.ax.tick_params(labelsize=9)

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(ana[y_slice, :], label='Analytical', color='darkorange', lw=2)
    ax3.plot(arl[y_slice, :], label='WaveTorch-ARL', color='steelblue', lw=2, ls='--')
    ax3.set_title(f'Cross-section y={y_slice}', pad=8)
    ax3.set_xlabel('x position')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='lower right', frameon=False)
    ax3.grid(True, ls='--', alpha=0.45)

    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(
        ana, vmin=-1, vmax=1, cmap=cmap, aspect='equal', interpolation='bicubic'
    )
    ax4.set_title('Analytical', pad=8)
    ax4.axis('off')
    div4 = make_axes_locatable(ax4)
    cax4 = div4.append_axes("right", size="5%", pad=0.08)
    cb4 = fig.colorbar(im4, cax=cax4, ticks=[-1, 0, 1])
    fmt4 = ScalarFormatter(useMathText=True)
    fmt4.set_powerlimits((-3, 3))
    cb4.ax.yaxis.set_major_formatter(fmt4)
    cb4.ax.tick_params(labelsize=9)
    cb4.update_ticks()

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(
        pbl, vmin=-1, vmax=1, cmap=cmap, aspect='equal', interpolation='bicubic'
    )
    ax5.set_title('WaveTorch-PBL', pad=8)
    ax5.axis('off')
    div5 = make_axes_locatable(ax5)
    cax5 = div5.append_axes("right", size="5%", pad=0.08)
    cb5 = fig.colorbar(im5, cax=cax5, ticks=[-1, 0, 1])
    fmt5 = ScalarFormatter(useMathText=True)
    fmt5.set_powerlimits((-3, 3))
    cb5.ax.yaxis.set_major_formatter(fmt5)
    cb5.ax.tick_params(labelsize=9)
    cb5.update_ticks()

    ax6 = fig.add_subplot(gs[1, 2])
    diff_pbl = ana - pbl
    im6 = ax6.imshow(
        diff_pbl,
        vmin=-3e-4,
        vmax=3e-4,
        cmap=cmap,
        aspect='equal',
        interpolation='bicubic',
    )
    ax6.set_title('Diff (Ana − PBL)', pad=8)
    ax6.axis('off')
    div6 = make_axes_locatable(ax6)
    cax6 = div6.append_axes("right", size="5%", pad=0.08)
    cb6 = fig.colorbar(im6, cax=cax6, ticks=[-0.0003, 0, 0.0003])
    cb6.ax.set_yticklabels([r"$-3\times10^{-4}$", r"$0$", r"$3\times10^{-4}$"])
    cb6.ax.tick_params(labelsize=9)

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(ana[y_slice, :], label='Analytical', color='darkorange', lw=2)
    ax7.plot(pbl[y_slice, :], label='WaveTorch-PBL', color='steelblue', lw=2, ls='--')
    ax7.set_title(f'Cross-section y={y_slice}', pad=8)
    ax7.set_xlabel('x position')
    ax7.set_ylabel('Amplitude')
    ax7.legend(loc='lower right', frameon=False)
    ax7.grid(True, ls='--', alpha=0.45)

    fig.suptitle(
        'Wave Propagation Model Comparison', fontsize=14, y=0.96, fontweight='bold'
    )
    # fig.savefig("wave_comparison_final.png", bbox_inches='tight', dpi=400, pad_inches=0.1)
    plt.show()


def run_and_cleanup(sim_runner, src):
    with torch.no_grad():
        output = sim_runner(src)
        result = output.squeeze().cpu()
    gc.collect()
    return result


def Build_solver(pointsource, compute_resource, sim_setting):
    def inner(speed, src):
        src_batch = []
        for i in range(len(src)):
            src_coordiante = [
                {"ix": int(src[i][0]), "iy": int(src[i][1]), "value": 1},
            ]
            src_matrix = pointsource(src_coordiante)
            src_batch.append(src_matrix)
        src_batch_input = torch.concatenate(src_batch, dim=0)
        src_batch_input = compute_resource(src_batch_input)
        sim_runner = SimRunner(compute_resource, speed, sim_setting)
        result = sim_runner(src_batch_input).squeeze()
        gc.collect()
        return result

    return inner
