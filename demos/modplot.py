import math
import os

import matplotlib.pyplot as plt
import numpy as np
import PyPDF2

# function for creating axes in the MOD plot style
def axes(ratio_plot=True, figsize=(4,4), gridspec_update=None,
         xlabel='', ylabel=r'Probability Density', ylabel_ratio='Ratio to\nSim.', units='', 
         xlim=(0,1), ylim=(0,1), ylim_ratio=(0.5,1.5),
         xticks=None, yticks=None, xtick_step=None, ytick_step=None, ytick_ratio_step=None,
         **kwargs):
    
    # gridspec options
    gridspec_kw = {'height_ratios': (3.5, 1) if ratio_plot else (1,), 'hspace': 0.0}
    if isinstance(gridspec_update, dict):
        gridspec_kw.update(gridspec_update)

    # get subplots
    nsubplots = 2 if ratio_plot else 1
    fig, axes = plt.subplots(nsubplots,  gridspec_kw=gridspec_kw, figsize=figsize)
    if nsubplots == 1:
        axes = [axes]
        
    # axes limits
    for ax in axes:
        ax.set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    if ratio_plot:
        axes[1].set_ylim(*ylim_ratio)
        
    # axes labels
    if units:
        xlabel = r'{} [{}]'.format(xlabel, units)
        ylabel = r'{} [{}{}]'.format(ylabel, units, r'$^{-1}$')
    axes[-1].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if ratio_plot:
        axes[1].set_ylabel(ylabel_ratio, fontsize=8)
        
    # tick settings
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, bottom=True, left=True, direction='in', which='both')
    if ratio_plot:
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(axis='y', labelsize=8)
    
    # tick locations and labels
    if xtick_step is not None:
        xticks_locs = [round(xlim[0] + i*xtick_step, 4) for i in range(1+math.floor((xlim[1]-xlim[0])/xtick_step))]
        axes[-1].set_xticks(xticks_locs)
        if xticks is None:
            axes[-1].set_xticklabels(list(map(str, xticks_locs)))
        else:
            axes[-1].set_xticklabels(xticks)
    if ytick_step is not None:
        yticks_locs = [round(ylim[0] + i*ytick_step, 4) for i in range(1+math.floor((ylim[1]-ylim[0])/ytick_step))]
        axes[0].set_yticks(yticks_locs)
        if yticks is None:
            axes[0].set_yticklabels(list(map(str, yticks_locs)))
        else:
            axes[0].set_yticklabels(yticks)
    if ytick_ratio_step is not None and ratio_plot:
        yticks = [round(ylim_ratio[0] + i*ytick_ratio_step, 4)
                  for i in range(1+round((ylim_ratio[1]-ylim_ratio[0])/ytick_ratio_step))][1:-1]
        axes[1].set_yticks(yticks)
        axes[1].set_yticklabels(list(map(str, yticks)))
    
    return fig, axes

# general function to handle the style
def style(func, **kwargs):
    if func == 'errorbar':
        defaults = {'fmt': 'o', 'lw': 1.5, 'capsize': 1.5, 'capthick': 1, 'markersize': 1.5}
    elif func == 'plot':
        defaults = {'lw': 1.5, 'ls': '-'}
    else:
        raise ValueError('unrecognized plot function {}'.format(func))
    defaults.update(kwargs)
    return defaults

# default style for cms data
def cms_style(*args, **kwargs):
    cms_default = {'func': 'errorbar', 'label': 'CMS 2011 Open Data', 'color': 'black'}
    cms_default.update(kwargs)
    return style(**cms_default)

# default style for sim
def sim_style(*args, **kwargs):
    sim_default = {'func': 'errorbar', 'label': 'CMS 2011 Simulation', 'color': 'orange'}
    sim_default.update(kwargs)
    return style(**sim_default)

# default style for gen
def gen_style(*args, **kwargs):
    gen_default = {'func': 'plot', 'ls': '--', 'label': r'\textsc{Pythia 6} Generation', 'color': 'blue'}
    gen_default.update(kwargs)
    return style(**gen_default)

# handle legend
def legend(ax=None, order=None, **kwargs):
    legend_opts = {'handlelength': 2.0, 'loc': 'upper right', 'frameon': False, 'numpoints': 2}
    legend_opts.update(kwargs)

    if ax is None:
        ax = plt.gca()

    handles, labels = ax.get_legend_handles_labels()

    # reorder curves
    if order is not None:
        if len(labels) != len(order):
            raise ValueError('length of \'order\' must match number of labels')
        handles = np.asarray(handles)[order]
        labels = np.asarray(labels)[order]
        
    # add legend
    ax.legend(handles, labels, **legend_opts)

# function for getting histograms from observable values
def calc_hist(vals, bins=10, weights=None, density=True):
    
    if weights is None:
        weights = np.ones(vals.shape)
    
    # compute histogram
    hist, bins = np.histogram(vals, bins=bins, weights=weights)
    
    # compute which bins the values are in
    digits = np.digitize(vals, bins)

    # compute the errors per bin
    # note that lowest bin value that digitize returns is 1
    # hence the range in the following list comprehension should start at 1
    errs = np.asarray([np.linalg.norm(weights[digits==i]) for i in range(1, len(bins))])

    # handle normalization
    if density:
        density_int = weights.sum() * (bins[1] - bins[0])
        hist /= density_int
        errs /= density_int
        
    return hist, errs, bins

# function to add a stamp to figures
def stamp(left_x, top_y,
          ax=None,
          delta_y=0.075,
          textops_update=None,
          **kwargs):
    
     # handle defualt axis
    if ax is None:
        ax = plt.gca()
    
    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'fontsize': 8.5,
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)
    
    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))
        if t is not None:
            ax.text(left_x, y, t, **textops)

# function for saving plots and optionally adding watermark     
def save(fig, name, add_watermark=True, **kwargs):
    name = name.replace(' ', '').replace('$', '')
    fig.savefig('../plots/bare/{}.pdf'.format(name), bbox_inches='tight')
    if add_watermark:
        watermark('{}.pdf'.format(name), **kwargs)
        
def watermark(plot_file, scale=0.12, tx=44, ty=251,
                         in_plots_dir='../plots/bare',
                         out_plots_dir='../plots',
                         logo_fpath='../plots/bare/MODLogo.pdf',
                         **kwargs):

    # ensure out_plots_dir exists
    os.makedirs(out_plots_dir, exist_ok=True)
    
    # open files for bare plot and the logo
    bare_plot = open(os.path.join(in_plots_dir, plot_file), 'rb')
    logo = open(logo_fpath, 'rb')
    
    # extract pdf pages for bare plot and the logo
    plot_page = PyPDF2.PdfFileReader(bare_plot).getPage(0)
    logo_page = PyPDF2.PdfFileReader(logo).getPage(0)
    
    # add the watermark
    plot_page.mergeScaledTranslatedPage(logo_page, scale, tx, ty, expand=True)
    
    # create a pdf writer for the new plot
    out_plot_pdf = PyPDF2.PdfFileWriter()
    out_plot_pdf.addPage(plot_page)
    
    # write new plot to PDF
    out_plot = open(os.path.join(out_plots_dir, plot_file), 'wb')
    out_plot_pdf.write(out_plot)
    
    # close all files
    bare_plot.close(); logo.close(); out_plot.close()
