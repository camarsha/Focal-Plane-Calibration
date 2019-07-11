"""
short module to give a R-like fit+residual plot function
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
import matplotlib.gridspec as gridspec

sbs.set_style('whitegrid')

def residual_plot(datax, datay, err, fit_funtion, xerr=None):

    res = datay-fit_funtion(datax)  # calculate the residuals

    # generate fit line
    x = np.linspace(min(datax), max(datax), 10000)  # 10000 points for line
    y = fit_funtion(x)

    # create the the grid that the fits and residuals will be show in.
    grid = gridspec.GridSpec(2, 1,
                             height_ratios=[2, 1],
                             wspace=0.0, hspace=0.0)
    # first the trend line
    ax0 = plt.subplot(grid[0])  # pick the larger area plot for fit plot
    ax0.set_xticklabels([])  # remove the x labels
    plt.ylabel(r'$\rho$ (cm)', fontsize=32)
    plt.errorbar(datax, datay, xerr=xerr,
                 yerr=err, fmt='ob')
    plt.plot(x, y, color='k', lw=2.0)  # the fit line
    plt.title('Fit and Residuals')  # add title on the top subplot
    # next the residuals
    ax1 = plt.subplot(grid[1])  # switch to smaller subplot
    plt.errorbar(datax, res, xerr=xerr, yerr=err, fmt='ob')  
    plt.ylabel('Residuals', fontsize=32)
    plt.xlabel('Channel #', fontsize=32)
    # draw a horizontal line at 0 to guide the eye
    plt.axhline(y=0, linewidth=2, color='k', linestyle='--')
    # these set the dashed border of the residual subplot
    [j.set_linewidth(3.0) for j in ax1.spines.values()]
    [j.set_color('k') for j in ax1.spines.values()]
    [j.set_linestyle('--') for j in ax1.spines.values()]

