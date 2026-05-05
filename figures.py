import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from matplotlib.dates import DateFormatter
import datetime
import os


def line_plot(means, x_inv, a, b, string=None, description=None, savedir=None, model_name=None, save=False):
    """
    Create and save line plot showing fitted slope vs. peak number.

    Parameters
    ----------
    means : ndarray
        Array of model peak times (centres).
    x_inv : ndarray
        Time array used for inverse scaling of GPR.
    a : float
        Slope of linear fit (seconds per peak).
    b : float
        Intercept of linear fit.
    string : str, optional
        String for folder naming.
    description : str, optional
        Filename identifier (usually date).
    savedir : str, optional
        Directory to save the plot.
    model_name : str, optional
        Name of fitted model ("gaussian", "lorentzian", etc.).
    """
    peak_no = np.arange(1, len(means) + 1)
    means_edit = means - x_inv[0]

    fig, ax = plt.subplots(figsize=[10, 8])
    model_text = f" ({model_name})" if model_name else ""
    ax.plot(peak_no, a * peak_no + b, color='crimson',
            label=f"Slope = {np.round(a, 2)} s{model_text}")
    ax.scatter(peak_no, means_edit, color='lightpink')

    ax.set_xlabel('Peak Number', fontsize='xx-large')
    ax.set_ylabel("Time of peak (s)", fontsize='xx-large')
    ax.tick_params(labelsize='xx-large')
    plt.legend(fontsize='xx-large')

    if not description:
        description = datetime.datetime.now().strftime('%Y%m%d')

    # --- Prepare save directory ---
    # if not savedir:
    #     base = os.path.expanduser('~/Decomp_repository')
    #     path = os.path.join(base, f'plots_{string or "default"}')
    #     os.makedirs(os.path.expanduser(path), exist_ok=True)
    #     savedir = os.path.expanduser(path)
    # else:
    #     os.makedirs(os.path.expanduser(savedir), exist_ok=True)

    # savefilename = f'line_plot_{model_name or "model"}_{description}.jpg'
    # if save is True:
    #     plt.savefig(os.path.join(savedir, savefilename))
    plt.show()
    return peak_no, a, b, means_edit


def model_decomp_plot(time, counts,
                      fit, fit_para, resid, yerr, text,
                      string=None, time_earth_format=None,
                      savedir=None, description=None, model_name=None, save=False):
    """
    Create and save decomposition plot for the fitted model
    (Gaussian, Lorentzian, or Asymmetric Gaussian).

    Parameters
    ----------
    time : ndarray
        Time array.
    counts : ndarray
        Observed count data.
    fit : ndarray
        Model fit (sum of components).
    fit_para : ndarray
        Individual component fits.
    resid : ndarray
        Residuals (fit - counts).
    yerr : ndarray
        Count uncertainties.
    string : str, optional
        String for folder naming.
    time_earth_format : pandas.Series datetime, optional
        Earth time representation of time array.
    savedir : str, optional
        Directory to save the plot.
    description : str, optional
        Filename identifier.
    model_name : str, optional
        Name of fitted model.
    """
    fig = plt.figure(figsize=[12, 8], constrained_layout=False)
    outer_grid = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.2)
    inner_grid = outer_grid[0, 0].subgridspec(2, 1, hspace=0.0, height_ratios=[1, 0.3])
    fig0 = fig.add_subplot(inner_grid[0])
    fig1 = fig.add_subplot(inner_grid[1])

    time_arr = time_earth_format if time_earth_format is not None else time

    # --- Top panel: data + model fit ---
    fig0.errorbar(time_arr, counts, label=r"STIX count rate", zorder=0)
    fig0.fill_between(time_arr, counts - yerr, counts + yerr, alpha=0.5, label='Error', zorder=0)


    label_text = (
        f"Linear combination of {model_name or 'Gaussians'}"
        f"\n α = {text[0]:.2f}" if text[0] is not None else ""
    )
    fig0.plot(time_arr, fit, label=label_text, linewidth=3, zorder=3)

    # Plot each component if provided
    if fit_para is not None and len(fit_para.shape) > 1:
        for i in range(len(fit_para)):
            fig0.plot(time_arr, fit_para[i], '--', linewidth=1.5)

    fig0.set_title(f"{model_name or 'Gaussian'} fit", fontsize='xx-large')
    fig0.legend(fontsize='large')
    fig0.set_ylabel('Count Rate', fontsize='xx-large')
    fig0.tick_params(labelsize='xx-large')

    # --- Bottom panel: residuals ---
    fig1.plot(time_arr, resid / yerr, 'x', label='Residuals / σ')
    fig1.axhline(0, color='black', linewidth=2)
    fig1.set_ylabel('Residual', fontsize='xx-large')
    fig1.set_ylim(-6, 6)
    fig1.tick_params(labelsize='xx-large')

    if not description:
        description = datetime.datetime.now().strftime('%Y%m%d')

    # --- Prepare save directory ---
    # if not savedir:
    #     base = os.path.expanduser('~/Decomp_repository')
    #     path = os.path.join(base, f'plots_{string or "default"}')
    #     os.makedirs(os.path.expanduser(path), exist_ok=True) if save is True else None
    #     savedir = os.path.expanduser(path)
    # else:
    #     os.makedirs(os.path.expanduser(savedir), exist_ok=True) if save is True else None

    # savefilename = f'{model_name or "model"}_decomp_plot_{description}.jpg'
    # if save is True:
    #     plt.savefig(os.path.join(savedir, savefilename))
    plt.show()

    return np.std(resid / yerr)


