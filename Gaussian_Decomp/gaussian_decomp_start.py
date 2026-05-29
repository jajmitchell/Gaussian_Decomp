from gp_regression import *
from gaussian_fit import peak_finder, guess_fn, bounds, multi_model_fit, single_model_fit, line_fit, decompose_components
from figures import model_decomp_plot, line_plot
import numpy as np
import json
import pickle
import datetime
import os
import pandas as pd



def hyperparam_opt(time, counts, params):
	"""Performs a grid search over a parameter space to find the optimal hyperparameters.
	Parameters
    ----------

    time : ndarray
        an array of times
    counts : ndarray
        an array of data points can be counts or count rate
    params : dictionary
    	a dictionary of the range of kernel hyperparameters to search over.
	"""
	X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts)

	grid = grid_search(X_scaled,y_scaled, params)
	return grid.best_score_, grid.best_params_


def analyse_series(time, counts, yerr,  frac_t,  frac_v, frac_w, thresh_peak, smoothing, alpha=None, ker_amplitude=None, length_scale=None, string=None, savedir=None, description=None, time_format=None, max_comp=None, min_width=None, max_width=None, model=None, save=False):
    """
    Analyse a single, generic time series using the model decomposition method.
    Fits Gaussian, Lorentzian, and Asymmetric Gaussian models, selects the best by chi-squared,
    saves results and plots.
    """

    # --- Set description for filenames ---
    # if not description:
    #     if time_format is not None:
    #         description = pd.to_datetime(time_format.iloc[0]).strftime('%Y%m%d')
    #     else:
    #         try:
    #             t0 = pd.to_datetime(time[0])
    #         except Exception:
    #             t0 = time[0]
    #         description = t0.strftime('%Y%m%d')

    if smoothing == 'gp':
        # --- Gaussian Process smoothing ---
        X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts)
        X_train, y_train = train_set(X_scaled, y_scaled)
        x_inv, mean_prediction_inv, std_inv = gp_fit(
            X_train, y_train, scaler_x, scaler_y, X_scaled, alpha, ker_amplitude, length_scale
        )
        
    elif smoothing == 'rm':
        # --- Running mean smoothing --- 
        x_inv, mean_prediction_inv, std_inv = running_mean_chi2(time, counts, yerr)

    # --- Peak finding ---
    peak_time, peak_vals, width_time = peak_finder(mean_prediction_inv, x_inv)

    # --- Smooth peaks that are narrower than min_width ---
    if min_width is not None and len(width_time) > 0:
        x_arr = np.asarray(x_inv).astype(float).reshape(-1)
        cadence = np.median(np.diff(x_arr))

        # Fixed smoothing kernel (in samples)
        win = 5
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win) / win

        max_passes = 20
        passes = 0

        while True:
            w = np.asarray(width_time).flatten().astype(float)
            if w.size == 0:
                break

            # Make min_width meaningful even if cadence is coarse
            eff_width = max(min_width, 2.0 * cadence)

            narrow_mask = w < eff_width
            if not np.any(narrow_mask):
                break

            passes += 1
            if passes > max_passes:
                print("Max passes reached in peak smoothing.")
                break

            y_mod = np.array(mean_prediction_inv, dtype=float, copy=True).reshape(-1)
            pt_arr = np.asarray(peak_time).flatten().astype(float)

            for p in pt_arr[narrow_mask]:
                # overwrite region: width = eff_width, centred on p
                left = p - 5 * eff_width
                right = p + 5 * eff_width

                i0 = np.searchsorted(x_arr, left, side="left")
                i1 = np.searchsorted(x_arr, right, side="right")

                if i1 - i0 < 3:
                    continue
                if i0 == 0 or i1 >= len(x_arr):
                    continue

                segment = y_mod[i0:i1]

                # apply fixed-window smoothing inside segment
                win_loc = min(win, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
                if win_loc < 3:
                    continue

                ker_loc = np.ones(win_loc) / win_loc
                seg_pad = np.pad(segment, (win_loc // 2, win_loc // 2), mode="edge")
                y_mod[i0:i1] = np.convolve(seg_pad, ker_loc, mode="valid")

            mean_prediction_inv = y_mod.reshape(mean_prediction_inv.shape)

            # re-find peaks for the next pass
            peak_time, peak_vals, width_time = peak_finder(mean_prediction_inv, x_inv)

    # ---- Filter peaks by error threshold ----
    yerr_interp = np.interp(peak_time.flatten(), time, yerr)
    mask = yerr_interp < thresh_peak * peak_vals.flatten()
    peak_time = peak_time[mask]
    peak_vals = peak_vals[mask]
    width_time = width_time[mask]

    if max_comp is not None:
        sorted_idx = np.argsort(peak_vals)[::-1]
        keep_idx = sorted_idx[:max_comp]
        peak_time = peak_time[keep_idx]
        peak_vals = peak_vals[keep_idx]
        width_time = width_time[keep_idx]
    
    # --- Initial guesses ---
    guess = np.asarray(guess_fn(peak_time, peak_vals, width_time))

    # --- Generate bounds ---
    bounds_lower, bounds_upper = bounds(
        peak_time, peak_vals, width_time, guess, frac_t, frac_v, frac_w,
        min_width=min_width, max_width=max_width
    )

    # Ensure bounds are arrays
    bounds_lower = np.array(bounds_lower, dtype=float).flatten()
    bounds_upper = np.array(bounds_upper, dtype=float).flatten()

    # --- Enforce width limits on bounds and drop invalids ---
    filtered_lower, filtered_upper = [], []

    for i in range(len(bounds_lower)//3):
        bl_amp, bl_mean, bl_width = bounds_lower[3*i:3*i+3]
        bu_amp, bu_mean, bu_width = bounds_upper[3*i:3*i+3]

        # Apply min/max constraints
        if min_width is not None:
            bl_width = max(bl_width, min_width)
        if max_width is not None:
            bu_width = min(bu_width, max_width)

        # Skip component entirely if bounds collapse (invalid interval)
        if bl_width > bu_width:
            continue

        filtered_lower.extend([bl_amp, bl_mean, bl_width])
        filtered_upper.extend([bu_amp, bu_mean, bu_width])

    bounds_lower = np.array(filtered_lower)
    bounds_upper = np.array(filtered_upper)

    epsilon = 1e-8
    # Make sure lower < upper everywhere
    for i in range(len(bounds_lower)):
        if bounds_lower[i] >= bounds_upper[i]:
            # expand upper bound slightly
            bounds_upper[i] = bounds_lower[i] + max(epsilon, 1e-6 * abs(bounds_lower[i]))

    # Clip guess strictly inside bounds
    guess = np.minimum(np.maximum(guess, bounds_lower + epsilon), bounds_upper - epsilon)

    # --- Perform model comparison fit ---
    best_model, popt, pcov, fit, resid, chi2 = multi_model_fit(time, counts, guess, bounds_lower, bounds_upper, model=model)

    # Build individual components
    n = 3
    if model == 'asymmetric_gaussian':
        n = 4
        
    fit_para = decompose_components(time, popt, best_model)

    resid_std = model_decomp_plot(
        time=time, counts=counts,
        fit=fit, fit_para=fit_para, resid=resid, yerr=yerr,
        string=string, description=description,
        time_earth_format=time_format, savedir=savedir,
        model_name=best_model, text = [alpha, ker_amplitude, length_scale], save=save
    )
    # --- Line fit for periodicity ---
    slope, intersect, spear, pearsonr = line_fit(popt, x_inv, n)
    line_plot(
        means=popt[::n], x_inv=x_inv, a=slope, b=intersect,
        string=string, description=description, savedir=savedir,
        model_name=best_model, save=save
    )
    if save is True:
        print(f"Saved decomposition and line fit plots for {description} in {savedir}")
        # --- Save results ---
        save_run(
            time, counts, yerr, alpha, ker_amplitude, length_scale, resid, resid_std,
            slope, popt, popt, fit, time_format=time_format, savedir=savedir, model_name=best_model, description=description
        )

    print(f'Analysis summary \n-----------------'
          f'\n Best model: {best_model}'
          f'\n Number of components: {len(popt)/n}'
          f'\n Slope of Line Fit: {np.round(slope,1)} s'
          f'\n Pearson correlation coefficient, pvalue: {np.round(pearsonr[0],3)}, {pearsonr[1]}'
          f'\n Residual std: {np.round(resid_std,2)}')

    return best_model, fit, popt, resid_std, slope

def error_analysis(time, counts, yerr, alpha, ker_amplitude, length_scale, frac_t, frac_v, frac_w, thresh_peak, smoothing, window_size,
                   string=None, savedir=None, time_format=None, description=None, min_width=None, max_width=None, max_comp=None):
    """
    Perform error analysis by adding random error to the original lightcurve and refitting 50 times
    using the previously selected best-fit model.
    """
    mean = 0
    std = 1
    slopes = []
    intersects = []
    popts = []

    if not description:
        if time_format is not None:
            description = pd.to_datetime(time_format.iloc[0]).strftime('%Y%m%d')
        else:
            try:
                t0 = pd.to_datetime(time[0])
            except Exception:
                t0 = time[0]
            description = t0.strftime('%Y%m%d')

    # --- Load best model from saved pickle ---
    savefilename = 'variables_' + description + ".pkl"
    base_dir = os.path.dirname(savedir)
    save_folder = os.path.join(base_dir, 'save')
    os.makedirs(os.path.expanduser(save_folder), exist_ok=True)
    fname = os.path.join(save_folder, savefilename)

    best_model = None
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)
            best_model = data.get('model_name', None)

    if best_model is None:
        print("Warning: Could not find 'model_name' in saved pickle. Defaulting to Gaussian.")
        best_model = "gaussian"
    else:
        print(f"Loaded best model from previous run: {best_model}")

    # --- Perform 50 iterations of noise-based refits ---
    for i in range(50):
        print(f'Iteration {i}')
        samples = np.random.normal(mean, std, size=counts.size)
        counts_new = counts + samples * yerr

        # --- Smoothing ---
        if smoothing == 'gp':
            X_scaled, y_scaled, scaler_x, scaler_y = rescale(time, counts_new)
            X_train, y_train = train_set(X_scaled, y_scaled)
            x_inv, mean_prediction_inv, std_inv = gp_fit(
                X_train, y_train, scaler_x, scaler_y, X_scaled, alpha, ker_amplitude, length_scale
            )
        elif smoothing == 'running mean':
            x_inv, mean_prediction_inv, std_inv = running_mean(time, counts_new, yerr, window_size)
        elif smoothing == 'adaptive rm':
            window_size = None  
            alpha, ker_amplitude, length_scale = [0, 0, 0]
            x_inv, mean_prediction_inv, std_inv = running_mean_chi2(time, counts_new, yerr)

        # --- Peak finding ---
        peak_time, peak_vals, width_time = peak_finder(mean_prediction_inv, x_inv)

        # --- Filter peaks by error threshold ---
        yerr_interp = np.interp(peak_time.flatten(), time, yerr)
        mask = yerr_interp < thresh_peak * peak_vals.flatten()
        peak_time = peak_time[mask]
        peak_vals = peak_vals[mask]
        width_time = width_time[mask]

        if max_comp is not None:
            sorted_idx = np.argsort(peak_vals)[::-1]
            keep_idx = sorted_idx[:max_comp]
            peak_time = peak_time[keep_idx]
            peak_vals = peak_vals[keep_idx]
            width_time = width_time[keep_idx]

        # --- Initial guesses ---
        guess = np.asarray(guess_fn(peak_time, peak_vals, width_time))

        # Enforce min/max width on guesses
        for j in range(len(guess)//3):
            if min_width is not None and guess[3*j+2] < min_width:
                guess[3*j+2] = min_width
            if max_width is not None and guess[3*j+2] > max_width:
                guess[3*j+2] = max_width

        # --- Generate bounds ---
        bounds_lower, bounds_upper = bounds(
            peak_time, peak_vals, width_time, guess,
            frac_t, frac_v, frac_w, min_width=min_width, max_width=max_width
        )

        # Ensure bounds are arrays and flatten
        bounds_lower = np.array(bounds_lower, dtype=float).flatten()
        bounds_upper = np.array(bounds_upper, dtype=float).flatten()

        # --- Enforce width limits and remove invalid bounds ---
        filtered_lower, filtered_upper = [], []
        for k in range(len(bounds_lower)//3):
            bl_amp, bl_mean, bl_width = bounds_lower[3*k:3*k+3]
            bu_amp, bu_mean, bu_width = bounds_upper[3*k:3*k+3]

            if min_width is not None:
                bl_width = max(bl_width, min_width)
            if max_width is not None:
                bu_width = min(bu_width, max_width)

            # Skip component if bounds collapse
            if bl_width > bu_width:
                continue

            filtered_lower.extend([bl_amp, bl_mean, bl_width])
            filtered_upper.extend([bu_amp, bu_mean, bu_width])

        bounds_lower = np.array(filtered_lower, dtype=float)
        bounds_upper = np.array(filtered_upper, dtype=float)

        # --- Ensure strictly valid bounds and clip guess ---
        epsilon = 1e-8
        for k in range(len(bounds_lower)):
            if bounds_lower[k] >= bounds_upper[k]:
                bounds_upper[k] = bounds_lower[k] + max(epsilon, 1e-6 * abs(bounds_lower[k]))

        guess = np.minimum(np.maximum(guess, bounds_lower + epsilon), bounds_upper - epsilon)

        # --- Fit using best model ---
        popt, pcov, fit, resid, chi2 = single_model_fit(
            time, counts, guess, model=best_model, bounds_lower=bounds_lower, bounds_upper=bounds_upper
        )

        # --- Build components & plot ---
        fit_para = decompose_components(time, popt, best_model)

        file_name = description + f'_Error_analysis_Iteration_{i}'
        n = 3 if best_model != 'asymmetric_gaussian' else 4

        resid_std = model_decomp_plot(
            time=time, counts=counts, x_inv=x_inv,
            mean_prediction_inv=mean_prediction_inv, std_inv=std_inv,
            fit=fit, fit_para=fit_para, resid=resid, yerr=yerr,
            string=string, description=file_name,
            time_earth_format=time_format, savedir=savedir,
            model_name=best_model, text=[alpha, ker_amplitude, length_scale]
        )

        slope, intersect, spear, pearsonr = line_fit(popt, x_inv, n)
        line_plot(means=popt[::n], x_inv=x_inv, a=slope, b=intersect,
                  string=string, description=file_name, savedir=savedir, model_name=best_model)

        slopes.append(slope)
        intersects.append(intersect)
        popts.append(popt)

    # --- Summary statistics ---
    mean_slope = float(np.mean(slopes))
    std_slope = float(np.std(slopes))

    # --- Update pickle with slope data ---
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)
    else:
        data = {}

    data.update({
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "slopes": slopes,
        "fit_parameters_error": popts,
    })

    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Analysis summary \n-----------------'
          f'\n Model: {best_model}'
          f'\n Slope mean: {np.round(mean_slope, 1)} s'
          f'\n Std: {np.round(std_slope, 1)} s'
          f'\n Min: {np.round(np.min(slopes), 1)}'
          f'\n Max: {np.round(np.max(slopes), 1)}')
def save_run(time, counts, yerr, alpha, ker_amplitude, length_scale, resid, resid_std, slope,
             gaus_params, fit_para, fit, time_format=None, description=None, savedir=None, model_name=None):
    """
    Save parameters of run to pickle file, including model name.
    """
    if time_format is not None:
        save_dict = {
            'time': time, 'counts': counts, 'yerr': yerr, 'alpha': alpha,
            'ker_amplitude': ker_amplitude, 'length_scale': length_scale,
            'resid': resid, 'resid_std': resid_std, 'time_format': time_format,
            'slope': slope, 'gaus_params': gaus_params, 'fit_para': fit_para,
            'fit': fit, 'model_name': model_name
        }
    else:
        save_dict = {
            'time': time, 'counts': counts, 'yerr': yerr, 'alpha': alpha,
            'ker_amplitude': ker_amplitude, 'length_scale': length_scale,
            'resid': resid, 'model_name': model_name
        }

    if not description:
        if time_format is not None:
            description = pd.to_datetime(time_format.iloc[0]).strftime('%Y%m%d')
        else:
            try:
                t0 = pd.to_datetime(time[0])
            except Exception:
                t0 = time[0]
            description = t0.strftime('%Y%m%d')

    if not savedir:
        savedir = os.path.expanduser(
            '~/Decomp_repository/save/'
        )
        os.makedirs(savedir, exist_ok=True)
    else:
        base_dir = os.path.dirname(savedir)
        savedir = os.path.join(base_dir, 'save')
        os.makedirs(os.path.expanduser(savedir), exist_ok=True)

    savefilename = 'variables_' + description + ".pkl"
    fname = os.path.join(savedir, savefilename)
    pickle.dump(save_dict, open(fname, 'wb'))