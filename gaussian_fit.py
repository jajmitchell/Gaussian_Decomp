from pyexpat import model
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
#from gp_regression import *

def peak_finder(mean_prediction,x_inv):
	grad = np.gradient(mean_prediction.reshape(-1))
	y = [0 for _ in range(len(mean_prediction))]
	idx = np.argwhere(np.diff(np.sign(grad - y)))
	val = [mean_prediction[i] for i in idx.reshape(-1)]
		
	remove_first = False
	remove_last = False
	
	if val[0] > val[1]:
		remove_first = True
	
	if val[-1] < val[-2]:
		remove_last = True
	
	# Apply the modifications after checking conditions
	if remove_first:
		idx = idx[1:]  # Remove the first element
		val = val[1:]  # Remove the first value
	
	if remove_last:
 		idx = idx[:-1]  # Remove the last element
 		val = val[:-1]  # Remove the last value

	time_idx = [x_inv[i].reshape(-1) for i in idx]

	idx_every_second = idx[1::2] 

	base_time = np.asarray(time_idx[::2])
	peak_time = np.asarray(time_idx[1::2])

	width_time = peak_time-base_time #in s
	peak_vals = np.asarray([mean_prediction[i] for i in idx_every_second]).reshape(-1)
	return peak_time, peak_vals, width_time

def guess_fn(peak_time,peak_vals,width_time):
	guess=[]
	for i in range(len(width_time)):
	    guess.append(peak_time.reshape(-1)[i]) 
	    guess.append(peak_vals[i])  
	    guess.append(width_time.reshape(-1)[i] / 2.355)
	return guess

def gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2] #sigma
        y = y + np.abs(amp) * np.exp(-((x - ctr)/wid) ** 2)
    return y

def lorentzian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y += np.abs(amp) * (wid ** 2) / ((x - ctr) ** 2 + wid ** 2)
    return y

def asymmetric_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        ctr = params[i]
        amp = params[i + 1]
        wl = params[i + 2]
        wr = params[i + 3]
        y += np.abs(amp) * np.where(x < ctr,
                                    np.exp(-((x - ctr) / wl) ** 2),
                                    np.exp(-((x - ctr) / wr) ** 2))
    return y

def bounds(peak_time, peak_vals, width_time, guess,
           frac_t, frac_v, frac_w, min_val=1e-6, min_width=None, max_width=None):

    bounds_lower, bounds_upper = [], []
    n_peaks = len(guess) // 3
    epsilon = 1e-8  # small safety buffer

    for i in range(n_peaks):
        t = np.asarray(peak_time[i]).item()
        v = np.asarray(peak_vals[i]).item()
        w = np.asarray(width_time[i]).item()
        g_t, g_v, g_w = guess[3*i:3*i+3]

        # Time bounds
        dt = max(abs(t) * frac_t, 10)
        lo_t, hi_t = t - dt, t + dt
        lo_t, hi_t = min(lo_t, g_t - epsilon), max(hi_t, g_t + epsilon)

        # Amplitude bounds
        dv = max(abs(v) * frac_v, 20)
        lo_v, hi_v = max(v - dv, min_val), v + dv
        lo_v, hi_v = min(lo_v, g_v - epsilon), max(hi_v, g_v + epsilon)

        # Width bounds
        dw = max(abs(w) * frac_w, 5)
        lo_w, hi_w = max(w - dw, min_val), w + dw
        if min_width is not None:
            lo_w = max(lo_w, min_width)
        if max_width is not None:
            hi_w = min(hi_w, max_width)

        # Ensure lower < upper
        if lo_w >= hi_w:
            lo_w = max(lo_w, min_width if min_width else lo_w)
            hi_w = lo_w + 1.0  # minimal gap
        lo_w, hi_w = min(lo_w, g_w - epsilon), max(hi_w, g_w + epsilon)

        bounds_lower.extend([lo_t, lo_v, lo_w])
        bounds_upper.extend([hi_t, hi_v, hi_w])

    return np.array(bounds_lower), np.array(bounds_upper)

def asymmetric_bounds(guess, bounds_lower=None, bounds_upper=None, model='asymmetric_gaussian'):
    """
    Expand symmetric bounds into asymmetric ones for the asymmetric Gaussian model.
    """
    if model != 'asymmetric_gaussian':
        return guess, bounds_lower, bounds_upper

    n = len(guess) // 3
    g2, bl2, bu2 = [], [], []

    for i in range(n):
        t, v, w = guess[3*i:3*i+3]

        # Slightly offset left/right guesses to break symmetry (not sure if 15% is a good choice, might have to play around here)
        wl = w * 0.85
        wr = w * 1.15

        g2 += [t, v, wl, wr]

        if bounds_lower is not None and bounds_upper is not None:
            lo_t, lo_v, lo_w = bounds_lower[3*i:3*i+3]
            hi_t, hi_v, hi_w = bounds_upper[3*i:3*i+3]

            # Make left and right bounds independent
            bl2 += [lo_t, lo_v, lo_w, w]
            bu2 += [hi_t, hi_v, w, hi_w]
        else:
            bl2, bu2 = None, None

    return g2, bl2, bu2


def line_fit(popt, x_inv, n):
	means= popt[::n]
	#std = popt[2::n]/np.sqrt(2)
	peak_no = np.arange(1,len(means)+1)
	means_edit = means - x_inv[0]
	a, b = np.polyfit(peak_no, means_edit, 1)	
	spear = spearmanr(peak_no, means_edit)
	pearson = pearsonr(peak_no,means_edit)
	return a, b, spear, pearson

def chi_squared(y_obs, y_fit):
	residual = y_obs - y_fit
	chi_squared = np.sum((residual) ** 2 / y_fit)
	return chi_squared

def multi_model_fit(time, counts, guess, bounds_lower=None, bounds_upper=None,
                    yerr=None, model=None):
    """
    Fit Gaussian, Lorentzian, and Asymmetric Gaussian models.
    If 'model' is given, fit ONLY that model.
    Otherwise, compare chi-square and pick the best.
    """

    time = np.asarray(time).reshape(-1)
    counts = np.asarray(counts).reshape(-1)

    # List of all models
    candidates = {
        'gaussian':          (gaussian, 3),
        'lorentzian':        (lorentzian, 3),
        'asymmetric_gaussian': (asymmetric_gaussian, 4),
    }


    # CASE 1 — user forces a specific model
    if model is not None:
        if model not in candidates:
            raise ValueError(f"Model '{model}' not recognised. "
                             f"Choose from {list(candidates.keys())}")

        func, k = candidates[model]

        # Prepare guesses and bounds for the forced model
        p0, bl, bu = asymmetric_bounds(guess, bounds_lower, bounds_upper, model=model)

        # Fit only that model
        if bl is not None and bu is not None:
            popt, pcov = curve_fit(func, time, counts, p0=p0, bounds=(bl, bu), maxfev=60000)
        else:
            popt, pcov = curve_fit(func, time, counts, p0=p0, maxfev=60000)

        fit = func(time, *popt)
        resid = counts - fit
        chi2 = chi_squared(counts, fit)

        print(f"Forced model: {model}, chi^2 = {chi2:.3f}")
        return model, popt, pcov, fit, resid, chi2

    # CASE 2 — finds best model through χ²
    else:
        results = []

        for name, (func, k) in candidates.items():
            try:
                p0, bl, bu = asymmetric_bounds(guess, bounds_lower, bounds_upper, model=name)

                if bl is not None and bu is not None:
                    popt, pcov = curve_fit(func, time, counts, p0=p0, bounds=(bl, bu), maxfev=60000)
                else:
                    popt, pcov = curve_fit(func, time, counts, p0=p0, maxfev=60000)

                fit = func(time, *popt)
                resid = counts - fit
                chi2 = chi_squared(counts, fit)
                results.append((name, popt, pcov, fit, resid, chi2))

            except Exception:
                continue

        if not results:
            raise RuntimeError("All model fits failed.")

        # Pick lowest chi^2
        best = min(results, key=lambda r: r[-1])
        best_model, popt, pcov, fit, resid, chi2 = best

        print(f"Best model: {best_model}, chi^2 = {chi2:.3f}")
        return best_model, popt, pcov, fit, resid, chi2


def single_model_fit(time, counts, guess, model, bounds_lower=None, bounds_upper=None, yerr=None):
    """
    Fit a single chosen model (for repeated fits in error_analysis).
    """
    time = np.asarray(time).reshape(-1)
    counts = np.asarray(counts).reshape(-1)

    if model == 'gaussian':
        func = gaussian
    elif model == 'lorentzian':
        func = lorentzian
    elif model == 'asymmetric_gaussian':
        func = asymmetric_gaussian
    else:
        raise ValueError(f"Unknown model: {model}")

    p0, bl, bu = asymmetric_bounds(guess, bounds_lower, bounds_upper, model=model)

    if bl is not None and bu is not None:
        popt, pcov = curve_fit(func, time, counts, p0=p0, bounds=(bl, bu), maxfev=60000)
    else:
        popt, pcov = curve_fit(func, time, counts, p0=p0, maxfev=60000)

    fit = func(time, *popt)
    resid = counts - fit
    chi2 = chi_squared(counts, fit)
    return popt, pcov, fit, resid, chi2

def decompose_components(time, popt, model_name):
    """
    Given the fitted parameters and model name,
    return individual component fits for plotting.
    """
    if model_name == 'gaussian':
        func = gaussian
        n_params = 3
    elif model_name == 'lorentzian':
        func = lorentzian
        n_params = 3
    elif model_name == 'asymmetric_gaussian':
        func = asymmetric_gaussian
        n_params = 4
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_components = len(popt) // n_params
    fit_components = np.zeros((n_components, len(time)))

    for i in range(n_components):
        params = popt[i * n_params:(i + 1) * n_params]
        fit_components[i] = func(time, *params)
    return fit_components