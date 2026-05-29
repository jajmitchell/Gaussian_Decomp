from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import preprocessing
import numpy as np

def rescale(time_arr,counts_arr):
	scaler_x = preprocessing.StandardScaler().fit(time_arr.reshape(-1,1))
	X_scaled = scaler_x.transform(time_arr.reshape(-1,1))
	scaler_y = preprocessing.StandardScaler().fit(counts_arr.reshape(-1,1))
	y_scaled = scaler_y.transform(counts_arr.reshape(-1,1))
	return X_scaled, y_scaled, scaler_x, scaler_y

def grid_search(X_scaled,y_scaled, params):
	gp = GaussianProcessRegressor()
	grid=GridSearchCV(estimator=gp, param_grid=params, scoring='neg_mean_squared_error').fit(X_scaled.reshape(-1,1), y_scaled.reshape(-1,1))
	return grid

def train_set(x_scaled, y_scaled):
	rng = np.random.RandomState(1)
	training_indices = rng.choice(np.arange(y_scaled.size), size=int(0.8*(y_scaled.size)), replace=False)
	X_train, y_train = x_scaled[training_indices].reshape(-1,1), y_scaled[training_indices].reshape(-1,1) 
	return X_train, y_train


def gp_fit(x_train,y_train, scaler_x, scaler_y, X_scaled, alpha, A, l):
	kernel = ConstantKernel(A, constant_value_bounds="fixed") * RBF(length_scale=l,length_scale_bounds="fixed") 
	gaussian_process = GaussianProcessRegressor(kernel=kernel,alpha=alpha,random_state=2)
	gaussian_process.fit(x_train,y_train)
	gaussian_process.kernel_ 

	resampled_arr=np.arange(np.min(X_scaled),np.max(X_scaled),0.01)

	mean_prediction, std_prediction = gaussian_process.predict(resampled_arr.reshape(-1,1), return_std=True)
	
	mean_prediction_inv = scaler_y.inverse_transform(mean_prediction.reshape(-1,1))
	x_inv = scaler_x.inverse_transform(resampled_arr.reshape(-1,1))
	std_inv = (np.max(mean_prediction_inv)/np.max(mean_prediction))*std_prediction.reshape(-1,1)

	return x_inv, mean_prediction_inv, std_inv

def gp_regression(times, counts, alpha, A, l):
    """
    Run full Gaussian Process Regression on unscaled time/counts data.
    Returns smoothed data and 1σ errors.
    """

    # --- 1. Rescale ---
    X_scaled, y_scaled, scaler_x, scaler_y = rescale(np.array(times), np.array(counts))

    # --- 2. Split into training set ---
    X_train, y_train = train_set(X_scaled, y_scaled)

    # --- 3. Fit GP and get predictions ---
    x_inv, smoothed, errors = gp_fit(X_train, y_train, scaler_x, scaler_y, X_scaled, alpha, A, l)

    return x_inv, smoothed, errors

def running_mean(times, counts, yerr, window_size):
    N = window_size
    counts_smoothed, errors_smoothed, times_smoothed = [], [], []

    for i in range(len(counts) - N + 1):
        # Window slices
        c_win = counts[i : i + N]
        e_win = yerr[i : i + N]
        t_win = times[i : i + N]

        # Inverse-variance weights
        weight = 1.0 / (e_win ** 2)
        W = np.sum(weight)

        # Weighted mean for counts
        m = np.sum(weight * c_win) / W

        # Propagated error of the weighted mean
        err = np.sqrt(1.0 / W)

        # Centre time of the window
        t = t_win[N // 2]

        counts_smoothed.append(m)
        errors_smoothed.append(err)
        times_smoothed.append(t)

    return np.array(times_smoothed), np.array(counts_smoothed), np.array(errors_smoothed)

def running_mean_chi2(times, counts, yerr,
                      Nmin=2, Nmax=10):
    times_s, counts_s, errors_s = [], [], []

    Ntot = len(counts)

    for i in range(Ntot):
        best_score = np.inf
        best = None

        for N in range(Nmin, Nmax + 1):
            half = N // 2
            left = i - half
            right = left + N

            if left < 0 or right > Ntot:
                continue

            c_win = counts[left:right]
            e_win = yerr[left:right]
            t_win = times[left:right]

            w = 1.0 / e_win**2
            W = np.sum(w)

            mu = np.sum(w * c_win) / W

            chi2 = np.sum((c_win - mu)**2 / e_win**2)
            chi2_red = chi2 / (N - 1)

            score = abs(chi2_red - 1.0)

            if score < best_score:
                best_score = score
                best = (mu, np.sqrt(1.0 / W), t_win[half])

        if best is not None:
            m, err, t = best
            times_s.append(t)
            counts_s.append(m)
            errors_s.append(err)

    return (
        np.array(times_s),
        np.array(counts_s),
        np.array(errors_s),
    )
