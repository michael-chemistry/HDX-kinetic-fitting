import numpy as np

def sequential_first_order_model(t, k1, k2, max_deut=0.95, min_d1=0.05):
    d1 = max_deut * (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    d2 = max_deut * (1 - ((k2 * np.exp(-k1 * t) - k1 * np.exp(-k2 * t)) / (k2 - k1)))
    d1 = np.maximum(d1, min_d1)
    d2 = np.minimum(d2, max_deut)
    d0 = np.maximum(0, 1 - d1 - d2)
    total = d0 + d1 + d2
    d0 /= total
    d1 /= total
    d2 /= total
    return d0, d1, d2

from scipy.optimize import curve_fit

def fit_kinetic_data(time_data, d0_data, d1_data, d2_data, initial_k1=0.01, initial_k2=0.005,
                     max_deut=0.95, min_d1=0.05):
    initial_params = [initial_k1, initial_k2]
    param_bounds = ([0, 0], [np.inf, np.inf])

    def custom_objective_func(dummy_time, k1, k2):
        d0_model, d1_model, d2_model = sequential_first_order_model(time_data, k1, k2, max_deut, min_d1)
        return np.concatenate([d0_model, d1_model, d2_model])

    y_obs = np.concatenate([d0_data, d1_data, d2_data])
    t_dummy = np.linspace(0, 1, len(y_obs))

    popt, pcov = curve_fit(
        custom_objective_func,
        t_dummy,
        y_obs,
        p0=initial_params,
        bounds=param_bounds,
        method='trf',
        maxfev=10000,
        ftol=1e-10,
        xtol=1e-10
    )

    k1_opt, k2_opt = popt
    k1_error, k2_error = np.sqrt(np.diag(pcov))
    d0_fit, d1_fit, d2_fit = sequential_first_order_model(time_data, k1_opt, k2_opt, max_deut, min_d1)
    y_fit_combined = np.concatenate([d0_fit, d1_fit, d2_fit])
    ss_total = np.sum((y_obs - np.mean(y_obs)) ** 2)
    ss_residual = np.sum((y_obs - y_fit_combined) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    return {
        'k1': k1_opt,
        'k2': k2_opt,
        'k1_error': k1_error,
        'k2_error': k2_error,
        'r_squared': r_squared,
        'd0_fit': d0_fit,
        'd1_fit': d1_fit,
        'd2_fit': d2_fit,
        'success': True
    }