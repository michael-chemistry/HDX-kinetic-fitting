
import numpy as np
from scipy.optimize import curve_fit


# === Sequential First-Order Model ===
def sequential_first_order_model(t, k1, k2, max_deut=0.95):
    # [Truncated for brevity — full content inserted during execution]
    if abs(k2 - k1) < 1e-10:
        k2 += 1e-10

    d1 = max_deut * (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    d2 = max_deut * (1 - ((k2 * np.exp(-k1 * t) - k1 * np.exp(-k2 * t)) / (k2 - k1)))
    d0 = 1.0 - d1 - d2
    return d0, d1, d2


# === Kinetic Fit Function ===
def fit_kinetic_data(time_data, d0_data, d1_data, d2_data,
                     initial_k1=0.01, initial_k2=0.005, max_deut=0.95):
    initial_params = [initial_k1, initial_k2]
    bounds = ([0, 0], [np.inf, np.inf])

    def combined_model(t_dummy, k1, k2):
        d0, d1, d2 = sequential_first_order_model(time_data, k1, k2, max_deut)
        return np.concatenate([d0, d1, d2])

    y_obs = np.concatenate([d0_data, d1_data, d2_data])
    t_dummy = np.linspace(0, 1, len(y_obs))

    try:
        popt, pcov = curve_fit(
            combined_model, t_dummy, y_obs,
            p0=initial_params, bounds=bounds,
            method='trf', maxfev=10000, ftol=1e-10, xtol=1e-10
        )
        k1, k2 = popt
        perr = np.sqrt(np.diag(pcov))

        d0_fit, d1_fit, d2_fit = sequential_first_order_model(time_data, k1, k2, max_deut)

        y_fit = np.concatenate([d0_fit, d1_fit, d2_fit])
        residuals = y_obs - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'k1': k1,
            'k2': k2,
            'k1_error': perr[0],
            'k2_error': perr[1],
            'd0_fit': d0_fit,
            'd1_fit': d1_fit,
            'd2_fit': d2_fit,
            'residuals': residuals,
            'r_squared': r_squared,
            'success': True,
            'message': 'Fit successful'
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }
