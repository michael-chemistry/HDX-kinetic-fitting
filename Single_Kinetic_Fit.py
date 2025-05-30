
import numpy as np
from scipy.optimize import curve_fit

def single_first_order_model(t, k1, max_deut=0.95):
    d1 = max_deut * (1 - np.exp(-k1 * t))
    d0 = 1.0 - d1
    return d0, d1

def fit_single_order_data(time_data, d1_data, initial_k1=0.01, max_deut=0.95):
    def model_wrapper(t, k1):
        return max_deut * (1 - np.exp(-k1 * t))

    try:
        popt, pcov = curve_fit(model_wrapper, time_data, d1_data, p0=[initial_k1], bounds=(0, np.inf))
        k1 = popt[0]
        perr = np.sqrt(np.diag(pcov))
        d1_fit = model_wrapper(time_data, k1)
        d0_fit = 1.0 - d1_fit

        ss_res = np.sum((d1_data - d1_fit) ** 2)
        ss_tot = np.sum((d1_data - np.mean(d1_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'k1': k1,
            'k1_error': perr[0],
            'd0_fit': d0_fit,
            'd1_fit': d1_fit,
            'r_squared': r_squared,
            'success': True,
            'message': 'Fit successful'
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }
