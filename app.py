
import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")
st.title("Sequential First-Order Kinetic Fitting Tool")

st.markdown(r'''
This tool fits experimental HDX data to a sequential first-order kinetic model:

- D₀ → D₁ → D₂

The model assumes two consecutive irreversible first-order reactions, with rates $k_1$ and $k_2$.
The analytical solutions for the fractions of D₀, D₁, and D₂ at time $t$ are:

$$
D_0(t) = 1 - D_1(t) - D_2(t)
$$

$$
D_1(t) = D_\text{max} \cdot \frac{k_1}{k_2 - k_1} (e^{-k_1 t} - e^{-k_2 t})
$$

$$
D_2(t) = D_\text{max} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1}\right]
$$

This model captures irreversible deuterium incorporation across 3 species. The `curve_fit` function from SciPy with the Trust Region Reflective (TRF) method is used to estimate $k_1$ and $k_2$, allowing non-negative bounds and robust behavior for correlated parameters.
''')

with st.expander("📜 Show fitting code logic"):
    st.code(
        '''
from scipy.optimize import curve_fit
def fit_kinetic_data(...):
    ...
    def combined_model(t_dummy, k1, k2):
        d0, d1, d2 = sequential_first_order_model(time_data, k1, k2, max_deut)
        return np.concatenate([d0, d1, d2])
    y_obs = np.concatenate([d0_data, d1_data, d2_data])
    popt, _ = curve_fit(combined_model, ...)
    return k1, k2, d0_fit, d1_fit, d2_fit
        ''', language="python"
    )

col1, col2 = st.columns([1.3, 2])

with col1:
    st.header("Upload and Fit Data")
    initial_k1 = st.number_input("Initial guess for k₁ (~0.01)", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k₂ (~½ of k₁)", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)

    uploaded_file = st.file_uploader("Upload your kinetic data (Excel)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        using_example = False
    else:
        df = pd.DataFrame({
            'time': np.linspace(0, 200, 10),
            'd0': np.linspace(1, 0.1, 10),
            'd1': np.linspace(0, 0.5, 10),
            'd2': np.linspace(0, 0.4, 10)
        })
        using_example = True
        st.info("No file uploaded. Example dataset loaded.")

    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0 = df['d0'].values
        d1 = df['d1'].values
        d2 = df['d2'].values

        min_d1 = 1.0 - max_deut

        result = fit_kinetic_data(time, d0, d1, d2,
                                   initial_k1=initial_k1,
                                   initial_k2=initial_k2,
                                   max_deut=max_deut,
                                   min_d1=min_d1)
        if result['success']:
            st.success("Model fit successfully!")
            st.metric("k₁", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
            st.metric("k₂", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
            st.metric("R²", f"{result['r_squared']:.5f}")
        else:
            st.error(result['message'])
    else:
        st.error("File must contain: time, d0, d1, d2")

with col2:
    if 'result' in locals() and result['success']:
        st.subheader("Observed vs Fitted Curves")
        d0_fit = 1.0 - result['d1_fit'] - result['d2_fit']
        fitted_df = pd.DataFrame({
            'time': time,
            'D0 Observed': d0,
            'D1 Observed': d1,
            'D2 Observed': d2,
            'D0 Fit': d0_fit,
            'D1 Fit': result['d1_fit'],
            'D2 Fit': result['d2_fit']
        })
        st.line_chart(fitted_df.set_index('time'))
