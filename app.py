import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model
import inspect

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

This model was chosen because it captures the kinetic progression between unmodified, singly modified, and doubly modified species without requiring assumptions of reversibility, which aligns with the irreversible nature of deuterium exchange in many experimental systems.
''')

with st.sidebar:
    st.header("Fitting Parameters")
    initial_k1 = st.number_input("Initial guess for k₁ (recommended: ~0.01)", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k₂ (recommended: ~0.005)", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)

    st.subheader("Download Example File")
    example_data = pd.DataFrame({
        'time': np.linspace(0, 200, 10),
        'd0': np.linspace(1, 0.1, 10),
        'd1': np.linspace(0, 0.5, 10),
        'd2': np.linspace(0, 0.4, 10)
    })
    st.download_button("Download CSV", example_data.to_csv(index=False), file_name="example_kinetics.csv")

uploaded_file = st.file_uploader("Upload your kinetic data (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = example_data
    st.info("Using example data. Upload your own Excel file to override.")

if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
    time = df['time'].values
    d0 = df['d0'].values
    d1 = df['d1'].values
    d2 = df['d2'].values

    min_d1 = 1.0 - max_deut

    result = fit_kinetic_data(time, d0, d1, d2,
                              initial_k1=initial_k1,
                              initial_k2=initial_k2,
                              max_deut=max_deut)

    if result['success']:
        st.success("Model fit successfully!")
        st.metric("k₁", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
        st.metric("k₂", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
        st.metric("R²", f"{result['r_squared']:.5f}")

        d0_fit = 1.0 - result['d1_fit'] - result['d2_fit']

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(time, d0, 'o', label='D₀ Observed')
        axs[0].plot(time, d1, 'o', label='D₁ Observed')
        axs[0].plot(time, d2, 'o', label='D₂ Observed')
        axs[0].plot(time, d0_fit, '-', label='D₀ Fit')
        axs[0].plot(time, result['d1_fit'], '-', label='D₁ Fit')
        axs[0].plot(time, result['d2_fit'], '-', label='D₂ Fit')
        axs[0].legend()
        axs[0].set_title("Observed vs Fitted")

        axs[1].plot(time.tolist() * 3, result['residuals'], 'r.')
        axs[1].axhline(0, color='gray', linestyle='--')
        axs[1].set_title("Residuals")

        st.pyplot(fig)
    else:
        st.error(result['message'])
else:
    st.error("Your file must include columns: time, d0, d1, and d2")

with st.expander("📜 Click to show the kinetic fitting function code"):
    st.code(inspect.getsource(fit_kinetic_data), language="python")
