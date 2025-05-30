
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model
from Single_Kinetic_Fit import fit_single_order_data, single_first_order_model

st.set_page_config(page_title="Kinetic Fitting Tool", layout="wide")
st.title("Kinetic Fitting Tool")

# Model descriptions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Single First-Order")
    st.markdown("**Mechanism**")
    st.latex(r"D_0 ightarrow D_1")

    st.markdown("**Equations**")
    st.latex(r"D_1(t) = D_{	ext{max}} \cdot \left(1 - e^{-k_1 t} ight)")
    st.latex(r"D_0(t) = 1 - D_1(t)")

    st.markdown("**Fitting Method: curve_fit**")
    st.markdown("- Ideal for simple, one-parameter models")
    st.markdown("- Provides fast and accurate results")
    st.markdown("- Supports parameter bounds")

with col2:
    st.subheader("Sequential First-Order")
    st.markdown("**Mechanism**")
    st.latex(r"D_0 ightarrow D_1 ightarrow D_2")

    st.markdown("**Equations**")
    st.latex(r"D_1(t) = D_{	ext{max}} \cdot rac{k_1}{k_2 - k_1}(e^{-k_1 t} - e^{-k_2 t})")
    st.latex(r"D_2(t) = D_{	ext{max}} \cdot \left[1 - rac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1} ight]")
    st.latex(r"D_0(t) = 1 - D_1(t) - D_2(t)")

    st.markdown("**Fitting Method: Trust Region Reflective (TRF)**")
    st.markdown("- Handles multi-parameter models")
    st.markdown("- Allows bound constraints")
    st.markdown("- Robust to parameter correlation")

# Model selection
model_choice = st.sidebar.selectbox("Select Kinetic Model", ["Sequential First-Order", "Single First-Order"])

# Parameters
with st.sidebar:
    st.header("Fitting Parameters")
    initial_k1 = st.number_input("Initial guess for k1", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k2 (Sequential only)", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)

    st.subheader("Download Example File")
    if model_choice == "Sequential First-Order":
        example_data = pd.DataFrame({
            'time': np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270]),
            'd0': np.array([1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]),
            'd1': np.array([0.0, 0.08, 0.18, 0.25, 0.28, 0.28, 0.25, 0.23, 0.20, 0.18]),
            'd2': np.array([0.0, 0.02, 0.07, 0.15, 0.22, 0.32, 0.40, 0.47, 0.55, 0.62]),
        })
        st.download_button("Download CSV", example_data.to_csv(index=False), file_name="example_kinetics.csv")
    else:
        example_data = pd.DataFrame({
            'time': np.array([0, 30, 60, 90, 120, 150, 180]),
            'd1': np.array([0.0, 0.1, 0.25, 0.45, 0.65, 0.8, 0.9])
        })
        st.download_button("Download CSV", example_data.to_csv(index=False), file_name="example_single_order.csv")

uploaded_file = st.file_uploader("Upload your kinetic data (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
else:
    df = example_data
    st.info("Using example data. Upload your own file to override.")

if model_choice == "Sequential First-Order":
    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0, d1, d2 = df['d0'].values, df['d1'].values, df['d2'].values
        result = fit_kinetic_data(time, d0, d1, d2, initial_k1, initial_k2, max_deut)
        if result['success']:
            st.success("Model fit successfully!")
            col1, col2, col3 = st.columns(3)
            col1.metric("k1", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
            col2.metric("k2", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
            col3.metric("R²", f"{result['r_squared']:.5f}")
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            axs[0].plot(time, d0, 'o', label='D0 Obs', color='blue')
            axs[0].plot(time, d1, 'o', label='D1 Obs', color='orange')
            axs[0].plot(time, d2, 'o', label='D2 Obs', color='green')
            axs[0].plot(time, result['d0_fit'], '-', label='D0 Fit', color='blue')
            axs[0].plot(time, result['d1_fit'], '-', label='D1 Fit', color='orange')
            axs[0].plot(time, result['d2_fit'], '-', label='D2 Fit', color='green')
            axs[0].legend()
            axs[0].set_title("Observed vs Fit")
            axs[1].scatter(time, d0 - result['d0_fit'], label='D0 Residual', color='blue')
            axs[1].scatter(time, d1 - result['d1_fit'], label='D1 Residual', color='orange')
            axs[1].scatter(time, d2 - result['d2_fit'], label='D2 Residual', color='green')
            axs[1].axhline(0, color='gray', linestyle='--')
            axs[1].legend()
            axs[1].set_title("Residuals")
            st.pyplot(fig)
        else:
            st.error(result['message'])
    else:
        st.error("File must include columns: time, d0, d1, d2")

if model_choice == "Single First-Order":
    if all(col in df.columns for col in ['time', 'd1']):
        time = df['time'].values
        d1 = df['d1'].values
        result = fit_single_order_data(time, d1, initial_k1, max_deut)
        if result['success']:
            st.success("Model fit successfully!")
            col1, col2 = st.columns(2)
            col1.metric("k1", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
            col2.metric("R²", f"{result['r_squared']:.5f}")
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            axs[0].plot(time, d1, 'o', label='D1 Obs', color='orange')
            axs[0].plot(time, result['d1_fit'], '-', label='D1 Fit', color='orange')
            axs[0].plot(time, 1 - d1, 'o', label='D0 Obs', color='blue')
            axs[0].plot(time, result['d0_fit'], '-', label='D0 Fit', color='blue')
            axs[0].legend()
            axs[0].set_title("Observed vs Fit")
            axs[1].scatter(time, d1 - result['d1_fit'], label='D1 Residual', color='orange')
            axs[1].scatter(time, (1 - d1) - result['d0_fit'], label='D0 Residual', color='blue')
            axs[1].axhline(0, color='gray', linestyle='--')
            axs[1].legend()
            axs[1].set_title("Residuals")
            st.pyplot(fig)
        else:
            st.error(result['message'])
    else:
        st.error("File must include columns: time, d1")

if model_choice == "Sequential First-Order":
    with st.expander("Show Sequential Fit Function Code"):
        st.code(inspect.getsource(fit_kinetic_data), language="python")
    with st.expander("Show Sequential Model Equation Code"):
        st.code(inspect.getsource(sequential_first_order_model), language="python")

if model_choice == "Single First-Order":
    with st.expander("Show Single Fit Function Code"):
        st.code(inspect.getsource(fit_single_order_data), language="python")
    with st.expander("Show Single Model Equation Code"):
        st.code(inspect.getsource(single_first_order_model), language="python")
