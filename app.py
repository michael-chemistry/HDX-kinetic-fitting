
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model
from Single_Kinetic_Fit import fit_single_order_data, single_first_order_model

st.set_page_config(page_title="Kinetic Fitting Tool", layout="wide")
st.title("Kinetic Fitting Tool")

model_choice = st.sidebar.selectbox("Select Kinetic Model", ["Sequential First-Order", "Single First-Order"])

st.markdown(r"""
This tool fits experimental HDX data to kinetic models.

**Sequential First-Order:**
- D0 → D1 → D2

**Single First-Order:**
- D0 → D1

Sequential assumes two exchangeable protons. Single is for one exchange site.
""")

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

            colors = {'d0': 'blue', 'd1': 'orange', 'd2': 'green'}
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))

            axs[0].plot(time, d0, 'o', label='D0 Obs', color=colors['d0'])
            axs[0].plot(time, d1, 'o', label='D1 Obs', color=colors['d1'])
            axs[0].plot(time, d2, 'o', label='D2 Obs', color=colors['d2'])
            axs[0].plot(time, result['d0_fit'], '-', label='D0 Fit', color=colors['d0'])
            axs[0].plot(time, result['d1_fit'], '-', label='D1 Fit', color=colors['d1'])
            axs[0].plot(time, result['d2_fit'], '-', label='D2 Fit', color=colors['d2'])
            axs[0].legend()
            axs[0].set_title("Observed vs Fit")

            axs[1].scatter(time, d0 - result['d0_fit'], label='D0 Resid', color=colors['d0'])
            axs[1].scatter(time, d1 - result['d1_fit'], label='D1 Resid', color=colors['d1'])
            axs[1].scatter(time, d2 - result['d2_fit'], label='D2 Resid', color=colors['d2'])
            axs[1].axhline(0, color='gray', linestyle='--')
            axs[1].legend()
            axs[1].set_title("Residuals")
            st.pyplot(fig)
        else:
            st.error(result['message'])
    else:
        st.error("File must include columns: time, d0, d1, d2")
else:
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
            axs[1].axhline(0, color='gray', linestyle='--')
            axs[1].legend()
            axs[1].set_title("Residuals")
            st.pyplot(fig)
        else:
            st.error(result['message'])
    else:
        st.error("File must include columns: time, d1")

if model_choice == "Sequential First-Order":
    with st.expander("Click to show the kinetic fitting function code"):
        st.code(inspect.getsource(fit_kinetic_data), language="python")
    with st.expander("Click to show the kinetic model equations"):
        st.code(inspect.getsource(sequential_first_order_model), language="python")
else:
    with st.expander("Click to show the kinetic fitting function code"):
        st.code(inspect.getsource(fit_single_order_data), language="python")
    with st.expander("Click to show the kinetic model equations"):
        st.code(inspect.getsource(single_first_order_model), language="python")
