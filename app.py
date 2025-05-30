
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from Sequential_Kinetic_Fit import fit_kinetic_data

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")
st.title("Sequential First-Order Kinetic Fitting Tool")

# === Model Description and Instructions ===
with st.sidebar:
    st.header("Fitting Parameters")
    initial_k1 = st.number_input("Initial guess for k₁", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k₂", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)

    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload Excel file (must have 'time', 'd0', 'd1', 'd2')", type=["xlsx"])

    st.subheader("Download Example")
    example_df = pd.DataFrame({
        'time': np.linspace(0, 200, 10),
        'd0': np.linspace(1, 0.1, 10),
        'd1': np.linspace(0, 0.5, 10),
        'd2': np.linspace(0, 0.4, 10)
    })
    st.download_button("Download Example CSV", example_df.to_csv(index=False), file_name="example_kinetics.csv")

# === Main Panel Content ===
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    ## 📘 Model Overview

    This tool fits experimental HDX data to a **sequential irreversible first-order kinetic model**:

    **D₀ → D₁ → D₂**

    ### Equations:
    - **D₀(t)** = 1 − D₁(t) − D₂(t)  
    - **D₁(t)** = Dₘₐₓ × k₁⁄(k₂−k₁) × (e^(−k₁t) − e^(−k₂t))  
    - **D₂(t)** = Dₘₐₓ × [1 − (k₂·e^(−k₁t) − k₁·e^(−k₂t))⁄(k₂−k₁)]

    The model is solved analytically and fit using nonlinear least-squares via **SciPy's `curve_fit`**.

    ### Optimizer Justification
    We use the **Trust Region Reflective (TRF)** method because it:
    - Enforces **non-negative rate constants**
    - Handles **correlated parameters** (k₁ and k₂)
    - Offers stable convergence in mildly nonlinear biological models

    """, unsafe_allow_html=True)

    with st.expander("🔍 Show fitting code transparency"):
        from Sequential_Kinetic_Fit import fit_kinetic_data
        import inspect
        st.code(inspect.getsource(fit_kinetic_data), language="python")

with col2:
    # Load either uploaded or example file
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = example_df

    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0 = df['d0'].values
        d1 = df['d1'].values
        d2 = df['d2'].values

        min_d1 = 1.0 - max_deut

        result = fit_kinetic_data(time, d0, d1, d2, initial_k1, initial_k2, max_deut)
        if result['success']:
            st.success("✅ Model fit successful")
            st.metric("k₁", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
            st.metric("k₂", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
            st.metric("R²", f"{result['r_squared']:.5f}")

            d0_fit = 1 - result['d1_fit'] - result['d2_fit']
            residuals = result['residuals']

            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=time, y=d0, mode='markers', name='D₀ Obs'))
            fig_main.add_trace(go.Scatter(x=time, y=d1, mode='markers', name='D₁ Obs'))
            fig_main.add_trace(go.Scatter(x=time, y=d2, mode='markers', name='D₂ Obs'))
            fig_main.add_trace(go.Scatter(x=time, y=d0_fit, mode='lines', name='D₀ Fit'))
            fig_main.add_trace(go.Scatter(x=time, y=result['d1_fit'], mode='lines', name='D₁ Fit'))
            fig_main.add_trace(go.Scatter(x=time, y=result['d2_fit'], mode='lines', name='D₂ Fit'))
            fig_main.update_layout(title="Observed vs Fitted", xaxis_title="Time", yaxis_title="Fraction"))

            st.plotly_chart(fig_main, use_container_width=True)

            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(y=residuals, mode='lines+markers', name='Residuals'))
            fig_resid.update_layout(title="Residuals", xaxis_title="Index", yaxis_title="Residual"))
            st.plotly_chart(fig_resid, use_container_width=True)
        else:
            st.error(result['message'])
    else:
        st.error("Your file must include columns: time, d0, d1, and d2.")

