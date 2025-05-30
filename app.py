# Streamlit App for Sequential Kinetic Fitting
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")
st.title("Sequential First-Order Kinetic Fitting Tool")

st.markdown(r"""
This tool fits experimental HDX data to a sequential first-order kinetic model:

- $D_0 \rightarrow D_1 \rightarrow D_2$

**Model Equations:**

$$D_0(t) = 1 - D_1(t) - D_2(t)$$

$$D_1(t) = D_{\text{max}} \cdot \frac{k_1}{k_2 - k_1}(e^{-k_1 t} - e^{-k_2 t})$$

$$D_2(t) = D_{\text{max}} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1}\right]$$

**Why this model?**
This model assumes irreversible, sequential first-order transitions between unmodified (D0), singly modified (D1), and doubly modified (D2) species. It captures the progression in HDX labeling over time.

**Why this optimizer?**
Fitting is done using `scipy.optimize.curve_fit` with the Trust Region Reflective (TRF) algorithm. TRF supports bound-constrained nonlinear least squares, ideal for ensuring physically meaningful (non-negative) rate constants. It also offers stability for models with correlated parameters like $k_1$ and $k_2$.
""")

# Example data
example_data = pd.DataFrame({
    'time': [0, 10, 20, 40, 60, 90, 120, 150, 180, 240],
    'd0': [1.0, 0.88, 0.74, 0.53, 0.39, 0.28, 0.20, 0.14, 0.10, 0.06],
    'd1': [0.0, 0.09, 0.17, 0.27, 0.30, 0.29, 0.25, 0.21, 0.16, 0.10],
    'd2': [0.0, 0.03, 0.09, 0.20, 0.31, 0.43, 0.55, 0.65, 0.74, 0.84]
})

with st.sidebar:
    st.header("Fitting Parameters")
    initial_k1 = st.number_input("Initial guess for k₁", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k₂", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)

    st.download_button(
        label="Download Example File",
        data=example_data.to_csv(index=False),
        file_name="example_kinetics.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload your kinetic data (Excel)", type=["xlsx"])

# Load and fit either uploaded or example data
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = example_data

if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
    time = df['time'].values
    d0 = df['d0'].values
    d1 = df['d1'].values
    d2 = df['d2'].values

    min_d1 = 1.0 - max_deut

    result = fit_kinetic_data(
        time, d0, d1, d2,
        initial_k1=initial_k1,
        initial_k2=initial_k2,
        max_deut=max_deut
    )

    if result['success']:
        st.success("Fit successful!")
        st.metric("k₁", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
        st.metric("k₂", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
        st.metric("R²", f"{result['r_squared']:.5f}")

        d0_fit = 1.0 - result['d1_fit'] - result['d2_fit']
        df_plot = pd.DataFrame({
            'Time': time,
            'D0 Observed': d0,
            'D1 Observed': d1,
            'D2 Observed': d2,
            'D0 Fit': d0_fit,
            'D1 Fit': result['d1_fit'],
            'D2 Fit': result['d2_fit'],
            'Residuals': result['residuals']
        })

        # Interactive main plot
        fig_main = go.Figure()
        for species in ['D0', 'D1', 'D2']:
            fig_main.add_trace(go.Scatter(x=df_plot['Time'], y=df_plot[f'{species} Observed'],
                                          mode='markers', name=f'{species} Observed'))
            fig_main.add_trace(go.Scatter(x=df_plot['Time'], y=df_plot[f'{species} Fit'],
                                          mode='lines', name=f'{species} Fit'))
        fig_main.update_layout(title="Observed vs Fitted", xaxis_title="Time", yaxis_title="Fraction")

        # Residuals plot
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=np.tile(time, 3), y=result['residuals'], mode='markers', name="Residuals"))
        fig_res.update_layout(title="Residuals", xaxis_title="Time", yaxis_title="Residual")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_main, use_container_width=True)
        with col2:
            st.plotly_chart(fig_res, use_container_width=True)

        with st.expander("🔍 Show fitting function code"):
            import inspect
            st.code(inspect.getsource(fit_kinetic_data), language='python')
    else:
        st.error(result['message'])
else:
    st.error("Data file must contain columns: time, d0, d1, d2")
