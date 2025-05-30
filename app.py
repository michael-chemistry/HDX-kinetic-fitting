
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")
st.title("Sequential First-Order Kinetic Fitting Tool")

st.markdown(r'''
This tool fits experimental HDX data to a sequential first-order kinetic model:

- D₀ → D₁ → D₂

The model assumes two consecutive irreversible first-order reactions, with rates $k_1$ and $k_2$.

### Model Equations

$$
D_0(t) = 1 - D_1(t) - D_2(t)
$$

$$
D_1(t) = D_{\text{max}} \cdot \frac{k_1}{k_2 - k_1} \left(e^{-k_1 t} - e^{-k_2 t}\right)
$$

$$
D_2(t) = D_{\text{max}} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1}\right]
$$

This model was chosen because it captures the kinetic progression between unmodified, singly modified, and doubly modified species without requiring assumptions of reversibility, which aligns with the irreversible nature of deuterium exchange in many experimental systems.

The optimization is performed using SciPy’s `curve_fit` with the **Trust Region Reflective (TRF)** algorithm, chosen for its support of bounds and robust handling of correlated parameters.
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
    st.download_button("Download CSV", example_data.to_csv(index=False), "example_kinetics.csv", "text/csv")

uploaded_file = st.file_uploader("Upload your kinetic data (Excel format)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0 = df['d0'].values
        d1 = df['d1'].values
        d2 = df['d2'].values

        min_d1 = 1.0 - max_deut  # Ensure conservation

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

            col1, col2 = st.columns(2)

            with col1:
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=time, y=d0, mode='markers', name='D0 Observed'))
                fig_main.add_trace(go.Scatter(x=time, y=d1, mode='markers', name='D1 Observed'))
                fig_main.add_trace(go.Scatter(x=time, y=d2, mode='markers', name='D2 Observed'))
                fig_main.add_trace(go.Scatter(x=time, y=d0_fit, mode='lines', name='D0 Fit'))
                fig_main.add_trace(go.Scatter(x=time, y=result['d1_fit'], mode='lines', name='D1 Fit'))
                fig_main.add_trace(go.Scatter(x=time, y=result['d2_fit'], mode='lines', name='D2 Fit'))
                fig_main.update_layout(title="Observed vs Fitted", xaxis_title="Time", yaxis_title="Fraction")
                st.plotly_chart(fig_main, use_container_width=True)

            with col2:
                residual_len = len(result['residuals']) // 3
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(x=time, y=result['residuals'][:residual_len], mode='lines+markers', name='D0 Residuals'))
                fig_resid.add_trace(go.Scatter(x=time, y=result['residuals'][residual_len:2*residual_len], mode='lines+markers', name='D1 Residuals'))
                fig_resid.add_trace(go.Scatter(x=time, y=result['residuals'][2*residual_len:], mode='lines+markers', name='D2 Residuals'))
                fig_resid.update_layout(title="Residuals", xaxis_title="Time", yaxis_title="Residual")
                st.plotly_chart(fig_resid, use_container_width=True)

            with st.expander("🔍 Click to view fitting function source code"):
                from inspect import getsource
                from Sequential_Kinetic_Fit import fit_kinetic_data
                st.code(getsource(fit_kinetic_data), language="python")
        else:
            st.error(result['message'])
    else:
        st.error("Your file must include columns: time, d0, d1, and d2")
else:
    st.info("No file uploaded. Showing example data preview.")
    st.dataframe(example_data)
