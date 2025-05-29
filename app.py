import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="centered")
st.title("Sequential First-Order Kinetic Fitting Tool")

st.markdown("""
This tool fits experimental HDX data to a sequential first-order kinetic model:

- D₀ → D₁ → D₂

The model assumes two consecutive irreversible first-order reactions, with rates $k_1$ and $k_2$.
The analytical solutions for the fractions of D₀, D₁, and D₂ at time $t$ are:

$$
D_1(t) = D_\text{max} \cdot \frac{k_1}{k_2 - k_1} (e^{-k_1 t} - e^{-k_2 t})
$$

$$
D_2(t) = D_\text{max} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1}\right]
$$

$$
D_0(t) = 1 - D_1(t) - D_2(t)
$$

This model was chosen because it captures the kinetic progression between unmodified, singly modified, and doubly modified species without requiring assumptions of reversibility, which aligns with the irreversible nature of deuterium exchange in many experimental systems.

The optimization is performed using the `curve_fit` function from SciPy with the Trust Region Reflective (TRF) algorithm.
This method was selected because it supports bound-constrained nonlinear least squares optimization, which is important for enforcing the physically meaningful constraint that rate constants must be non-negative. It also performs robustly for problems with moderate residuals and correlated parameters like $k_1$ and $k_2$.

---
""")

with st.expander("📜 Click to show/hide the source code"):
    with open("app.py", "r") as f:
        st.code(f.read(), language="python")

st.markdown("""
### Step-by-Step Instructions
1. Download the example CSV to understand the required format.
2. Upload your experimental data file (Excel format).
3. Optionally adjust the initial guesses and fitting constraints.
4. View optimized parameters ($k_1$, $k_2$, $R^2$) and fitted curves.

---
""")

with st.sidebar:
    st.header("Fitting Parameters")
    initial_k1 = st.number_input("Initial guess for k₁ (recommended: ~0.01)", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k₂ (recommended: ~0.005, or ~½ of k₁)", value=0.005, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)
    min_d1 = st.slider("Minimum D₁ Value", 0.0, 0.1, 0.05, 0.01)

    st.subheader("Download Example File")
    example_data = pd.DataFrame({
        'time': np.linspace(0, 200, 10),
        'd0': np.linspace(1, 0.1, 10),
        'd1': np.linspace(0, 0.5, 10),
        'd2': np.linspace(0, 0.4, 10)
    })
    csv = example_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="example_kinetics.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your kinetic data (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0 = df['d0'].values
        d1 = df['d1'].values
        d2 = df['d2'].values

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

            st.subheader("Observed vs Fitted Values")
            fitted_df = pd.DataFrame({
                'time': time,
                'D0 Observed': d0,
                'D1 Observed': d1,
                'D2 Observed': d2,
                'D0 Fit': result['d0_fit'],
                'D1 Fit': result['d1_fit'],
                'D2 Fit': result['d2_fit']
            })
            st.dataframe(fitted_df, use_container_width=True)

            st.line_chart(fitted_df.set_index('time'))
        else:
            st.error(result['message'])
    else:
        st.error("Your file must include columns: time, d0, d1, and d2")
