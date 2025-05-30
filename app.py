
import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")

# Sidebar theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""<style>body { background-color: #111; color: #eee; }</style>""", unsafe_allow_html=True)

# Page title and description
st.title("Sequential First-Order Kinetic Fitting Tool")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown(r"""
### About the Model
This tool fits experimental HDX data to a sequential irreversible first-order model:

**Reaction Pathway:**  
$D_0 \rightarrow D_1 \rightarrow D_2$

**Equations:**  
$D_0(t) = 1 - D_1(t) - D_2(t)$  
$D_1(t) = D_{\text{max}} \cdot \frac{k_1}{k_2 - k_1} (e^{-k_1 t} - e^{-k_2 t})$  
$D_2(t) = D_{\text{max}} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1}\right]$

**Why this model?**  
It captures irreversible sequential exchange events observed in many HDX-MS studies and uses a nonlinear least-squares approach with bounds, solved using SciPy's Trust Region Reflective (TRF) algorithm.

---

### Instructions
1. Download the example data file to understand the expected format.
2. Upload your own Excel file with columns: `time`, `d0`, `d1`, and `d2`.
3. Adjust initial parameter guesses in the sidebar.
4. View the fit and optimized parameters.
""")

with col2:
    st.subheader("Upload and Plot")
    uploaded_file = st.file_uploader("Upload your kinetic data (Excel format)", type=["xlsx"])

    st.sidebar.header("Fitting Settings")    
    initial_k1 = st.sidebar.number_input("Initial k₁", value=0.01, format="%.5f")    
    initial_k2 = st.sidebar.number_input("Initial k₂ (≈½ of k₁)", value=0.005, format="%.5f")    
    max_deut = st.sidebar.slider("Max Deuterium Incorporation (Dmax)", 0.0, 1.0, 0.95, 0.01)

    # Load example or uploaded data
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.DataFrame({
            'time': np.linspace(0, 200, 10),
            'd0': np.linspace(1, 0.2, 10),
            'd1': np.linspace(0, 0.4, 10),
            'd2': np.linspace(0, 0.4, 10)
        })
        st.info("Showing example data — upload your own to fit.")

    if all(col in df.columns for col in ['time', 'd0', 'd1', 'd2']):
        time = df['time'].values
        d0 = df['d0'].values
        d1 = df['d1'].values
        d2 = df['d2'].values

        result = fit_kinetic_data(time, d0, d1, d2,
                                   initial_k1=initial_k1,
                                   initial_k2=initial_k2,
                                   max_deut=max_deut)

        if result['success']:
            st.success("Model fit successfully!")
            st.metric("k₁", f"{result['k1']:.5f} ± {result['k1_error']:.5f}")
            st.metric("k₂", f"{result['k2']:.5f} ± {result['k2_error']:.5f}")
            st.metric("R²", f"{result['r_squared']:.5f}")

            d0_fit = result['d0_fit']
            d1_fit = result['d1_fit']
            d2_fit = result['d2_fit']

            fit_df = pd.DataFrame({
                'time': time,
                'D0 Observed': d0,
                'D1 Observed': d1,
                'D2 Observed': d2,
                'D0 Fit': d0_fit,
                'D1 Fit': d1_fit,
                'D2 Fit': d2_fit
            })
            st.line_chart(fit_df.set_index('time'))
            st.dataframe(fit_df, use_container_width=True)
        else:
            st.error(result['message'])
    else:
        st.warning("Ensure your file includes columns: time, d0, d1, d2")

    # Download example file
    st.sidebar.subheader("Download Example File")
    example_csv = df.to_csv(index=False)
    b64 = base64.b64encode(example_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="example_kinetics.csv">Download Example CSV</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
