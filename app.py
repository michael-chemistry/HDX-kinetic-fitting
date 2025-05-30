
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
    st.markdown(r"""
**Mechanism**  
D0 → D1

**Equations**  
$$
D_1(t) = D_{\text{max}} \cdot \left(1 - e^{-k_1 t} \right)
$$  
$$
D_0(t) = 1 - D_1(t)
$$

**Fitting Method: curve_fit**  
- Ideal for simple, one-parameter models  
- Provides fast and accurate results  
- Supports parameter bounds
""")

with col2:
    st.subheader("Sequential First-Order")
    st.markdown(r"""
**Mechanism**  
D0 → D1 → D2

**Equations**  
$$
D_1(t) = D_{\text{max}} \cdot \frac{k_1}{k_2 - k_1}(e^{-k_1 t} - e^{-k_2 t})
$$  
$$
D_2(t) = D_{\text{max}} \cdot \left[1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1} \right]
$$  
$$
D_0(t) = 1 - D_1(t) - D_2(t)
$$

**Fitting Method: Trust Region Reflective (TRF)**  
- Handles multi-parameter models  
- Allows bound constraints  
- Robust to parameter correlation
""")
