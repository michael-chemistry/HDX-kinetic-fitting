
# ✅ Enhanced version with model selection, including non-sequential model
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from Sequential_Kinetic_Fit import fit_kinetic_data, sequential_first_order_model

st.set_page_config(page_title="Sequential Kinetic Fit", layout="wide")

st.title("First-Order Kinetic Fitting Tool")

st.markdown(r'''
This tool fits experimental HDX data using first-order kinetic models for compounds with **one to three exchangeable protons**.
It provides four options:

- Non-Sequential (D₀ → D₁)
- Single First-Order (D₀ → D₁ with explicit max)
- Two-Step Sequential (D₀ → D₁ → D₂)
- Triple Sequential (D₀ → D₁ → D₂ → D₃)

These models are commonly applied in hydrogen-deuterium exchange (HDX) analysis of small molecules and peptides. The Trust Region Reflective (TRF) algorithm is used for its robustness and support for bounded parameters.
''')

# === Model Definitions ===
def nonsequential_model(t, k1, max_deut=0.95):
    d1 = max_deut * (1 - np.exp(-k1 * t))
    d0 = 1 - d1
    return d0, d1, np.zeros_like(t)

def single_first_order_model(t, k1, max_deut=0.95):
    d1 = max_deut * (1 - np.exp(-k1 * t))
    d0 = 1 - d1
    return d0, d1, np.zeros_like(t)

def triple_first_order_model(t, k1, k2, k3, max_deut=0.95):
    if any(abs(x) < 1e-10 for x in [k2-k1, k3-k2, k3-k1]):
        k2 += 1e-10; k3 += 2e-10
    d1 = max_deut * (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    d2 = max_deut * (k1*k2 / ((k3 - k2)*(k2 - k1))) *          ((np.exp(-k1*t)/(k3 - k1)) - (np.exp(-k2*t)/(k3 - k2)))
    d3 = max_deut * (1 - d1 - d2)
    d0 = 1 - d1 - d2 - d3
    return d0, d1, d2

# (Rest of the content will be appended in next step to avoid token limits)

# === Sidebar Configuration ===
with st.sidebar:
    st.header("Model Selection")
    model_type = st.selectbox("Choose kinetic model", [
        "Non-Sequential", "Single First-Order", "Two-Step Sequential", "Triple Sequential"
    ])
    initial_k1 = st.number_input("Initial guess for k1", value=0.01, format="%.5f")
    initial_k2 = st.number_input("Initial guess for k2", value=0.005, format="%.5f")
    initial_k3 = st.number_input("Initial guess for k3 (if needed)", value=0.001, format="%.5f")
    max_deut = st.slider("Max Deuterium Incorporation", 0.0, 1.0, 0.95, 0.01)
    batch_mode = st.checkbox("Batch process all Excel sheets", value=False)

# === File Upload ===
st.subheader("Upload kinetic data")
uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

# === Core Fitting Logic ===
def fit_model(model_type, time, d0, d1, d2):
    from scipy.optimize import curve_fit

    if model_type in ["Non-Sequential", "Single First-Order"]:
        model_func = nonsequential_model if model_type == "Non-Sequential" else single_first_order_model

        def wrapper(t, k1):
            return np.concatenate(model_func(t, k1, max_deut))

        y_data = np.concatenate([d0, d1, np.zeros_like(d2)])
        popt, pcov = curve_fit(wrapper, np.tile(time, 3), y_data, p0=[initial_k1], bounds=(0, np.inf))
        d0f, d1f, d2f = model_func(time, *popt, max_deut)
        r2 = 1 - np.sum((y_data - wrapper(np.tile(time, 3), *popt))**2)/np.sum((y_data - np.mean(y_data))**2)
        return {"k1": popt[0], "d0_fit": d0f, "d1_fit": d1f, "d2_fit": d2f, "r_squared": r2, "success": True}

    elif model_type == "Triple Sequential":
        def wrapper(t, k1, k2, k3):
            return np.concatenate(triple_first_order_model(t, k1, k2, k3, max_deut))
        y_data = np.concatenate([d0, d1, d2])
        popt, pcov = curve_fit(wrapper, np.tile(time, 3), y_data, p0=[initial_k1, initial_k2, initial_k3], bounds=(0, np.inf))
        d0f, d1f, d2f = triple_first_order_model(time, *popt, max_deut)
        r2 = 1 - np.sum((y_data - wrapper(np.tile(time, 3), *popt))**2)/np.sum((y_data - np.mean(y_data))**2)
        return {"k1": popt[0], "k2": popt[1], "k3": popt[2], "d0_fit": d0f, "d1_fit": d1f, "d2_fit": d2f, "r_squared": r2, "success": True}

    else:
        return fit_kinetic_data(time, d0, d1, d2, initial_k1, initial_k2, max_deut)
