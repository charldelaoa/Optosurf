import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Create a function to generate the Lissajous pattern
def generate_pattern(a, b, delta):
    t = np.linspace(0, 10, 1000)
    x = np.sin(a * t + delta)
    y = np.sin(b * t)
    return x, y

# Create the Streamlit app
def app():
    st.title("Lissajous Pattern Generator")
    a = st.sidebar.slider("a", 1, 10, 3)
    b = st.sidebar.slider("b", 1, 10, 5)
    delta = st.slider("delta", 0.0, np.pi, 0.0)
    x, y = generate_pattern(a, b, delta)
    plt.plot(x, y)
    st.pyplot()

if __name__ == "__main__":
    app()