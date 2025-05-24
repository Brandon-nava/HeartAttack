import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle5
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Clasificación de ataque cardíaco", layout="wide")

# Cargar configuraciones de modelos
config_modelos = {
    'Modelo 1': {
        'modelo': joblib.load('modeloDataset1.pkl'),
        'scaler': joblib.load('escalador1.pkl'),
        'variables': ['Age', 'CK-MB', 'Troponin']
    },
    'Modelo 2': {
        'modelo': joblib.load('modeloDataset2.pkl'),
        'scaler': joblib.load('escalador2.pkl'),
        'variables': ['exang', 'cp', 'oldpeak', 'thalach', 'ca','target']
    }
}

'''
# Columna para imagen
col1, col2 = st.columns([1, 3])
with col1:
    imagen = Image.open("imagenes/ataque_cardiaco.jpg")
    st.image(imagen, use_column_width=True)

with col2:
    st.title("Clasificación de ataque cardíaco")
    st.write("""
    Esta aplicación predice el riesgo de un posible ataque cardíaco utilizando diferentes algoritmos de clasificación.
    """)
'''
st.markdown("---")

# Selección de modelo
st.subheader("Seleccione el modelo para la predicción")
modelo_nombre = st.selectbox("Modelo", list(config_modelos.keys()))

# Obtener modelo, scaler y variables
config = config_modelos[modelo_nombre]
modelo = config['modelo']
scaler = config['scaler']
vars_requeridas = config['variables']

st.markdown("### Ingrese los datos del paciente")

# Entradas condicionales
valores = {}
if 'edad' in vars_requeridas:
    valores['edad'] = st.number_input("Edad", min_value=0, max_value=120, value=45)

if 'ckmb' in vars_requeridas:
    ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
    valores['ckmb'] = np.log(ckmb + 1e-10)

if 'troponina' in vars_requeridas:
    troponina = st.number_input("Troponina", value=0.003, min_value=0.000, format="%.3f", step=0.001)
    valores['troponina'] = np.log(troponina + 1e-10)

# Procesar predicción al hacer clic
if st.button("Predecir"):
    entrada = np.array([[valores[v] for v in vars_requeridas]])
    entrada_scaled = scaler.transform(entrada)
    pred = modelo.predict(entrada_scaled)[0]

    resultado = "Positivo" if pred == 1 else "Negativo"
    color = "red" if pred == 1 else "green"

    st.markdown(f"### Resultado: <span style='color:{color}'>{resultado}</span>", unsafe_allow_html=True)

    # Mostrar probabilidades si están disponibles
    if hasattr(modelo, 'predict_proba'):
        proba = modelo.predict_proba(entrada_scaled)[0]
        st.write(f"Probabilidad Negativa: {proba[0]:.2f}")
        st.write(f"Probabilidad Positiva: {proba[1]:.2f}")
