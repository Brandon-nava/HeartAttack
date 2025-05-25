import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Clasificación de ataque cardíaco", layout="wide")

# Cargar configuraciones de modelos
config_modelos = {
    'Modelo 1': {
        'modelo': joblib.load('modelo_1.pkl'),
        'scaler': joblib.load('escalador1.pkl'),
        'variables': ['Age', 'CK-MB', 'Troponin']
    },
    'Modelo 2': {
        'modelo': joblib.load('modeloDataset2.pkl'),
        'scaler': joblib.load('escalador2.pkl'),
        'variables': ['exang', 'cp', 'oldpeak', 'thalach', 'ca','target']
    }
}

st.title("Clasificación de ataque cardíaco")
st.write("""
Esta aplicación predice el riesgo de un posible ataque cardíaco utilizando diferentes algoritmos de clasificación.
""")

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

if 'Age' in vars_requeridas:
    valores['Age'] = st.number_input("Edad", min_value=0, max_value=120, value=45)

if 'CK-MB' in vars_requeridas:
    ckmb = st.number_input("CK-MB", value=2.86, min_value=0.00, format="%.2f")
    valores['CK-MB'] = np.log(ckmb + 1e-10)

if 'Troponin' in vars_requeridas:
    troponina = st.number_input("Troponina", value=0.003, min_value=0.000, format="%.3f", step=0.001)
    valores['Troponin'] = np.log(troponina + 1e-10)

if 'exang' in vars_requeridas:
    valores['exang'] = st.selectbox("¿Ejercicio inducido angina? (exang)", options=[0, 1])

if 'cp' in vars_requeridas:
    valores['cp'] = st.selectbox("Tipo de dolor de pecho (cp)", options=[0, 1, 2, 3])

if 'oldpeak' in vars_requeridas:
    valores['oldpeak'] = st.number_input("Depresión del ST (oldpeak)", value=1.0, format="%.2f")

if 'thalach' in vars_requeridas:
    valores['thalach'] = st.number_input("Frecuencia cardiaca máxima (thalach)", min_value=50, max_value=250, value=150)

if 'ca' in vars_requeridas:
    valores['ca'] = st.number_input("Número de vasos coloreados (ca)", min_value=0, max_value=4, value=0)

if 'target' in vars_requeridas:
    valores['target'] = st.selectbox("¿Condición presente? (target)", options=[0, 1])

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
        etiquetas = ['Negativo', 'Positivo']
        colores = ['#58b915', '#e57b0b']  # verde y rojo

        fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')  # fondo transparente del canvas
        wedges, texts, autotexts = ax.pie(
            proba,
            labels=etiquetas,
            autopct='%1.1f%%',
            startangle=90,
            colors=colores,
            textprops={'color': 'white', 'weight': 'bold', 'fontsize': 12}
        )
        ax.axis('equal')  # para que sea un círculo
        fig.patch.set_alpha(0)  # fondo transparente del gráfico completo
        st.pyplot(fig)



