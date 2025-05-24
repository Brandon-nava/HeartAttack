
# Cargar modelos y escaladores
def load_model_and_scaler(model_path, scaler_path):
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    return model, scaler

# Cargar métricas desde archivo
def load_metrics(json_path):
    return pd.read_json(json_path)

# Visualización de probabilidad
def plot_pie(prob):
    fig, ax = plt.subplots()
    ax.pie([1 - prob, prob], labels=["Negativo", "Positivo"], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    st.pyplot(fig)

# Visualización de matriz de confusión
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

# Visualización de curva ROC
def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Diccionario de modelos
model_info = {
    "Modelo 1 (Age, CK-MB, Troponin)": {
        "features": ['Age', 'CK-MB', 'Troponin'],
        "model_path": 'modeloDataset1.pkl',
        "scaler_path": 'escalador1.pkl',
        "metrics_path": 'metricas_modelo_1.json'
    },
    "Modelo 2 (exang, cp, oldpeak, thalach, ca, target)": {
        "features": ['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'target'],
        "model_path": 'modeloDataset2.pkl',
        "scaler_path": 'escalador2.pkl',
        "metrics_path": 'metricas_modelo_2.json'
    }
}

st.title("Predicción de Ataques Cardiacos")
opcion = st.selectbox("Selecciona el modelo a usar", list(model_info.keys()))
info = model_info[opcion]

# Cargar modelo y escalador correspondiente
model, scaler = load_model_and_scaler(info["model_path"], info["scaler_path"])

st.subheader("Ingresar valores")
user_input = []
for var in info["features"]:
    user_input.append(st.number_input(f"{var}", step=0.1))

if st.button("Predecir"):
    input_scaled = scaler.transform([user_input])
    start = time.time()
    proba = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]
    elapsed_time = time.time() - start

    st.subheader("Resultado de Predicción")
    plot_pie(proba)
    st.write(f"Predicción: {'Positivo' if pred == 1 else 'Negativo'}")
    st.write(f"Tiempo de inferencia: {elapsed_time:.4f} segundos")

# Sección de métricas
st.sidebar.title("Visualización de métricas")
metrica = st.sidebar.radio("Selecciona una métrica", ["Tabla de métricas", "Curva ROC", "Matriz de Confusión"])

metrics_df = load_metrics(info["metrics_path"])
y_true = metrics_df["y_true"]
y_pred = metrics_df["y_pred"]
y_proba = metrics_df["y_proba"]

st.subheader("Métricas del modelo")
if metrica == "Tabla de métricas":
    st.table(metrics_df[["accuracy", "precision", "recall", "f1"]].drop_duplicates())
elif metrica == "Curva ROC":
    plot_roc(y_true, y_proba)
elif metrica == "Matriz de Confusión":
    plot_confusion_matrix(y_true, y_pred)
