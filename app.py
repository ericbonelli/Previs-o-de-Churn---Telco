import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =======================
# Carregar modelo e pipeline
# =======================
@st.cache_resource
def load_pipeline_model():
    with open('pipeline_preproc.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('modelo_churn.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_pipeline_model()

THRESHOLD = 0.3

st.title("📱 Previsão de Churn - Telco (Threshold = 0.3)")

# =======================
# Entrada manual (um cliente)
# =======================
st.header("Prever churn para um cliente")

gender = st.selectbox('Gênero', ['Female', 'Male'])
SeniorCitizen = st.selectbox('É idoso?', [0, 1])
Partner = st.selectbox('Possui parceiro(a)?', ['Yes', 'No'])
Dependents = st.selectbox('Possui dependentes?', ['Yes', 'No'])
tenure = st.number_input('Meses de contrato', min_value=0, max_value=100, value=1)
PhoneService = st.selectbox('Telefone ativo?', ['Yes', 'No'])
MultipleLines = st.selectbox('Mais de uma linha?', ['Yes', 'No', 'No phone service'])
InternetService = st.selectbox('Tipo de Internet', ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox('Segurança online?', ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox('Backup online?', ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox('Proteção de dispositivo?', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('Suporte técnico?', ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox('TV Streaming?', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('Filmes Streaming?', ['Yes', 'No', 'No internet service'])
Contract = st.selectbox('Tipo de contrato', ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox('Fatura sem papel?', ['Yes', 'No'])
PaymentMethod = st.selectbox('Método de pagamento', [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
MonthlyCharges = st.number_input('Valor mensal (R$)', min_value=0.0, max_value=10000.0, value=70.0)
TotalCharges = st.number_input('Valor total já pago (R$)', min_value=0.0, max_value=100000.0, value=70.0)

input_dict = {
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
}

if st.button('Prever Churn (1 cliente)'):
    df_input = pd.DataFrame(input_dict)
    X_proc = preprocessor.transform(df_input)
    prob = model.predict_proba(X_proc)[:, 1][0]
    pred = int(prob >= THRESHOLD)
    st.write(f"Probabilidade de churn: **{prob:.1%}**")
    st.write("Previsão:", "🔴 RISCO ALTO de churn" if pred else "🟢 Risco baixo de churn")

# =======================
# Entrada em lote (upload CSV)
# =======================
st.header("Prever churn em lote (arquivo CSV)")

st.markdown("""
O arquivo deve conter as mesmas colunas do formulário acima, sem a coluna 'Churn'.
""")

file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
if file:
    df_batch = pd.read_csv(file)
    X_proc_batch = preprocessor.transform(df_batch)
    probs = model.predict_proba(X_proc_batch)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    result = df_batch.copy()
    result['Churn_Prob'] = probs
    result['Churn_Pred'] = preds
    st.write(result)
    st.success(f"Total de clientes previstos como RISCO ALTO de churn: {(preds==1).sum()} de {len(preds)}")
    st.download_button("Baixar resultados (CSV)", result.to_csv(index=False), "churn_predicoes.csv", "text/csv")
