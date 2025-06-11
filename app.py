import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Previsão de Churn - Telco", layout="wide")

# ========================
# Tutorial e instruções
# ========================
st.title("📱 Previsão de Churn - Telco")

st.markdown("""
## 👋 Bem-vindo ao sistema de previsão de Churn!

### Como usar:

- **Previsão individual:** Preencha o formulário com os dados do cliente e clique em **Prever Churn**.
- **Previsão em lote:** Faça upload de um arquivo CSV seguindo o modelo (baixe o exemplo abaixo).
- O app irá mostrar a **probabilidade de churn** e, se for alta, recomendar uma ação para retenção.

#### O que fazer quando um cliente tem risco ALTO de churn?
> Entre em contato IMEDIATO, priorize o atendimento, avalie a possibilidade de oferecer condições especiais ou bônus para retenção.
""")

# ========================
# Carregar pipeline e modelo
# ========================
@st.cache_resource
def load_pipeline_model():
    with open('pipeline_preproc.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('modelo_churn.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_pipeline_model()
THRESHOLD = 0.3

# ========================
# Previsão Individual
# ========================
st.header("Prever churn para um cliente")

with st.form(key='form_cliente'):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gênero', ['Female', 'Male'])
        SeniorCitizen = st.selectbox('É idoso?', [0, 1])
        Partner = st.selectbox('Possui parceiro(a)?', ['Yes', 'No'])
        Dependents = st.selectbox('Possui dependentes?', ['Yes', 'No'])
        tenure = st.number_input('Meses de contrato', min_value=0, max_value=100, value=1)
        PhoneService = st.selectbox('Telefone ativo?', ['Yes', 'No'])

    with col2:
        MultipleLines = st.selectbox('Mais de uma linha?', ['Yes', 'No', 'No phone service'])
        InternetService = st.selectbox('Tipo de Internet', ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox('Segurança online?', ['Yes', 'No', 'No internet service'])
        OnlineBackup = st.selectbox('Backup online?', ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox('Proteção de dispositivo?', ['Yes', 'No', 'No internet service'])
        TechSupport = st.selectbox('Suporte técnico?', ['Yes', 'No', 'No internet service'])

    with col3:
        StreamingTV = st.selectbox('TV Streaming?', ['Yes', 'No', 'No internet service'])
        StreamingMovies = st.selectbox('Filmes Streaming?', ['Yes', 'No', 'No internet service'])
        Contract = st.selectbox('Tipo de contrato', ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox('Fatura sem papel?', ['Yes', 'No'])
        PaymentMethod = st.selectbox('Método de pagamento', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        MonthlyCharges = st.number_input('Valor mensal (R$)', min_value=0.0, max_value=10000.0, value=70.0)
        TotalCharges = st.number_input('Valor total já pago (R$)', min_value=0.0, max_value=100000.0, value=70.0)

    submit_button = st.form_submit_button(label='Prever Churn (1 cliente)')

if submit_button:
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
    df_input = pd.DataFrame(input_dict)
    X_proc = preprocessor.transform(df_input)
    prob = model.predict_proba(X_proc)[:, 1][0]
    pred = int(prob >= THRESHOLD)
    st.write(f"Probabilidade de churn: **{prob:.1%}**")
    if pred:
        st.error("🔴 RISCO ALTO de churn!\n\n**Ação recomendada:** Entre em contato imediato com o cliente e ofereça benefícios para retenção.")
    else:
        st.success("🟢 Risco baixo de churn.")

# ========================
# Previsão em lote (CSV)
# ========================
st.header("Prever churn em lote (arquivo CSV)")

st.markdown("""
O arquivo deve conter as mesmas colunas do formulário acima (exceto `Churn`).  
Você pode baixar um [exemplo de CSV aqui](https://raw.githubusercontent.com/seuusuario/seu-repo-churn/main/exemplo_clientes.csv).
""")

file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
if file:
    df_batch = pd.read_csv(file)
    # Proteção para tipos e ordem de colunas
    try:
        X_proc_batch = preprocessor.transform(df_batch)
        probs = model.predict_proba(X_proc_batch)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)
        result = df_batch.copy()
        result['Churn_Probabilidade'] = probs
        result['Churn_Predicao'] = preds
        result['Prioridade'] = np.where(preds==1, '🔴 ALTO — Aja já!', '🟢 Baixo')
        result['Ação recomendada'] = np.where(preds==1, 
                                              'Entrar em contato imediato e oferecer benefício!', 
                                              '-')
        # Destaque visual no DataFrame
        def color_risk(val):
            if val == '🔴 ALTO — Aja já!':
                return 'background-color: #ffcccc; font-weight: bold'
            elif val == '🟢 Baixo':
                return 'background-color: #ccffcc'
            return ''
        st.write("**Resultados:**")
        st.dataframe(result.style.applymap(color_risk, subset=['Prioridade']))

        st.warning(f"Total de clientes com risco ALTO: {(result['Churn_Predicao']==1).sum()} de {len(result)}")
        st.download_button("Baixar resultados (CSV)", result.to_csv(index=False), "churn_predicoes.csv", "text/csv")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo. Confira se as colunas estão corretas. Detalhes: {e}")

# ========================
# Rodapé
# ========================
st.markdown("---")
st.markdown("""
App desenvolvido para previsão de churn com base em dados reais de telecom.  
Dúvidas? [Seu Nome](mailto:seu@email.com)
""")
