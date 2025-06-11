# 📊 Previsão de Churn em Telecom — CRISP-DM + Streamlit

Este projeto aplica um pipeline completo de ciência de dados, usando a metodologia **CRISP-DM**, para prever o churn de clientes de uma empresa de telecomunicações. O resultado é um app Streamlit pronto para deploy, permitindo previsões individuais ou em lote.

---

## 🛠️ **Resumo das Etapas CRISP-DM**

### **Etapa 1: Entendimento do Negócio**
- Objetivo: Antecipar e mitigar o risco de cancelamento de clientes (“churn”).
- Métrica-alvo: Churn (`Yes`/`No`).

### **Etapa 2: Entendimento dos Dados**
- Base original: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (~7.043 registros, 21 colunas).
- Principais variáveis: características demográficas, serviços contratados, valores pagos.
- Explorou-se distribuição das variáveis, identificação de valores ausentes e correlações relevantes.

### **Etapa 3: Preparação dos Dados**
- Conversão de tipos e tratamento de valores ausentes.
- Remoção de colunas irrelevantes (`customerID`).
- Pipeline robusto: imputação, padronização (`StandardScaler`), one-hot encoding, garantia de ausência de NaN.

### **Etapa 4: Modelagem**
- Balanceamento de classes com **SMOTE**.
- Testes de Random Forest, XGBoost, Logistic Regression.
- Otimização de hiperparâmetros com GridSearchCV.
- Ajuste de threshold para maximizar recall.
- Principais resultados:

| Modelo / Threshold   | Precision | Recall | F1-score |
|---------------------|:---------:|:------:|:--------:|
| Random Forest       |   0.574   | 0.588  |  0.581   |
| Logistic Regression |   0.514   | 0.802  |  0.627   |
| XGBoost             |   0.579   | 0.690  |  0.630   |
| threshold = 0.3     |   0.438   | 0.922  |  0.594   |

### **Etapa 5: Avaliação**
- O modelo final de regressão logística, com threshold = 0.3, atingiu recall acima de 92% para churn, priorizando retenção máxima.
- Perfis de risco: contratos mensais, clientes com pouco tempo de casa e alto valor mensal.

### **Etapa 6: Deploy**
- App Streamlit interativo (entrada manual e em lote).
- Deploy possível localmente ou em nuvem (Streamlit Cloud, Hugging Face Spaces).

---

## 💻 **Como rodar o app**

### **1. Clone o repositório**
```bash
git clone https://github.com/seuusuario/seu-repo-churn.git
cd seu-repo-churn
