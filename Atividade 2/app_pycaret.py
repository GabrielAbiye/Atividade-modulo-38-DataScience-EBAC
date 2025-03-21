# Imports
import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import load_model, predict_model
import joblib  # Para carregar o pipeline do sklearn


import joblib

modelo = joblib.load("model_final.pkl")
print(modelo)




# Configuração da página
st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

# Função para converter DataFrame para CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter DataFrame para Excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Função principal
def main():
    st.write("## Escorando o modelo gerado no PyCaret ou Scikit-learn")
    st.markdown("---")

    # Upload de arquivo
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    # Escolha do modelo
    st.sidebar.write("## Escolha o modelo")
    model_choice = st.sidebar.selectbox("Escolha o modelo que deseja utilizar",
                                        options=["LightGBM (PyCaret)", "Regressão Logística (sklearn)"])

    # Verifica se há conteúdo carregado
    if data_file_1 is not None:
        file_extension = data_file_1.name.split('.')[-1]

        if file_extension == 'csv':
            df_credit = pd.read_csv(data_file_1)
        elif file_extension == 'ftr':
            df_credit = pd.read_feather(data_file_1)
        else:
            st.error("Formato de arquivo não suportado. Use CSV ou Feather.")
            st.stop()

        # Ajustar amostragem para evitar erro
        df_credit = df_credit.sample(min(50000, len(df_credit)), random_state=42)

        # Aplicar modelo
        if model_choice == "LightGBM (PyCaret)":
            model_saved = load_model('model_final')
            predict = predict_model(model_saved, data=df_credit)

        elif model_choice == "Regressão Logística (sklearn)":
            pipeline = joblib.load('model_regressao_logistica.pkl')
            predictions = pipeline.predict(df_credit)
            predict = df_credit.copy()
            predict['Previsão'] = predictions  # Adiciona a previsão ao DataFrame

        # Criar arquivo Excel para download
        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

# Iniciar a aplicação
if __name__ == '__main__':
    main()

