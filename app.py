import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuração da Página ---
# Usar o modo "wide" para aproveitar melhor o espaço da tela.
st.set_page_config(layout="wide", page_title="Dashboard de Previsão de Cancelamento")

# --- Carregamento e Cache dos Dados ---
# @st.cache_data garante que os dados sejam carregados apenas uma vez, melhorando a performance.
@st.cache_data
def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e realiza um pré-processamento básico.
    O caminho do arquivo deve ser relativo para funcionar no Streamlit Cloud.
    """
    try:
        df = pd.read_csv(file_path)
        # Tratamento simples de valores nulos para agilizar a análise.
        df['children'].fillna(df['children'].median(), inplace=True)
        df['agent'].fillna(0, inplace=True)
        df['company'].fillna(0, inplace=True)
        df = df.dropna(subset=['country'])
        return df
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no mesmo diretório que o `app.py`.")
        return None

# Carrega os dados usando o caminho relativo.
# O arquivo CSV deve estar na mesma pasta que o script app.py.
df = load_data("hotel_bookings.csv")

# Interrompe a execução se os dados não forem carregados
if df is None:
    st.stop()

# --- Barra Lateral (Sidebar) com Filtros e Configurações ---
st.sidebar.image("https://i.imgur.com/rIOdD7P.png", width=150) # Logo Fictícia
st.sidebar.title("Painel de Controle")
st.sidebar.markdown("Use os filtros abaixo para explorar os dados e configurar o modelo de previsão.")

st.sidebar.header("Filtros de Visualização", divider='rainbow')

# Filtro por tipo de hotel
hotel_type = st.sidebar.multiselect(
    '🏨 Selecione o Tipo de Hotel:',
    options=df['hotel'].unique(),
    default=df['hotel'].unique()
)

# Filtro por tempo de espera (lead time)
lead_time_range = st.sidebar.slider(
    '🗓️ Selecione o Tempo de Espera (Lead Time):',
    min_value=int(df['lead_time'].min()),
    max_value=int(df['lead_time'].max()),
    value=(int(df['lead_time'].min()), int(df['lead_time'].max()))
)

# Aplicar filtros ao dataframe
df_filtered = df[
    (df['hotel'].isin(hotel_type)) &
    (df['lead_time'].between(lead_time_range[0], lead_time_range[1]))
]

# --- Título Principal ---
st.title("🚀 Dashboard de Análise de Cancelamento de Reservas")
st.markdown("Bem-vindo! Este dashboard interativo foi criado para analisar dados de reservas de hotéis e construir um modelo preditivo para identificar o risco de cancelamento.")


# --- Seção da Capa ---
st.markdown("---")
col1_capa, col2_capa, col3_capa = st.columns([1, 3, 1])

with col1_capa:
    st.image("https://logodownload.org/wp-content/uploads/2017/11/universidade-de-brasilia-unb-logo-9.png", width=200)

with col2_capa:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="font-weight:bold;">Análise Interativa de Regressão Logística</h2>
            <p><strong>Disciplina:</strong> Sistemas de Informação e Engenharia de Produção</p>
            <p><strong>Autores:</strong> Bruno da Mata e Cristiano Rose</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3_capa:
    st.image("https://logodownload.org/wp-content/uploads/2017/11/universidade-de-brasilia-unb-logo-9.png", width=200)

st.markdown("---")


# --- Abas para Organizar o Conteúdo ---
tab1, tab2, tab3 = st.tabs(["📊 Análise Exploratória", "🤖 Modelagem Preditiva", "🧠 Interpretação do Modelo"])

# --- Aba 1: Análise Exploratória dos Dados ---
with tab1:
    st.header("Análise Exploratória dos Dados Filtrados")
    st.markdown(f"A seleção atual contém **{df_filtered.shape[0]}** registros de um total de **{df.shape[0]}**.")

    # Expander para mostrar/ocultar a tabela de dados
    with st.expander("Clique para ver a tabela de dados completa"):
        st.dataframe(df_filtered)

    st.subheader("Estatísticas Descritivas", divider='gray')
    st.write("Visão geral das variáveis numéricas do conjunto de dados filtrado.")
    st.dataframe(df_filtered.describe(), use_container_width=True)

    st.subheader("Visualizações Interativas", divider='gray')
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Pizza: Distribuição de Cancelamentos
        fig_pie = px.pie(df_filtered, names='is_canceled',
                         title='Distribuição de Cancelamentos',
                         color_discrete_sequence=px.colors.sequential.Teal,
                         labels={'is_canceled': 'Status do Cancelamento'})
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Gráfico de Barras: Cancelamentos por Tipo de Hotel
        fig_bar = px.histogram(df_filtered, x='hotel', color='is_canceled',
                               barmode='group',
                               title='Cancelamentos por Tipo de Hotel',
                               labels={'hotel': 'Tipo de Hotel', 'is_canceled': 'Status'},
                               color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        # Histograma de uma variável numérica selecionável
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
        selected_num_col = st.selectbox('Selecione uma variável numérica para ver sua distribuição:', options=numeric_cols, index=numeric_cols.index('lead_time'))
        fig_hist = px.histogram(df_filtered, x=selected_num_col, color='is_canceled',
                                title=f'Distribuição de {selected_num_col}',
                                marginal='box',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        # Gráfico de Dispersão para explorar relações
        st.write("Relação entre ADR, Lead Time e Cancelamento")
        fig_scatter = px.scatter(df_filtered.sample(min(1000, len(df_filtered))), # Amostra para performance
                                 x='lead_time', y='adr', color='is_canceled',
                                 title='ADR (Diária Média) vs. Lead Time',
                                 labels={'lead_time': 'Tempo de Espera (dias)', 'adr': 'Diária Média (ADR)'},
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_scatter, use_container_width=True)


# --- Aba 2: Modelagem Preditiva ---
with tab2:
    st.header("Treinamento do Modelo de Regressão Logística")
    st.markdown("Nesta seção, você pode selecionar as variáveis e treinar o modelo para prever `is_canceled`.")

    # --- Configurações do Modelo na Barra Lateral ---
    st.sidebar.header("Configuração do Modelo", divider='rainbow')

    # Seleção de variáveis para o modelo
    all_features = [col for col in df.columns if col not in ['is_canceled', 'reservation_status_date', 'reservation_status']]
    default_features = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 'booking_changes', 'previous_cancellations', 'market_segment', 'deposit_type', 'customer_type']
    
    # Garantir que as features padrão existam no dataframe
    valid_default_features = [f for f in default_features if f in all_features]
    
    selected_features = st.sidebar.multiselect(
        '⚙️ Selecione as variáveis preditoras:',
        options=all_features,
        default=valid_default_features
    )
    
    use_smote = st.sidebar.checkbox("Aplicar SMOTE para balancear os dados?", value=True)

    if st.sidebar.button("▶️ Treinar Modelo", type="primary", use_container_width=True):
        if not selected_features:
            st.warning("⚠️ Por favor, selecione ao menos uma variável para treinar o modelo.")
        else:
            with st.spinner("👩‍💻 Pré-processando dados e treinando o modelo... Isso pode levar alguns segundos."):
                # Preparação dos dados
                X = df_filtered[selected_features].copy()
                y = df_filtered['is_canceled'].copy()

                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                categorical_features = X.select_dtypes(include=['object']).columns.tolist()

                # Cria o pre-processador com OneHotEncoder para variáveis categóricas
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
                    ],
                    remainder='passthrough'
                )

                # Aplica o pré-processamento
                X_processed = preprocessor.fit_transform(X)
                
                # Divisão em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)

                # Aplica SMOTE se a opção estiver marcada
                if use_smote:
                    st.info("Aplicando SMOTE para balancear as classes no conjunto de treino...")
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train
                
                # Treinamento do modelo
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_resampled, y_train_resampled)
                
                # Previsões
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                st.success("✅ Modelo treinado com sucesso!")

                # Armazenar resultados no session_state para usar em outras abas
                st.session_state.model_trained = True
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_proba = y_proba
                st.session_state.selected_features = selected_features
                st.session_state.numeric_features = numeric_features
                st.session_state.categorical_features = categorical_features
                st.session_state.X = X # Armazena o dataframe original das features

    # Exibir resultados se o modelo já foi treinado
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.subheader("Resultados da Avaliação", divider='gray')
        
        st.markdown("#### Métricas de Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Acurácia", f"{accuracy_score(st.session_state.y_test, st.session_state.y_pred):.2%}")
        col2.metric("Precisão", f"{precision_score(st.session_state.y_test, st.session_state.y_pred):.2%}")
        col3.metric("Recall", f"{recall_score(st.session_state.y_test, st.session_state.y_pred):.2%}")
        col4.metric("F1-Score", f"{f1_score(st.session_state.y_test, st.session_state.y_pred):.2%}")
        col5.metric("AUC", f"{roc_auc_score(st.session_state.y_test, st.session_state.y_proba):.3f}")

        st.markdown("#### Visualizações de Avaliação")
        col_roc, col_cm = st.columns(2)
        
        with col_roc:
            # Curva ROC
            fpr, tpr, _ = roc_curve(st.session_state.y_test, st.session_state.y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc_score(st.session_state.y_test, st.session_state.y_proba):.3f}'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Referência', line=dict(dash='dash')))
            fig_roc.update_layout(title='<b>Curva ROC</b>', xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos')
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_cm:
            # Matriz de Confusão
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            fig_cm = px.imshow(cm, text_auto=True, 
                               labels=dict(x="Predição", y="Real", color="Contagem"),
                               x=['Não Cancelou', 'Cancelou'], y=['Não Cancelou', 'Cancelou'],
                               color_continuous_scale='Blues')
            fig_cm.update_layout(title='<b>Matriz de Confusão</b>')
            st.plotly_chart(fig_cm, use_container_width=True)

# --- Aba 3: Interpretação do Modelo ---
with tab3:
    st.header("Interpretação dos Resultados do Modelo")

    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.info("ℹ️ Por favor, treine um modelo na aba 'Modelagem Preditiva' para ver os resultados aqui.")
    else:
        # Recuperar objetos do session_state
        model = st.session_state.model
        preprocessor = st.session_state.preprocessor
        numeric_features = st.session_state.numeric_features
        categorical_features = st.session_state.categorical_features
        X = st.session_state.X

        st.subheader("Coeficientes do Modelo", divider='gray')
        st.markdown("Os coeficientes indicam o impacto de cada variável na probabilidade de cancelamento.")

        # Extrair nomes das features após o OneHotEncoding
        try:
            ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
            feature_names = numeric_features + ohe_feature_names
        except (AttributeError, KeyError):
             st.error("Não foi possível obter os nomes das features categóricas.")
             feature_names = []
        
        if feature_names:
            coefs = pd.DataFrame(
                model.coef_[0],
                index=feature_names,
                columns=['Log-Odds']
            )
            coefs['Odds Ratio'] = np.exp(coefs['Log-Odds'])
            coefs['Impacto %'] = (coefs['Odds Ratio'] - 1) * 100
            
            st.dataframe(coefs.sort_values('Odds Ratio', ascending=False), use_container_width=True)

            with st.expander("Análise Detalhada dos Coeficientes"):
                top_features = coefs.sort_values('Odds Ratio', ascending=False).head(5)
                bottom_features = coefs.sort_values('Odds Ratio', ascending=True).head(5)
                
                st.markdown("##### Fatores que mais **AUMENTAM** a chance de cancelamento:")
                for index, row in top_features.iterrows():
                    st.write(f"- **{index}**: Aumenta a chance de cancelamento em **{row['Impacto %']:.2f}%**.")

                st.markdown("---")
                st.markdown("##### Fatores que mais **REDUZEM** a chance de cancelamento:")
                for index, row in bottom_features.iterrows():
                     st.write(f"- **{index}**: Reduz a chance de cancelamento em **{(1 - row['Odds Ratio'])*100:.2f}%**.")

        st.subheader("Curva Logística Interativa", divider='gray')
        st.markdown("Veja como a probabilidade de cancelamento muda ao variar uma variável, mantendo as outras constantes.")

        # Permitir a seleção de uma variável contínua
        continuous_features = st.session_state.numeric_features
        if not continuous_features:
            st.warning("Nenhuma variável contínua foi selecionada no modelo para gerar a curva.")
        else:
            var_to_plot = st.selectbox("Selecione a variável contínua para o eixo X:", options=continuous_features)

            # Criar um range de valores para a variável selecionada
            var_range = np.linspace(X[var_to_plot].min(), X[var_to_plot].max(), 100)
            
            # Criar um dataframe de predição
            pred_df = pd.DataFrame({col: [X[col].median() if col in numeric_features else X[col].mode()[0]] * 100 for col in st.session_state.selected_features})
            pred_df[var_to_plot] = var_range
            
            # Pré-processar os dados de predição
            pred_df_processed = preprocessor.transform(pred_df)
            
            # Calcular as probabilidades
            probabilities = model.predict_proba(pred_df_processed)[:, 1]
            
            # Plotar a curva
            fig_logistic = px.line(x=var_range, y=probabilities, 
                                   labels={'x': var_to_plot, 'y': 'Probabilidade de Cancelamento'},
                                   title=f'Curva Logística para "{var_to_plot}"')
            fig_logistic.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_logistic, use_container_width=True)

            st.info(f"""
            **Como interpretar este gráfico:**
            O eixo X mostra os diferentes valores da variável **{var_to_plot}**.
            O eixo Y mostra a probabilidade prevista de um cliente cancelar a reserva para cada um desses valores, assumindo que todas as outras variáveis no modelo estão fixas em seus valores médios (mediana para números, moda para categorias).
            A forma de "S" da curva é característica da regressão logística.
            """)