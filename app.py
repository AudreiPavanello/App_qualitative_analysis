import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from collections import Counter
import re
import string
import io # Para exportar Excel

# Novas importações
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np # Para LDA

# --- Configuração da Página e Cache ---

st.set_page_config(page_title="Análise Qualitativa de Textos", layout="wide")

# --- Modelos e Dados (Carregamento Cacheado) ---

@st.cache_resource
def load_sentiment_model():
    """Carrega o modelo de análise de sentimentos."""
    st.write("Carregando modelo de sentimento...") # Feedback
    try:
        # Usar um modelo mais leve se performance for crítica ou manter o atual
        model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        # model_name = "neuralmind/bert-base-portuguese-cased" # Exemplo alternativo (requereria fine-tuning p/ sentimento)
        sentiment_pipeline = pipeline("text-classification", model=model_name)
        st.success("Modelo de sentimento carregado.")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de sentimento: {e}")
        st.stop()

# Cache para vetorizador e modelo LDA (depende do texto e num_topics)
@st.cache_data(show_spinner="Realizando Modelagem de Tópicos (LDA)...")
def perform_topic_modeling(texts, stop_words, num_topics=5, n_top_words=10):
    """Realiza LDA nos textos fornecidos e retorna tópicos e atribuições."""
    if not texts or len(texts) < num_topics: # Verifica se há texto suficiente
        st.warning("Não há textos suficientes para realizar a modelagem de tópicos com a configuração atual.")
        return None, None, None

    vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_df=0.9, min_df=2, ngram_range=(1,2)) # Min_df=2 ignora palavras raras
    try:
        tfidf = vectorizer.fit_transform(texts)
        # Checa se o vocabulário não está vazio após aplicar stop words e min_df
        if tfidf.shape[1] == 0:
             st.warning("Nenhum termo válido encontrado após pré-processamento para LDA.")
             return None, None, None
    except ValueError:
        st.warning("Não foi possível vetorizar o texto para LDA (pode ser devido a todos os textos serem curtos ou apenas stop words).")
        return None, None, None


    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=10) # Random state para reprodutibilidade
    try:
        lda_topic_matrix = lda.fit_transform(tfidf)
    except Exception as e:
        st.error(f"Erro durante o treinamento do LDA: {e}")
        return None, None, None

    # Obter as palavras mais importantes por tópico
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f"Tópico {topic_idx + 1}"] = top_features

    # Atribuir o tópico mais provável a cada documento
    doc_topic_assignments = lda_topic_matrix.argmax(axis=1)

    return topics, doc_topic_assignments, lda # Retorna o modelo LDA treinado também

@st.cache_data
def load_data(uploaded_file):
    """Carrega dados de um arquivo .csv ou .xlsx."""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Tenta detectar encoding comum ou usa fallback
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use .csv ou .xlsx")
            return None
        df.dropna(axis=0, how='all', inplace=True)
        # Converter todas as colunas para string para evitar erros posteriores
        # df = df.astype(str) # Cuidado: pode converter números indesejadamente
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

# --- Funções Auxiliares ---

# Stopwords (manter a lista definida anteriormente)
stop_words_pt = set([
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma',
    'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele',
    'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já',
    'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era',
    'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles',
    'estão', 'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha',
    'têm', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe',
    'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocês', 'vos',
    'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
    'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto',
    'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram',
    'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam',
    'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há',
    'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos',
    'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei',
    'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos',
    'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja',
    'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será',
    'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'têm', 'tinha',
    'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha',
    'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem',
    'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam', 'pra', 'pro', 'tá', 'né',
    'aí', 'lá', 'cá', 'etc', 'sobre', 'muita', 'muitos', 'coisa', 'coisas', 'pode', 'onde'
])

def clean_text(text):
    """Limpa o texto: minúsculas, remove pontuação, números e espaços extras."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_word_counts(texts, stop_words, n_top_words=20):
    """Conta as palavras mais frequentes, excluindo stop words."""
    # Limpa e junta os textos
    cleaned_texts = [clean_text(text) for text in texts if isinstance(text, str)]
    all_text = ' '.join(cleaned_texts)
    words = all_text.split()

    # Remove stop words e palavras curtas
    words = [word for word in words if word not in stop_words and len(word) > 2]

    if not words:
        return pd.DataFrame(columns=['Palavra', 'Frequência'])

    word_counts = Counter(words)
    most_common = word_counts.most_common(n_top_words)

    df_counts = pd.DataFrame(most_common, columns=['Palavra', 'Frequência'])
    return df_counts

def analyze_sentiment(text, pipe):
    """Aplica o pipeline de sentimento."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {"label": "N/A", "score": 0.0, "label_pt": "N/A"}
    try:
        result = pipe(str(text))[0] # Garante que é string
        label_map = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
        result['label_pt'] = label_map.get(result['label'], result['label'].capitalize()) # Mapeia ou capitaliza
        return result
    except Exception as e:
        print(f"Erro ao analisar sentimento do texto: '{str(text)[:50]}...' ({e}). Marcando como Erro.")
        return {"label": "Erro", "score": 0.0, "label_pt": "Erro"}

@st.cache_data(show_spinner="Gerando nuvem de palavras...")
def generate_word_cloud(texts, stop_words):
    """Gera uma figura matplotlib com a nuvem de palavras."""
    # Limpa e junta os textos
    cleaned_texts = [clean_text(text) for text in texts if isinstance(text, str)]
    full_text = ' '.join(cleaned_texts)

    if not full_text.strip():
        st.warning("Não há texto válido para gerar a nuvem de palavras.")
        return None

    wordcloud_gen = WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=stop_words,
                              min_font_size=10).generate(full_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_gen, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

def to_excel(df):
    """Converte DataFrame para bytes Excel."""
    output = io.BytesIO()
    # Use 'xlsxwriter' ou 'openpyxl'
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='AnaliseQualitativa')
    except ImportError:
         st.error("Biblioteca 'openpyxl' não encontrada. Instale com 'pip install openpyxl'")
         return None
    processed_data = output.getvalue()
    return processed_data

# --- Interface Principal ---

st.title("📊 Ferramenta de Análise Qualitativa de Textos")
st.markdown("""
Carregue um arquivo `.xlsx` ou `.csv`, selecione a coluna com texto e explore
análises de sentimento, tópicos e palavras frequentes.
""")

# --- Carregar Pipeline de Sentimento ---
sentiment_pipeline = load_sentiment_model()

# --- Upload e Seleção de Coluna ---
uploaded_file = st.sidebar.file_uploader("1. Escolha o arquivo (.csv ou .xlsx)", type=['csv', 'xlsx'])

if uploaded_file:
    df_original = load_data(uploaded_file)

    if df_original is not None:
        st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado ({len(df_original)} linhas).")

        # Seleciona apenas colunas que parecem ser textuais ou 'object'
        potential_text_cols = [col for col in df_original.columns if pd.api.types.is_string_dtype(df_original[col]) or pd.api.types.is_object_dtype(df_original[col])]
        if not potential_text_cols:
             st.sidebar.error("Nenhuma coluna de texto detectada no arquivo.")
             st.stop()

        text_column = st.sidebar.selectbox(
            "2. Selecione a coluna com o texto",
            options=[''] + potential_text_cols,
            index=0
        )

        # Botão de Análise na Sidebar
        analyze_button = st.sidebar.button("🚀 Iniciar Análise Completa", key="analyze_button", disabled=(not text_column))

        # --- Lógica Principal da Análise ---
        if analyze_button:
            if text_column not in df_original.columns:
                st.error(f"Coluna '{text_column}' não encontrada.")
            else:
                # Realiza a análise de sentimento
                progress_bar = st.progress(0, text="Analisando sentimentos...")
                df_analysis = df_original.copy()
                # Garante que a coluna de texto é string e lida com NaNs
                texts_to_analyze = df_analysis[text_column].fillna('').astype(str)
                results = []
                total_rows = len(texts_to_analyze)

                for i, text in enumerate(texts_to_analyze):
                    result = analyze_sentiment(text, sentiment_pipeline)
                    results.append(result)
                    progress_bar.progress((i + 1) / total_rows, text=f"Analisando sentimentos... {i+1}/{total_rows}")

                progress_bar.empty()
                df_results = pd.DataFrame(results)
                df_analysis['Sentimento_Label'] = df_results['label_pt']
                df_analysis['Sentimento_Score'] = df_results['score'].round(4)

                st.session_state['df_analyzed'] = df_analysis # Guarda no estado
                st.session_state['text_column'] = text_column
                st.session_state['analysis_done'] = True # Flag para indicar que análise foi feita
                st.success("Análise de sentimento concluída!")
                # Força o rerender para mostrar os resultados/abas
                st.rerun()

        # --- Exibição dos Resultados e Análises Adicionais ---
        if st.session_state.get('analysis_done', False) and 'df_analyzed' in st.session_state:
            df_analyzed = st.session_state['df_analyzed']
            text_column = st.session_state['text_column']

            st.sidebar.markdown("---")
            st.sidebar.header("Filtros")
            # Filtro por Sentimento
            available_sentiments = sorted([s for s in df_analyzed['Sentimento_Label'].unique() if s not in ['N/A', 'Erro']])
            selected_sentiments = st.sidebar.multiselect(
                "Filtrar por Sentimento:",
                options=available_sentiments,
                default=available_sentiments # Começa com todos selecionados
            )

            if not selected_sentiments:
                st.sidebar.warning("Selecione pelo menos um sentimento.")
                # Mostra tudo se nada for selecionado, ou pode optar por mostrar nada
                df_display = df_analyzed.copy()
            else:
                 # Inclui N/A e Erro se não estiverem explicitamente filtrados
                 sentiments_to_show = selected_sentiments + [s for s in ['N/A', 'Erro'] if s in df_analyzed['Sentimento_Label'].unique()]
                 df_display = df_analyzed[df_analyzed['Sentimento_Label'].isin(sentiments_to_show)].copy()


            st.sidebar.markdown("---")
            st.sidebar.header("Exportar")
            if not df_display.empty:
                 excel_data = to_excel(df_display)
                 if excel_data:
                     st.sidebar.download_button(
                         label="Baixar Resultados Filtrados (.xlsx)",
                         data=excel_data,
                         file_name=f"analise_{uploaded_file.name.split('.')[0]}.xlsx",
                         mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                     )
            else:
                 st.sidebar.info("Nenhum dado para exportar com os filtros atuais.")

            # --- Abas para Diferentes Análises ---
            tab1, tab2, tab3, tab4 = st.tabs(["📜 Tabela de Dados", "📊 Sentimento Geral", "🔍 Análise de Tópicos (LDA)", "☁️ Nuvem de Palavras"])

            with tab1:
                st.header("Tabela de Dados Analisados")
                st.markdown(f"Exibindo {len(df_display)} de {len(df_analyzed)} linhas com base nos filtros.")
                # Seleciona colunas relevantes para exibir
                cols_to_show = [text_column, 'Sentimento_Label', 'Sentimento_Score']
                # Adiciona coluna de Tópico se existir (será criada na tab3)
                if 'Tópico_Predito' in df_display.columns:
                    cols_to_show.append('Tópico_Predito')
                st.dataframe(df_display[cols_to_show])

            with tab2:
                st.header("Análise Geral de Sentimento")
                st.markdown("_Gráficos baseados nos dados filtrados._")

                if not df_display.empty:
                    col1, col2 = st.columns(2)
                    sentiment_counts_display = df_display['Sentimento_Label'].value_counts().reset_index()
                    sentiment_counts_display.columns = ['Sentimento', 'Contagem']

                    with col1:
                        st.subheader("Distribuição Percentual")
                        sentiment_counts_pie = sentiment_counts_display[~sentiment_counts_display['Sentimento'].isin(['N/A', 'Erro'])]
                        if not sentiment_counts_pie.empty:
                            fig_pie = px.pie(sentiment_counts_pie, names='Sentimento', values='Contagem',
                                             title="Sentimentos (Excluindo N/A e Erro)",
                                             color='Sentimento',
                                             color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey'})
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("Não há dados de sentimento (Positivo/Negativo/Neutro) para exibir no gráfico de pizza com os filtros atuais.")

                    with col2:
                        st.subheader("Contagem Total")
                        fig_bar_total = px.bar(sentiment_counts_display, x='Sentimento', y='Contagem',
                                                title="Contagem Absoluta por Sentimento", text_auto=True,
                                                 color='Sentimento',
                                                 color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey', 'N/A':'lightblue', 'Erro': 'orange'})
                        st.plotly_chart(fig_bar_total, use_container_width=True)

                    # Palavras Frequentes por Sentimento (dentro da Tab 2)
                    st.markdown("---")
                    st.subheader("Palavras Mais Frequentes por Sentimento")
                    st.markdown("_Considerando os dados filtrados. Exclui stop words e números._")
                    n_top_words_sentiment = st.slider("Número de palavras a exibir por sentimento:", 5, 30, 10, key="n_words_sentiment")

                    combined_word_counts_df = pd.DataFrame()
                    sentiments_in_display = [s for s in df_display['Sentimento_Label'].unique() if s not in ['N/A', 'Erro']]

                    if not sentiments_in_display:
                         st.info("Nenhum sentimento (Positivo/Negativo/Neutro) presente nos dados filtrados para análise de palavras.")
                    else:
                        for sentiment in sentiments_in_display:
                            # Filtra df_display para o sentimento atual
                            sentiment_texts = df_display[df_display['Sentimento_Label'] == sentiment][text_column].dropna().tolist()
                            if sentiment_texts: # Procede apenas se houver textos para o sentimento
                                df_word_counts = get_word_counts(sentiment_texts, stop_words_pt, n_top_words=n_top_words_sentiment)
                                if not df_word_counts.empty:
                                    df_word_counts['Sentimento'] = sentiment # Adiciona coluna de sentimento
                                    combined_word_counts_df = pd.concat([combined_word_counts_df, df_word_counts], ignore_index=True)

                        if not combined_word_counts_df.empty:
                            fig_words_sent = px.bar(combined_word_counts_df, x='Frequência', y='Palavra', color='Sentimento',
                                                    facet_col='Sentimento', facet_col_wrap=3,
                                                    title=f'Top {n_top_words_sentiment} Palavras por Sentimento',
                                                    labels={'Palavra': '', 'Frequência': 'Contagem'},
                                                    height=200 * len(sentiments_in_display), # Ajusta altura
                                                    color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey'})
                            fig_words_sent.update_yaxes(matches=None, showticklabels=True)
                            fig_words_sent.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_words_sent, use_container_width=True)
                        else:
                            st.info("Não foram encontradas palavras frequentes (após filtros e remoção de stop words) para os sentimentos selecionados.")

                else:
                    st.warning("Nenhum dado para exibir com os filtros selecionados.")

            with tab3:
                st.header("Análise de Tópicos (LDA)")
                st.markdown("_Descobre temas recorrentes nos textos **filtrados**. Pode levar um tempo._")

                if not df_display.empty:
                    texts_for_lda = df_display[text_column].dropna().astype(str).tolist()

                    if not texts_for_lda:
                         st.warning("Não há textos válidos nos dados filtrados para a análise de tópicos.")
                    else:
                        num_topics = st.slider("Número de Tópicos a Identificar:", 2, 15, 5, key="num_topics_lda")

                        # Realiza a modelagem de tópicos
                        topics, doc_topic_assignments, lda_model = perform_topic_modeling(texts_for_lda, stop_words_pt, num_topics)

                        if topics is not None and doc_topic_assignments is not None:
                             # Mapeia os assignments de volta para o df_display (cuidado com índices se houve dropna)
                             # Criar uma série com o mesmo índice do df_display que entrou em texts_for_lda
                             topic_series = pd.Series(doc_topic_assignments, index=df_display[text_column].dropna().index)
                             # Adiciona a série ao df_display, preenchendo NaNs onde o texto original era NaN
                             df_display['Tópico_ID'] = topic_series
                             df_display['Tópico_Predito'] = df_display['Tópico_ID'].apply(lambda x: f"Tópico {int(x) + 1}" if pd.notna(x) else "N/A")
                             # Atualiza o session state para que a Tabela de Dados possa mostrar a coluna
                             st.session_state['df_analyzed'].loc[df_display.index, 'Tópico_Predito'] = df_display['Tópico_Predito']


                             st.subheader("Tópicos Identificados e Principais Palavras")
                             st.table(pd.DataFrame(topics))

                             st.subheader("Distribuição dos Tópicos")
                             topic_counts = df_display['Tópico_Predito'].value_counts().reset_index()
                             topic_counts.columns = ['Tópico', 'Contagem']
                             fig_topics = px.bar(topic_counts, x='Tópico', y='Contagem', title="Distribuição de Documentos por Tópico", text_auto=True)
                             fig_topics.update_xaxes(categoryorder='total descending') # Ordena por frequência
                             st.plotly_chart(fig_topics, use_container_width=True)

                             # Cruzamento Tópico x Sentimento
                             st.subheader("Sentimento por Tópico")
                             # Filtra apenas sentimentos válidos para o gráfico
                             df_cross = df_display[df_display['Sentimento_Label'].isin(['Positivo', 'Negativo', 'Neutro'])]
                             if not df_cross.empty:
                                 topic_sentiment_counts = df_cross.groupby(['Tópico_Predito', 'Sentimento_Label']).size().reset_index(name='Contagem')
                                 fig_cross = px.bar(topic_sentiment_counts, x='Tópico_Predito', y='Contagem', color='Sentimento_Label',
                                                     title="Distribuição de Sentimentos Dentro de Cada Tópico", barmode='group', text_auto=True,
                                                     color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey'})
                                 st.plotly_chart(fig_cross, use_container_width=True)
                             else:
                                 st.info("Não há dados com sentimentos Positivo/Negativo/Neutro nos tópicos identificados para exibir o cruzamento.")

                        # else: A função perform_topic_modeling já exibe warnings/errors

                else:
                    st.warning("Nenhum dado para realizar análise de tópicos com os filtros selecionados.")


            with tab4:
                st.header("Nuvem de Palavras")
                st.markdown("_Visualização das palavras mais frequentes nos textos **filtrados**._")

                if not df_display.empty:
                    texts_for_wc = df_display[text_column].dropna().astype(str).tolist()
                    if texts_for_wc:
                        st.subheader("Nuvem Geral (Dados Filtrados)")
                        fig_wc_general = generate_word_cloud(texts_for_wc, stop_words_pt)
                        if fig_wc_general:
                             st.pyplot(fig_wc_general)

                        st.markdown("---")
                        st.subheader("Nuvem por Sentimento Selecionado")
                        # Gera nuvens para cada sentimento selecionado pelo usuário
                        if selected_sentiments: # Verifica se o usuário selecionou algum sentimento válido
                             for sentiment in selected_sentiments:
                                 st.markdown(f"**Sentimento: {sentiment}**")
                                 sentiment_texts_wc = df_display[df_display['Sentimento_Label'] == sentiment][text_column].dropna().astype(str).tolist()
                                 if sentiment_texts_wc:
                                     fig_wc_sentiment = generate_word_cloud(sentiment_texts_wc, stop_words_pt)
                                     if fig_wc_sentiment:
                                         st.pyplot(fig_wc_sentiment)
                                 else:
                                     st.info(f"Não há texto suficiente para gerar a nuvem para o sentimento '{sentiment}' com os filtros atuais.")
                        else:
                             st.info("Selecione sentimentos na barra lateral para ver nuvens específicas.")

                    else:
                         st.warning("Não há textos válidos nos dados filtrados para gerar nuvens de palavras.")
                else:
                    st.warning("Nenhum dado para gerar nuvens de palavras com os filtros selecionados.")


        elif uploaded_file and not text_column:
            st.sidebar.warning("Por favor, selecione a coluna que contém o texto.")
        elif analyze_button and not text_column: # Caso o botão seja clicado sem coluna
             st.sidebar.error("Selecione uma coluna de texto antes de iniciar a análise.")

elif not uploaded_file:
    st.info("Aguardando o upload de um arquivo .csv ou .xlsx na barra lateral.")