import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from collections import Counter
import re
import string
import io # Para exportar Excel

# --- Configuração da Página e Cache ---

st.set_page_config(page_title="Análise de Sentimentos PT-BR", layout="wide")

# Cache para carregar o modelo apenas uma vez
@st.cache_resource
def load_model():
    """Carrega o modelo de análise de sentimentos."""
    try:
        model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        sentiment_pipeline = pipeline("text-classification", model=model_name)
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop() # Interrompe a execução se o modelo não carregar

# Cache para carregar e processar dados (evita reprocessar se o arquivo não mudar)
# Usamos um hash do conteúdo do arquivo como parte da chave de cache
@st.cache_data
def load_data(uploaded_file):
    """Carrega dados de um arquivo .csv ou .xlsx."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado. Use .csv ou .xlsx")
            return None
        # Limpeza básica: remove linhas onde todas as colunas são NaN
        df.dropna(axis=0, how='all', inplace=True)
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

# --- Funções Auxiliares ---

def clean_text(text):
    """Limpa o texto: minúsculas, remove pontuação, números e espaços extras."""
    if not isinstance(text, str):
        return "" # Retorna string vazia se não for texto
    text = text.lower()
    text = re.sub(r'\d+', '', text) # Remove números
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove pontuação
    text = text.strip() # Remove espaços no início/fim
    text = re.sub(r'\s+', ' ', text) # Substitui múltiplos espaços por um único
    return text

def get_word_counts(texts, sentiment_label, stop_words, n_top_words=20):
    """Conta as palavras mais frequentes para um determinado sentimento, excluindo stop words."""
    all_text = ' '.join([clean_text(text) for text in texts if isinstance(text, str)])
    words = all_text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words and len(word) > 2] # Ignora palavras curtas

    if not words:
        return pd.DataFrame(columns=['Palavra', 'Frequência'])

    word_counts = Counter(words)
    most_common = word_counts.most_common(n_top_words)

    df_counts = pd.DataFrame(most_common, columns=['Palavra', 'Frequência'])
    df_counts['Sentimento'] = sentiment_label # Adiciona coluna de sentimento para plotagem
    return df_counts

def analyze_sentiment(text, pipe):
    """Aplica o pipeline de sentimento de forma segura."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {"label": "N/A", "score": 0.0} # Retorna neutro/N/A para texto inválido/vazio
    try:
        # O pipeline retorna uma lista, pegamos o primeiro elemento
        result = pipe(text)[0]
        # Mapear labels para português (se necessário e se o modelo não o fizer)
        # O modelo lxyuan já retorna 'positive', 'negative', 'neutral'
        label_map = {'positive': 'Positivo', 'negative': 'Negativo', 'neutral': 'Neutro'}
        result['label_pt'] = label_map.get(result['label'], result['label']) # Usa o label original se não mapeado
        return result
    except Exception as e:
        # st.warning(f"Erro ao analisar texto: '{text[:50]}...' ({e}). Marcando como N/A.")
        print(f"Erro ao analisar texto: '{text[:50]}...' ({e}). Marcando como N/A.") # Log no console
        return {"label": "Erro", "score": 0.0, "label_pt": "Erro"}

# Função para gerar link de download de Excel (necessário por causa do formato binário)
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='AnaliseSentimento')
    processed_data = output.getvalue()
    return processed_data

# --- Interface do Streamlit ---

st.title("📊 Ferramenta de Análise de Sentimentos (PT-BR)")
st.markdown("""
Carregue um arquivo `.xlsx` ou `.csv` contendo uma coluna com textos (respostas, comentários, etc.)
para realizar a análise de sentimentos.
""")

# Carregar Pipeline (será executado apenas uma vez)
sentiment_pipeline = load_model()

# --- Upload e Preparação dos Dados ---
uploaded_file = st.sidebar.file_uploader("1. Escolha o arquivo (.csv ou .xlsx)", type=['csv', 'xlsx'])

if uploaded_file:
    df_original = load_data(uploaded_file)

    if df_original is not None:
        st.sidebar.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
        st.subheader("Pré-visualização dos Dados Originais")
        st.dataframe(df_original.head())

        # Seleção da Coluna de Texto
        text_column = st.sidebar.selectbox(
            "2. Selecione a coluna com o texto para análise",
            options=[''] + list(df_original.columns), # Adiciona opção vazia
            index=0 # Começa com a opção vazia selecionada
        )

        if text_column:
            # Botão para iniciar análise
            if st.sidebar.button("🚀 Iniciar Análise de Sentimentos", key="analyze_button"):
                if text_column not in df_original.columns:
                     st.error(f"Coluna '{text_column}' não encontrada no arquivo.")
                else:
                    # Verifica se a coluna parece conter texto (heurística simples)
                    if not pd.api.types.is_string_dtype(df_original[text_column].dropna()):
                        st.warning(f"A coluna '{text_column}' não parece ser do tipo texto. A análise pode falhar ou gerar resultados inesperados.")

                    # Realiza a análise
                    progress_bar = st.progress(0, text="Analisando textos...")
                    df_analysis = df_original.copy()
                    total_rows = len(df_analysis)
                    results = []

                    for i, text in enumerate(df_analysis[text_column]):
                        result = analyze_sentiment(text, sentiment_pipeline)
                        results.append(result)
                        # Atualiza a barra de progresso
                        progress_bar.progress((i + 1) / total_rows, text=f"Analisando textos... {i+1}/{total_rows}")

                    progress_bar.empty() # Limpa a barra de progresso

                    # Adiciona os resultados ao DataFrame
                    # Usar pd.json_normalize pode ser mais robusto se a estrutura do dict mudar
                    df_results = pd.DataFrame(results)
                    df_analysis['Sentimento_Label'] = df_results['label_pt']
                    df_analysis['Sentimento_Score'] = df_results['score'].round(4) # Arredonda score

                    # Armazena no estado da sessão para persistir
                    st.session_state['df_analyzed'] = df_analysis
                    st.session_state['text_column'] = text_column # Guarda a coluna analisada
                    st.success("Análise concluída!")

            # --- Exibição dos Resultados e Visualizações ---
            if 'df_analyzed' in st.session_state and st.session_state.get('text_column') == text_column:
                df_analyzed = st.session_state['df_analyzed']

                st.subheader("Resultados da Análise de Sentimentos")
                st.dataframe(df_analyzed[[text_column, 'Sentimento_Label', 'Sentimento_Score']])

                st.markdown("---")
                st.header("📊 Visualizações")

                col1, col2 = st.columns(2)

                with col1:
                    # 1. Gráfico de Distribuição dos Sentimentos
                    st.subheader("Distribuição dos Sentimentos")
                    sentiment_counts = df_analyzed['Sentimento_Label'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentimento', 'Contagem']

                    # Remover 'N/A' e 'Erro' das contagens para o gráfico principal se desejar
                    sentiment_counts_filtered = sentiment_counts[~sentiment_counts['Sentimento'].isin(['N/A', 'Erro'])]

                    if not sentiment_counts_filtered.empty:
                        fig_pie = px.pie(sentiment_counts_filtered, names='Sentimento', values='Contagem',
                                         title="Distribuição Percentual",
                                         color='Sentimento',
                                         color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey'})
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                         st.warning("Não há dados suficientes para gerar o gráfico de distribuição (excluindo N/A e Erros).")

                with col2:
                    # Mostrar contagem exata incluindo N/A e Erro
                    st.subheader("Contagem Total por Sentimento")
                    fig_bar_total = px.bar(sentiment_counts, x='Sentimento', y='Contagem',
                                            title="Contagem Absoluta", text_auto=True,
                                             color='Sentimento',
                                             color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey', 'N/A':'lightblue', 'Erro': 'orange'})
                    st.plotly_chart(fig_bar_total, use_container_width=True)


                # 2. Palavras Mais Frequentes por Sentimento
                st.markdown("---")
                st.subheader("Palavras Mais Frequentes por Sentimento")
                st.markdown("_Excluindo palavras comuns (stop words) e números._")

                # Stop words básicas em Português (pode ser expandido ou usar NLTK/spaCy se disponível)
                # Fonte simples: https://gist.github.com/alopes/5358189
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


                sentiments_to_plot = df_analyzed['Sentimento_Label'].unique()
                sentiments_to_plot = [s for s in sentiments_to_plot if s not in ['N/A', 'Erro']] # Exclui N/A e Erro

                if not sentiments_to_plot:
                    st.warning("Não há categorias de sentimento válidas para exibir contagem de palavras.")
                else:
                    all_word_counts_df = pd.DataFrame()
                    n_top = st.slider("Número de palavras a exibir por sentimento:", 5, 50, 15)

                    for sentiment in sentiments_to_plot:
                        sentiment_texts = df_analyzed[df_analyzed['Sentimento_Label'] == sentiment][text_column]
                        df_word_counts = get_word_counts(sentiment_texts, sentiment, stop_words_pt, n_top_words=n_top)
                        if not df_word_counts.empty:
                             all_word_counts_df = pd.concat([all_word_counts_df, df_word_counts], ignore_index=True)

                    if not all_word_counts_df.empty:
                         # Usar facet_col para criar um gráfico por sentimento
                         fig_words = px.bar(all_word_counts_df, x='Frequência', y='Palavra', color='Sentimento',
                                            facet_col='Sentimento', facet_col_wrap=3, # Ajuste o wrap conforme necessário
                                            title=f'Top {n_top} Palavras Mais Frequentes por Sentimento',
                                            labels={'Palavra': '', 'Frequência': 'Contagem'},
                                            height=250*len(sentiments_to_plot), # Ajusta altura dinamicamente
                                            color_discrete_map={'Positivo':'green', 'Negativo':'red', 'Neutro':'grey'})
                         fig_words.update_yaxes(matches=None, showticklabels=True) # Garante que os eixos Y sejam independentes
                         fig_words.update_layout(yaxis={'categoryorder':'total ascending'}) # Ordena palavras pela frequência
                         st.plotly_chart(fig_words, use_container_width=True)
                    else:
                         st.info("Não foram encontradas palavras frequentes para exibir (após remover stop words e palavras curtas).")


                # --- Exportação ---
                st.markdown("---")
                st.header("📥 Exportar Resultados")

                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                     csv_data = df_analyzed.to_csv(index=False).encode('utf-8')
                     st.download_button(
                         label="Baixar Resultados em CSV",
                         data=csv_data,
                         file_name=f"analise_sentimento_{uploaded_file.name.split('.')[0]}.csv",
                         mime='text/csv',
                     )

                with col_exp2:
                     excel_data = to_excel(df_analyzed)
                     st.download_button(
                         label="Baixar Resultados em Excel (.xlsx)",
                         data=excel_data,
                         file_name=f"analise_sentimento_{uploaded_file.name.split('.')[0]}.xlsx",
                         mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                     )


        elif uploaded_file and not text_column: # Se um arquivo foi carregado mas nenhuma coluna selecionada
            st.sidebar.warning("Por favor, selecione a coluna que contém o texto.")

elif not uploaded_file:
    st.info("Aguardando o upload de um arquivo .csv ou .xlsx na barra lateral.")