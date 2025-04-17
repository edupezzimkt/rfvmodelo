import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Dashboard RFV de Clientes Spazzio")

# Upload do arquivo
df = st.file_uploader("📂 Faça o upload do arquivo CSV de pedidos", type=["csv"])

df = pd.read_csv(df, parse_dates=['pedido_data'])  # certifique-se que a coluna 'pedido_data' está presente

# Remover duplicatas
df.drop_duplicates(subset=['cpf_cnpj', 'pedido_numero'], keep='last', inplace=True)

resumo = df['valor_total_pedido'].describe()

# Extrair valores relevantes
min_val = resumo['min']
q1 = resumo['25%']
q2 = resumo['50%']
q3 = resumo['75%']
max_val = resumo['max']

# Função para cálculo de RFV
def calcular_rfv(df):
    snapshot_date = df['pedido_data'].max() + pd.Timedelta(days=1)
    rfv = df.groupby('cpf_cnpj').agg({
        'pedido_data': lambda x: (snapshot_date - x.max()).days,
        'pedido_numero': 'nunique',
        'valor_total_pedido': 'sum'
    }).reset_index()
    rfv.columns = ['cpf_cnpj', 'recencia', 'frequencia', 'valor']
    return rfv

# Função de segmentação RFV
def segmentar_rfv(df_rfv):
    r_labels = [5, 4, 3, 2, 1]
    f_labels = v_labels = [1, 2, 3, 4, 5]

    # Verificar se o DataFrame tem dados suficientes
    if df_rfv.empty:
        st.error("❌ Nenhum dado disponível para segmentação RFV.")
        return df_rfv

    # Verificar valores únicos
    recencia_unique = df_rfv['recencia'].nunique()
    st.write(f"🔍 Valores únicos em 'recencia': {recencia_unique}")

    # Segmentação R
    if recencia_unique >= 5:
        df_rfv['R_quartil'] = pd.qcut(df_rfv['recencia'], 5, labels=r_labels, duplicates='drop')
    else:
        st.warning("⚠️ Poucos valores únicos em 'recencia' para usar qcut. Atribuindo valor default.")
        df_rfv['R_quartil'] = 3

    # Frequência
    freq_unique = df_rfv['frequencia'].nunique()
    st.write(f"🔍 Valores únicos em 'frequencia': {freq_unique}")
    if freq_unique >= 2:
        df_rfv['F_quartil'] = pd.cut(df_rfv['frequencia'], bins=[0,1,2,3,5,df_rfv['frequencia'].max()],
                                     labels=f_labels, include_lowest=True)
    else:
        df_rfv['F_quartil'] = 3

    # Valor
    valor_unique = df_rfv['valor'].nunique()
    st.write(f"🔍 Valores únicos em 'valor': {valor_unique}")
    if valor_unique >= 2:
        resumo = df_rfv['valor'].describe()
        q1, q2, q3 = resumo['25%'], resumo['50%'], resumo['75%']
        max_val = resumo['max']
        bins = [float('-inf'), q1, q2, q3, (q3 + (q3 - q1)), max_val]
        df_rfv['V_quartil'] = pd.cut(df_rfv['valor'], bins=bins, labels=v_labels, include_lowest=True)
    else:
        df_rfv['V_quartil'] = 3

    # Score RFV
    df_rfv['RFV_score'] = df_rfv['R_quartil'].astype(str) + df_rfv['F_quartil'].astype(str) + df_rfv['V_quartil'].astype(str)

    # Segmentos
    def rotulo_segmento(row):
        try:
            r, f = int(row['R_quartil']), int(row['F_quartil'])
        except:
            return 'Indefinido'
        if r == 5 and f == 5: return 'Campeões'
        elif r >= 4 and f >= 4: return 'Clientes fiéis'
        elif r == 5 and f <= 2: return 'Clientes recentes'
        elif r == 3 and f == 3: return 'Potenciais fiéis'
        elif r == 2 and f == 2: return 'Hibernando'
        elif r <= 2 and f <= 2: return 'Perdidos'
        elif r <= 2 and f >= 4: return 'Não podemos perdê-los'
        elif r == 3 and f <= 2: return 'Prestes a hibernar'
        elif r >= 4 and f == 2: return 'Precisam de atenção'
        elif r >= 3 and f == 1: return 'Em risco'
        else: return 'Outros'

    df_rfv['Segmento'] = df_rfv.apply(rotulo_segmento, axis=1)
    return df_rfv


# Filtro por data
min_data, max_data = df['pedido_data'].min(), df['pedido_data'].max()
data_inicio, data_fim = st.date_input("\U0001F4C5 Selecione o período de pedidos:", [min_data, max_data], format="DD/MM/YYYY")
df = df[(df['pedido_data'] >= pd.to_datetime(data_inicio)) & (df['pedido_data'] <= pd.to_datetime(data_fim))]

# Calcular e segmentar RFV
df_rfv = calcular_rfv(df)
df_rfv = segmentar_rfv(df_rfv)

# Adiciona nome e email ao DataFrame segmentado
df_rfv = df_rfv.merge(
    df[['cpf_cnpj', 'cliente_nome', 'cliente_email']].drop_duplicates(subset='cpf_cnpj'),
    on='cpf_cnpj',
    how='left'
)

# Contagem e percentual de frequência
frequencia_counts = df_rfv['frequencia'].apply(lambda x: str(x) if x <= 5 else '5+')
frequencia_percentual = frequencia_counts.value_counts(normalize=True) * 100
frequencia_percentual = frequencia_percentual.round(2)

# Filtros de RFV no sidebar
with st.sidebar:
    st.header("\U0001F3AF Filtros de RFV")
    rec_min, rec_max = st.slider("Recência (dias)", int(df_rfv['recencia'].min()), int(df_rfv['recencia'].max()), (int(df_rfv['recencia'].min()), int(df_rfv['recencia'].max())))
    freq_min, freq_max = st.slider("Frequência (pedidos)", int(df_rfv['frequencia'].min()), int(df_rfv['frequencia'].max()), (int(df_rfv['frequencia'].min()), int(df_rfv['frequencia'].max())))
    val_min, val_max = st.slider("Valor total (R$)", float(df_rfv['valor'].min()), float(df_rfv['valor'].max()), (float(df_rfv['valor'].min()), float(df_rfv['valor'].max())))

df_filtrado = df_rfv[
    (df_rfv['recencia'].between(rec_min, rec_max)) &
    (df_rfv['frequencia'].between(freq_min, freq_max)) &
    (df_rfv['valor'].between(val_min, val_max))
]

# Ordem correta
labels = ['1', '2', '3', '4', '5', '5+']

st.subheader("📌 Distribuição de Clientes por Frequência de Compra")

col_freqs = st.columns(6)
for i, label in enumerate(labels):
    valor = frequencia_percentual.get(label, 0)
    col_freqs[i].metric(label=f"{label} compra(s)", value=f"{valor:.2f}%")

st.markdown("<hr style='border:1px solid #ccc; margin: 25px 0;'>", unsafe_allow_html=True)

# Métricas principais
col1, col2, col3 = st.columns(3)
col1.metric("Total de clientes", df_filtrado['cpf_cnpj'].nunique())
col2.metric("Pedidos no período", df[df['cpf_cnpj'].isin(df_filtrado['cpf_cnpj'])]['pedido_numero'].nunique())
col3.metric("Faturamento no período", f"R$ {df[df['cpf_cnpj'].isin(df_filtrado['cpf_cnpj'])]['valor_total_pedido'].sum():,.2f}")

st.markdown("<hr style='border:1px solid #ccc; margin: 25px 0;'>", unsafe_allow_html=True)

# Gráficos de distribuição RFV
st.subheader("\U0001F4CA Distribuição RFV")
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(px.histogram(df_filtrado, x='recencia', nbins=20, title="Recência"), use_container_width=True)
with col2:
    st.plotly_chart(px.histogram(df_filtrado, x='frequencia', nbins=20, title="Frequência"), use_container_width=True)
with col3:
    st.plotly_chart(px.histogram(df_filtrado, x='valor', nbins=20, title="Valor"), use_container_width=True)

# Treemap RFV
st.subheader("\U0001F4CC Mapa de Segmentação RFV proporcional (Treemap)")
treemap_df = df_filtrado.groupby('Segmento').agg(Qtd=('cpf_cnpj', 'count')).reset_index()
treemap_df['%'] = (100 * treemap_df['Qtd'] / treemap_df['Qtd'].sum()).round(2)
treemap_df['label'] = treemap_df['%'].astype(str) + "%<br>" + treemap_df['Segmento']

fig = px.treemap(
    treemap_df,
    path=['Segmento'],
    values='Qtd',
    color='Segmento',
    color_discrete_sequence=px.colors.qualitative.Set3,
    custom_data=['%', 'Segmento']
)
fig.update_traces(texttemplate="%{customdata[0]}<br>%{customdata[1]}", textposition="middle center", textfont_size=14)
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=600)
st.plotly_chart(fig, use_container_width=True)


# Tabela de clientes
st.subheader("\U0001F4CB Clientes Segmentados")
st.dataframe(df_filtrado[['cpf_cnpj', 'cliente_nome', 'cliente_email', 'recencia', 'frequencia', 'valor', 'Segmento']])

# Visão geral por segmento
st.subheader("\U0001F4E6 Visão Geral por Segmento")
resumo = df_filtrado.groupby("Segmento").agg({
    "cpf_cnpj": "count",
    "valor": "sum"
}).rename(columns={
    "cpf_cnpj": "Qtd Clientes",
    "valor": "Faturamento"
}).sort_values(by="Faturamento", ascending=False)
st.dataframe(resumo)


# # Heatmap Recência x Frequência
# st.subheader("\U0001F9E9 Matriz RF (Heatmap tradicional)")
# matriz = df_filtrado.groupby(['R_quartil', 'F_quartil']).size().reset_index(name='Clientes')
# matriz_pivot = matriz.pivot(index='F_quartil', columns='R_quartil', values='Clientes')
# fig, ax = plt.subplots(figsize=(5, 3))
# sns.heatmap(matriz_pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar=False, ax=ax)
# ax.set_title("Matriz RF (Recência x Frequência)")
# ax.set_xlabel("Recência")
# ax.set_ylabel("Frequência")
# st.pyplot(fig)

# Agrupamentos
recencia_count = df_rfv['R_quartil'].value_counts().sort_index()
frequencia_count = df_rfv['F_quartil'].value_counts().sort_index()
valor_count = df_rfv['V_quartil'].value_counts().sort_index()

# Criação dos gráficos
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# Recência
axs[0].bar(recencia_count.index, recencia_count.values, color='gold')
axs[0].set_title('Distribuição por Recência')
axs[0].set_xlabel('Recência')
axs[0].set_ylabel('Qtd. Clientes')

# Frequência
axs[1].bar(frequencia_count.index, frequencia_count.values, color='gold')
axs[1].set_title('Distribuição por Frequência')
axs[1].set_xlabel('Frequência')
axs[1].set_ylabel('Qtd. Clientes')

# Valor
axs[2].bar(valor_count.index, valor_count.values, color='gold')
axs[2].set_title('Distribuição por Valor')
axs[2].set_xlabel('Valor')
axs[2].set_ylabel('Qtd. Clientes')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
