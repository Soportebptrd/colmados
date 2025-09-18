import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
import pydeck as pdk
from sklearn.cluster import KMeans
from streamlit_extras.metric_cards import style_metric_cards
from urllib.parse import quote
from fpdf import FPDF
import base64
from io import BytesIO
import tempfile

# =============================================
# 1. SECCIÓN DE AUTENTICACIÓN (AL PRINCIPIO DEL ARCHIVO)
# =============================================

# Configuración de usuarios y contraseñas
USUARIOS = {
    "egeronimo": "1603",
    "jpena": "2025"
}

def check_auth():
    """Verifica si el usuario está autenticado"""
    return st.session_state.get("autenticado", False)

def login():
    """Muestra el formulario de login"""
    st.title("🔐 Acceso al Dashboard")
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Ingresar")
        
        if submit:
            if usuario in USUARIOS and USUARIOS[usuario] == password:
                st.session_state["autenticado"] = True
                st.session_state["usuario"] = usuario
                st.rerun()  # Recarga la app para mostrar el dashboard
            else:
                st.error("❌ Usuario o contraseña incorrectos")

def logout():
    """Cierra la sesión del usuario"""
    st.session_state["autenticado"] = False
    st.session_state["usuario"] = None
    st.rerun()

# =============================================
# 2. VERIFICACIÓN DE AUTENTICACIÓN (ANTES DEL DASHBOARD)
# =============================================
if not check_auth():
    login()
    st.stop()  # Detiene la ejecución si no está autenticado

# =============================================
# 3. EL RESTO DE TU DASHBOARD (CONTENIDO PROTEGIDO)
# =============================================

# ----------------------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------------------
st.set_page_config(
    page_title="🚀 Dashboard Levantamiento de Mercado - Niveo", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stMetric {border: 1px solid #dee2e6; border-radius: 10px; padding: 15px;}
        .stPlotlyChart {border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .stDataFrame {border-radius: 10px;}
        .css-1aumxhk {background-color: #ffffff;}
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0;}
        .stTabs [aria-selected="true"] {background-color: #f0f2f6;}
        .stAlert {border-radius: 10px;}
        .green {color: #28a745;}
        .red {color: #dc3545;}
        .yellow {color: #ffc107;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------
# FUNCIONES DE PROCESAMIENTO DE DATOS
# ----------------------------------------
@st.cache_data(ttl=3600)
def cargar_datos():
    SHEET_ID = "12zKOOHUdDBX7TySNFLzUwulKeCY3gcMdk7sunQHmVqI"
    SHEET_NAME = "Form_responses"
    
    try:
        sheet_encoded = quote(SHEET_NAME)
        url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_encoded}"
        df = pd.read_csv(url, dtype=str)
        
        # Limpiar nombres de columnas (eliminar valores pegados)
        df.columns = [col.split('\n')[0].strip() for col in df.columns]
        
        # Columnas requeridas con nombres alternativos
        required_cols = [
            'SELECCION BARRIO/SECTOR',
            'Geolocalizacion',
            'Mapeado',
            'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO',
            'A QUE PRECIO VENDE CADA UNA DE LAS MARCAS'
        ]
        
        # Verificación flexible de columnas
        for col in required_cols:
            if col not in df.columns:
                # Buscar variaciones del nombre
                col_limpia = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
                encontrada = False
                for col_real in df.columns:
                    if re.sub(r'[^a-zA-Z0-9]', '', col_real.lower()) == col_limpia:
                        # Renombrar la columna al nombre esperado
                        df.rename(columns={col_real: col}, inplace=True)
                        encontrada = True
                        break
                
                if not encontrada:
                    st.warning(f"Columna importante no encontrada: {col}")
        
        return df
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame({
            'Timestamp': [datetime.now()],
            'SELECCION BARRIO/SECTOR': ['24 DE ABRIL'],
            'TIPO DE COLMADO': ['COLMADO PEQ'],
            'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO': ['Niveo, Scott'],
            'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.': ['SEMANAL'],
            'Geolocalizacion': ['Latitude: 19.4080807, Longitude: -70.5333939'],
            'CAMBIO DE NOMBRE?': ['NO'],
            'NOMBRE DEL ENCARGADO DEL NEGOCIO O DUENO': ['Ejemplo'],
            'CUAL ES LA QUE LE DEJA MAYOR BENEFICIO': ['Niveo'],
            'CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.': ['Mucho'],
            'PRECIO DEL PAPEL HIGIENICO NIVEO': [150],
            'PRECIO DEL PAPEL HIGIENICO SCOTT': [160],
            'PRECIO DEL PAPEL HIGIENICO FAMILIA': [155]
        })

def procesar_marcas(texto):
    """Procesa las respuestas múltiples de marcas de forma consistente"""
    if pd.isna(texto):
        return []
    
    # Limpieza básica y estandarización
    texto = str(texto).upper().strip()
    
    # Manejar diferentes separadores (comas, puntos y comas, saltos de línea)
    separadores = r'[,;\n]'
    
    # Procesar cada marca
    marcas = []
    for marca in re.split(separadores, texto):
        marca = marca.strip()
        if marca:
            # Eliminar espacios múltiples y estandarizar
            marca = ' '.join(marca.split())
            marcas.append(marca)
    
    return marcas

def encontrar_columna_exacta(df, nombre_buscado):
    """Busca una columna ignorando mayúsculas, espacios y caracteres especiales"""
    if not isinstance(nombre_buscado, str):
        return None
    
    nombre_buscado_limpio = re.sub(r'[^a-zA-Z0-9]', '', nombre_buscado.lower())
    
    for col in df.columns:
        if not isinstance(col, str):
            continue
        col_limpio = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
        if col_limpio == nombre_buscado_limpio:
            return col
    
    return None

def extract_coordinates(df):
    # Inicializar columnas
    df['Latitud'] = np.nan
    df['Longitud'] = np.nan
    
    # Procesar cada fila
    for idx, row in df.iterrows():
        # Primero intentar con Geolocalizacion
        if pd.notna(row.get('Geolocalizacion')):
            text = str(row['Geolocalizacion'])
            
            # Patrones para extraer coordenadas (formato observado en tus datos)
            patterns = [
                r'Longitude:\s*(-?\d+\.\d+)\s*Latitude:\s*(-?\d+\.\d+)',  # Formato exacto de tus datos
                r'Latitude:\s*(-?\d+\.\d+).*Longitude:\s*(-?\d+\.\d+)',   # Formato alternativo
                r'Longitude:\s*(-?\d+\.\d+).*Latitude:\s*(-?\d+\.\d+)'    # Otro formato posible
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    if 'Latitude' in pattern and pattern.index('Latitude') < pattern.index('Longitude'):
                        lat, lon = match.groups()
                    else:
                        lon, lat = match.groups()
                    
                    try:
                        df.at[idx, 'Latitud'] = float(lat)
                        df.at[idx, 'Longitud'] = float(lon)
                        break  # Si encontramos un match, salimos del bucle
                    except (ValueError, TypeError):
                        continue
        
        # Si no se encontró en Geolocalizacion, intentar con Mapeado
        if pd.isna(df.at[idx, 'Latitud']) and pd.notna(row.get('Mapeado')):
            text = str(row['Mapeado'])
            
            # Extraer coordenadas del formato alternativo en Mapeado
            mapeado_patterns = [
                r'Latitude:\s*(-?\d+\.\d+).*Longitude:\s*(-?\d+\.\d+)',
                r'Longitude:\s*(-?\d+\.\d+).*Latitude:\s*(-?\d+\.\d+)'
            ]
            
            for pattern in mapeado_patterns:
                match = re.search(pattern, text)
                if match:
                    if 'Latitude' in pattern and pattern.index('Latitude') < pattern.index('Longitude'):
                        lat, lon = match.groups()
                    else:
                        lon, lat = match.groups()
                    
                    try:
                        df.at[idx, 'Latitud'] = float(lat)
                        df.at[idx, 'Longitud'] = float(lon)
                        break
                    except (ValueError, TypeError):
                        continue
    
    # Filtrar coordenadas válidas para RD
    dr_mask = (df['Latitud'].between(17, 20)) & (df['Longitud'].between(-72, -68))
    return df[dr_mask].copy()

def extraer_precios(texto_marcas, texto_precios):
    """Extrae los precios correspondientes a cada marca"""
    if pd.isna(texto_marcas) or pd.isna(texto_precios):
        return {}
    
    marcas = procesar_marcas(texto_marcas)
    precios = [p.strip() for p in str(texto_precios).split('\n') if p.strip()]
    
    # Emparejar marcas con precios
    precios_por_marca = {}
    for i, marca in enumerate(marcas):
        if i < len(precios):
            try:
                precio = float(precios[i].replace(',', '.'))
                precios_por_marca[marca] = precio
            except ValueError:
                continue
    
    return precios_por_marca

def preprocesar_datos(df):
    # Limpiar nombres de columnas
    df.columns = [col.strip() for col in df.columns]
    
    # Limpieza específica de la columna de barrio/sector
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        df['SELECCION BARRIO/SECTOR'] = (
            df['SELECCION BARRIO/SECTOR']
            .astype(str)
            .str.strip()
            .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        )
    
    # Procesamiento de fechas
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Fecha'] = df['Timestamp'].dt.date
        df['Mes'] = df['Timestamp'].dt.month_name(locale='es')
        df['Semana'] = df['Timestamp'].dt.isocalendar().week
        df['Dia'] = df['Timestamp'].dt.day_name(locale='es')
        df['Hora'] = df['Timestamp'].dt.hour
    
    # Procesamiento mejorado de geolocalización
    if 'Geolocalizacion' in df.columns or 'Mapeado' in df.columns:
        df = extract_coordinates(df)
    
    # Procesamiento consistente de marcas
    if 'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO' in df.columns:
        # Usar la nueva función de procesamiento
        df['MARCAS_LISTA'] = df['CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO'].apply(procesar_marcas)
        
        # Extraer precios para cada marca
        if 'A QUE PRECIO VENDE CADA UNA DE LAS MARCAS' in df.columns:
            precios_data = []
            for idx, row in df.iterrows():
                marcas = row['MARCAS_LISTA']
                precios_texto = str(row['A QUE PRECIO VENDE CADA UNA DE LAS MARCAS']) if pd.notna(row['A QUE PRECIO VENDE CADA UNA DE LAS MARCAS']) else ""
                
                # Procesar precios (asumiendo que están en el mismo orden que las marcas)
                precios = []
                for p in re.split(r'[\n,;]', precios_texto):
                    try:
                        precio_limpio = re.sub(r'[^\d.]', '', p.replace(',', '.'))
                        if precio_limpio:
                            precios.append(float(precio_limpio))
                    except ValueError:
                        continue
                
                # Emparejar marcas con precios
                for i in range(min(len(marcas), len(precios))):
                    precios_data.append({
                        'ID': idx,
                        'Marca': marcas[i],
                        'Precio': precios[i],
                        'Sector': row.get('SELECCION BARRIO/SECTOR', ''),
                        'Tipo': row.get('TIPO DE COLMADO', '')
                    })
            
            # Crear DataFrame de precios
            df_precios = pd.DataFrame(precios_data)
            
            # Pivotear para tener columnas por marca
            for marca in df_precios['Marca'].unique():
                col_name = f'PRECIO_{marca}'
                df[col_name] = np.nan
                precios_marca = df_precios[df_precios['Marca'] == marca]
                for idx, precio in zip(precios_marca['ID'], precios_marca['Precio']):
                    df.at[idx, col_name] = precio
    
    # Procesamiento de marcas rentables
    col_rentabilidad = encontrar_columna_exacta(df, 'CUAL ES LA QUE LE DEJA MAYOR BENEFICIO')
    if col_rentabilidad:
        df['MARCA_RENTABLE'] = df[col_rentabilidad].apply(
            lambda x: str(x).upper().strip() if pd.notna(x) else np.nan
        )
    
    return df

# ----------------------------------------
# CARGAR Y PROCESAR DATOS
# ----------------------------------------
df = cargar_datos()
df = preprocesar_datos(df)
         
# Crear columna MARCAS_LISTA si no existe
if 'MARCAS_LISTA' not in df.columns and 'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO' in df.columns:
    df['MARCAS_LISTA'] = df['CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO'].apply(
        lambda x: [m.strip().upper() for m in str(x).split(',')] if pd.notna(x) else []
    )

# Verificar si podemos explotar las marcas
if 'MARCAS_LISTA' in df.columns:
    marcas_explotadas = df.explode('MARCAS_LISTA').dropna(subset=['MARCAS_LISTA'])
else:
    st.warning("No se encontró columna de marcas para explotar")
    marcas_explotadas = pd.DataFrame()

# ----------------------------------------
# SIDEBAR CON FILTROS
# ----------------------------------------
with st.sidebar:
    st.title("Filtros Avanzados")
    
    # Filtro de fecha
    if 'Fecha' in df.columns:
        fecha_min = df['Fecha'].min()
        fecha_max = df['Fecha'].max()
        rango_fechas = st.date_input(
            "Rango de fechas",
            [fecha_min, fecha_max],
            min_value=fecha_min,
            max_value=fecha_max
        )
        if len(rango_fechas) == 2:
            df = df[(df['Fecha'] >= rango_fechas[0]) & (df['Fecha'] <= rango_fechas[1])]
    
    # Filtro de sectores con opción "Todos"
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        try:
            # Obtener y limpiar valores únicos
            sectores = [
                str(x).strip() 
                for x in df['SELECCION BARRIO/SECTOR'].dropna().unique()
                if str(x).strip() not in ['nan', 'None', '']
            ]
            
            # Ordenar alfabéticamente
            sectores_ordenados = sorted(sectores, key=lambda x: x.lower())
            
            sector_seleccionado = st.selectbox(
                "Seleccionar sector",
                options=["Todos"] + sectores_ordenados,
                index=0
            )
            
            # Aplicar filtro
            if sector_seleccionado != 'Todos':
                df = df[df['SELECCION BARRIO/SECTOR'].astype(str).str.strip() == sector_seleccionado]
                
        except Exception as e:
            st.error(f"Error al procesar sectores: {str(e)}")
            st.write("Valores problemáticos:", df['SELECCION BARRIO/SECTOR'].unique())
    
    # Filtro de tipo de establecimiento con opción "Todos"
    if 'TIPO DE COLMADO' in df.columns:
        tipos_options = ['Todos'] + sorted(df['TIPO DE COLMADO'].dropna().unique().tolist())
        tipo_seleccionado = st.selectbox(
            "Tipo de establecimiento",
            options=tipos_options,
            index=0
        )
        if tipo_seleccionado != 'Todos':
            df = df[df['TIPO DE COLMADO'] == tipo_seleccionado]
    
    # Filtro por marcas presentes con opción "Todas"
    if not marcas_explotadas.empty:
        marcas_options = ['Todas'] + sorted(marcas_explotadas['MARCAS_LISTA'].dropna().unique().tolist())
        marca_seleccionada = st.selectbox(
            "Filtrar por marcas presentes",
            options=marcas_options,
            index=0
        )
        if marca_seleccionada != 'Todas':
            df = df[df['MARCAS_LISTA'].apply(lambda x: marca_seleccionada in x if isinstance(x, list) else False)]
    
    # Filtro por frecuencia de compra
    if 'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.' in df.columns:
        frecuencias = sorted(df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].dropna().unique().tolist())
        frecuencias_seleccionadas = st.multiselect(
            "Filtrar por frecuencia de compra",
            options=frecuencias,
            default=frecuencias
        )
        if frecuencias_seleccionadas:
            df = df[df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].isin(frecuencias_seleccionadas)]
    
    st.markdown(f"**Datos mostrados:** {len(df)} de {len(cargar_datos())} registros")
    
    if st.button("Resetear filtros"):
        st.rerun()
    
        st.write(f"👤 Usuario: **{st.session_state['usuario']}**")
    if st.button("🚪 Cerrar sesión", type="primary"):
        logout()

# ----------------------------------------
# FUNCIONES AUXILIARES PARA LOS TABS
# ----------------------------------------
def generar_mapa_calor_sectores(df):
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        conteo_sectores = df['SELECCION BARRIO/SECTOR'].value_counts().reset_index()
        conteo_sectores.columns = ['Sector', 'Cantidad']
        
        fig = px.bar(
            conteo_sectores,
            x='Sector',
            y='Cantidad',
            title="Distribución por Sector",
            color='Cantidad',
            color_continuous_scale='Viridis'
        )
        return fig
    return None

def generar_nube_palabras(textos, titulo):
    if textos.empty:
        return None
    
    texto_completo = ' '.join(textos.dropna().astype(str))
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(texto_completo)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(titulo, fontsize=16)
    return fig

def analizar_precios(df):
    precios_data = []
    marcas_posibles = ['NIVEO', 'SCOTT', 'FAMILIA', 'ELITE', 'DOMINO', 'GAVIOTA', 'BINGO', 'HI', 'PETALO', 'SOFT']
    
    for marca in marcas_posibles:
        col_precio = f'PRECIO {marca}'
        if col_precio in df.columns:
            precios = df[col_precio].dropna()
            if not precios.empty:
                precios_data.append({
                    'Marca': marca,
                    'Promedio': precios.mean(),
                    'Mínimo': precios.min(),
                    'Máximo': precios.max(),
                    'Mediana': precios.median(),
                    'Conteo': len(precios)  # Esta es la columna que agregamos
                })
    
    # Verificamos si hay datos antes de crear el DataFrame
    if precios_data:
        df_precios = pd.DataFrame(precios_data)
        # Verificamos si la columna 'Conteo' existe antes de ordenar
        if 'Conteo' in df_precios.columns:
            return df_precios.sort_values('Conteo', ascending=False)
        return df_precios
    return pd.DataFrame() 

# ----------------------------------------
# FUNCIONES PARA RESUMEN EJECUTIVO PDF
# ----------------------------------------
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        # Configuración para mejor compatibilidad
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Resumen Ejecutivo - Levantamiento de Mercado', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        # Reemplazar caracteres problemáticos
        body = body.replace('•', '-').replace('´', "'").replace('`', "'")
        # Manejar encoding para Streamlit Cloud
        try:
            self.multi_cell(0, 8, body)
        except:
            # Alternativa si hay problemas de encoding
            body_safe = body.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 8, body_safe)
        self.ln()

def generar_resumen_ejecutivo(df, marcas_explotadas):
    pdf = PDFReport()
    pdf.add_page()
    
    # Portada
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 40, 'RESUMEN EJECUTIVO', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Levantamiento de Marcas Industrias Nigua', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Fecha de generacion: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(30)
    
    # 1. Resumen Ejecutivo
    pdf.chapter_title('1. Resumen Ejecutivo')
    total_establecimientos = len(df)
    pdf.chapter_body(f"""
    Este reporte presenta los hallazgos clave del levantamiento de mercado realizado para Industrias Nigua, 
    abarcando un total de {total_establecimientos} establecimientos analizados. El estudio revela insights 
    cruciales sobre la presencia de marcas, comportamiento de precios, distribucion geografica y 
    oportunidades de mercado.
    """)
    
    # 2. Hallazgos Principales
    pdf.chapter_title('2. Hallazgos Principales')
    
    # Métricas clave
    metricas_texto = ""
    if 'MARCAS_LISTA' in df.columns and not marcas_explotadas.empty:
        total_marcas = marcas_explotadas['MARCAS_LISTA'].nunique()
        marca_lider = marcas_explotadas['MARCAS_LISTA'].value_counts().index[0]
        penetracion_lider = (marcas_explotadas['MARCAS_LISTA'].value_counts().iloc[0] / total_establecimientos * 100)
        
        sectores_count = df['SELECCION BARRIO/SECTOR'].nunique() if 'SELECCION BARRIO/SECTOR' in df.columns else "N/A"
        
        metricas_texto = f"""
    - Penetracion de mercado: Se identificaron {total_marcas} marcas diferentes en el mercado
    - Marca lider: {marca_lider} con {penetracion_lider:.1f}% de presencia
    - Distribucion geografica: Cobertura en {sectores_count} sectores
        """
    else:
        metricas_texto = "No hay datos suficientes de marcas para el analisis."
    
    pdf.chapter_body(metricas_texto)
    
    # 3. Análisis de Presencia de Marcas
    pdf.chapter_title('3. Analisis de Presencia de Marcas')
    
    if not marcas_explotadas.empty:
        top_5_marcas = marcas_explotadas['MARCAS_LISTA'].value_counts().head(5)
        presencia_text = "\n".join([f"- {marca}: {count} establecimientos ({count/total_establecimientos*100:.1f}%)" 
                                  for marca, count in top_5_marcas.items()])
        
        pdf.chapter_body(f"Top 5 marcas por presencia:\n\n{presencia_text}")
    else:
        pdf.chapter_body("No se encontraron datos de marcas para el analisis.")
    
    # 4. Análisis de Precios
    pdf.chapter_title('4. Analisis de Precios')
    
    # Buscar columnas de precios
    precio_cols = [col for col in df.columns if col.startswith('PRECIO_')]
    if precio_cols:
        precios_info = []
        for col in precio_cols:
            marca = col.replace('PRECIO_', '')
            precios = df[col].dropna()
            if not precios.empty:
                precios_info.append(f"- {marca}: ${precios.mean():.2f} (promedio)")
        
        if precios_info:
            pdf.chapter_body("Precios promedio por marca:\n\n" + "\n".join(precios_info[:5]))  # Limitar a 5 marcas
        else:
            pdf.chapter_body("No se encontraron datos de precios validos.")
    else:
        pdf.chapter_body("No se encontraron columnas de precios en los datos.")
    
    # 5. Recomendaciones Estratégicas
    pdf.chapter_title('5. Recomendaciones Estrategicas')
    pdf.chapter_body("""
    - Incrementar presencia en sectores con baja penetracion
    - Desarrollar estrategias competitivas de precios
    - Fortalecer relacion con distribuidores clave
    - Implementar programas de fidelizacion
    - Realizar seguimiento continuo del mercado
    """)
    
    # 6. Metodología
    pdf.add_page()
    pdf.chapter_title('6. Metodologia')
    pdf.chapter_body(f"""
    Este levantamiento se realizo mediante visitas presenciales a establecimientos en diferentes sectores, 
    utilizando un formulario estructurado para capturar informacion sobre presencia de marcas, precios, 
    frecuencia de compra y percepcion de rentabilidad.
    
    Metodologia:
    - Muestra: {total_establecimientos} establecimientos
    - Alcance: Multiple sectores geograficos
    - Periodo: {datetime.now().strftime("%B %Y")}
    - Tecnica: Entrevistas directas con responsables de establecimientos
    """)
    
    # 7. Limitaciones
    pdf.chapter_title('7. Limitaciones')
    pdf.chapter_body("""
    - Los datos de precios dependen de la precision reportada por los establecimientos
    - La muestra puede no ser representativa de todos los sectores
    - Algunos establecimientos pueden no haber proporcionado informacion completa
    """)
    
    # Guardar PDF en buffer
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# Función alternativa sin gráficos para mayor estabilidad
def generar_reporte_basico(df, marcas_explotadas):
    """Versión simplificada y más robusta del reporte"""
    pdf = PDFReport()
    pdf.add_page()
    
    # Portada
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 40, 'INFORME EJECUTIVO', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Industrias Nigua - Levantamiento de Mercado', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generado el: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'C')
    pdf.ln(30)
    
    # Resumen
    pdf.chapter_title('RESUMEN EJECUTIVO')
    
    total_establecimientos = len(df)
    texto_resumen = f"""
    Este informe presenta los resultados del levantamiento de mercado realizado por Industrias Nigua.
    
    DATOS PRINCIPALES:
    - Total de establecimientos visitados: {total_establecimientos}
    """
    
    if 'SELECCION BARRIO/SECTOR' in df.columns:
        sectores = df['SELECCION BARRIO/SECTOR'].nunique()
        texto_resumen += f"""
    - Sectores cubiertos: {sectores}
        """
    
    if not marcas_explotadas.empty:
        marcas_count = marcas_explotadas['MARCAS_LISTA'].nunique()
        texto_resumen += f"""
    - Marcas identificadas: {marcas_count}
        """
    
    pdf.chapter_body(texto_resumen)
    
    # Hallazgos clave
    pdf.add_page()
    pdf.chapter_title('HALLAZGOS PRINCIPALES')
    
    hallazgos = """
    1. DISTRIBUCION DE MARCAS
    - Presencia de multiples competidores en el mercado
    - Oportunidades de crecimiento identificadas
    - Variabilidad en la distribucion por sectores
    
    2. COMPORTAMIENTO DE PRECIOS
    - Rango de precios diverso entre competidores
    - Oportunidades de posicionamiento competitivo
    - Potencial para estrategias de valor agregado
    
    3. COBERTURA GEOGRAFICA
    - Amplia distribucion en multiple sectores
    - Areas de oportunidad para expansion
    - Potencial para fortalecer presencia
    """
    
    pdf.chapter_body(hallazgos)
    
    # Recomendaciones
    pdf.chapter_title('RECOMENDACIONES ESTRATEGICAS')
    
    recomendaciones = """
    1. EXPANSION DE PRESENCIA
    - Fortalecer distribucion en sectores sub-atendidos
    - Desarrollar programas de incentivos para puntos de venta
    - Implementar estrategias de visual merchandising
    
    2. OPTIMIZACION DE PRECIOS
    - Analizar estructura competitiva de precios
    - Desarrollar promociones y bundles estrategicos
    - Implementar estrategias de valor percibido
    
    3. FORTALECIMIENTO OPERATIVO
    - Mejorar logistica de distribucion
    - Fortalecer relacion con distribuidores clave
    - Implementar sistema de monitoreo continuo
    """
    
    pdf.chapter_body(recomendaciones)
    
    # Guardar PDF
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    return pdf_buffer

def generar_resumen_ejecutivo_completo(df, marcas_explotadas):
    """Genera un resumen ejecutivo completo que responde a las preguntas del cuestionario"""
    pdf = PDFReport()
    pdf.add_page()
    
    # Portada
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 40, 'RESUMEN EJECUTIVO - LEVANTAMIENTO DE MERCADO', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Industrias Nigua - Análisis de Canales', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generado el: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.ln(30)
    
    total_establecimientos = len(df)
    
    # 1. RESPUESTAS A LAS PREGUNTAS DEL CUESTIONARIO
    pdf.chapter_title('RESPUESTAS A LAS PREGUNTAS DEL CUESTIONARIO')
    
    # Pregunta 1: ¿Cuáles marcas están presentes?
    pdf.chapter_title('1. PRESENCIA DE MARCAS EN EL MERCADO')
    if not marcas_explotadas.empty:
        top_10 = marcas_explotadas['MARCAS_LISTA'].value_counts().head(10)
        respuesta_1 = "Las marcas con mayor presencia en el mercado son:\n\n"
        for marca, count in top_10.items():
            porcentaje = (count / total_establecimientos) * 100
            respuesta_1 += f"• {marca}: {count} establecimientos ({porcentaje:.1f}% de penetración)\n"
        
        # Presencia específica de NIVEO
        niveo_presente = 'NIVEO' in marcas_explotadas['MARCAS_LISTA'].values
        if niveo_presente:
            niveo_count = (marcas_explotadas['MARCAS_LISTA'] == 'NIVEO').sum()
            niveo_porcentaje = (niveo_count / total_establecimientos) * 100
            respuesta_1 += f"\nNIVEO se encuentra en {niveo_count} establecimientos ({niveo_porcentaje:.1f}% del mercado)"
        
        pdf.chapter_body(respuesta_1)
    
    # Pregunta 2: Precios de venta
    pdf.add_page()
    pdf.chapter_title('2. ESTRUCTURA DE PRECIOS POR MARCA')
    
    precio_cols = [col for col in df.columns if col.startswith('PRECIO_')]
    if precio_cols:
        respuesta_2 = "Los precios promedio de venta reportados son:\n\n"
        for col in precio_cols:
            marca = col.replace('PRECIO_', '')
            precios = df[col].dropna()
            if not precios.empty:
                respuesta_2 += f"• {marca}: ${precios.mean():.2f} (rango: ${precios.min():.2f} - ${precios.max():.2f})\n"
        
        pdf.chapter_body(respuesta_2)
    
    # Pregunta 3: Marca más rentable
    pdf.chapter_title('3. RENTABILIDAD PERCIBIDA POR MARCA')
    
    # Buscar columna de rentabilidad con diferentes nombres posibles
    col_rentabilidad = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['BENEFICIO', 'RENTABILIDAD', 'MAYOR GANANCIA']):
            col_rentabilidad = col
            break
    
    if col_rentabilidad and col_rentabilidad in df.columns:
        rentabilidad = df[col_rentabilidad].value_counts()
        respuesta_3 = "Las marcas consideradas más rentables por los establecimientos:\n\n"
        for marca, count in rentabilidad.items():
            if pd.notna(marca) and str(marca).strip() not in ['', 'nan', 'None']:
                porcentaje = (count / rentabilidad.sum()) * 100
                respuesta_3 += f"• {marca}: {count} menciones ({porcentaje:.1f}%)\n"
        
        pdf.chapter_body(respuesta_3)
    
    # Pregunta 4: Razones de rentabilidad
    pdf.chapter_title('4. RAZONES DE RENTABILIDAD')
    
    # Buscar columna de "por qué" con diferentes nombres
    col_porque = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['POR QUE', 'RAZON', 'MOTIVO', 'EXPLICACION']):
            col_porque = col
            break
    
    if col_porque and col_porque in df.columns:
        razones = df[col_porque].dropna()
        if not razones.empty:
            # Análisis de razones más comunes
            razones_comunes = razones.value_counts().head(5)
            respuesta_4 = "Principales razones mencionadas para la rentabilidad:\n\n"
            for razon, count in razones_comunes.items():
                respuesta_4 += f"• {razon}: {count} menciones\n"
            
            pdf.chapter_body(respuesta_4)
    
    # Pregunta 5: Proveedores y canales de compra
    pdf.add_page()
    pdf.chapter_title('5. CANALES DE DISTRIBUCIÓN Y PROVEEDORES')
    
    # Buscar columna de proveedores
    col_proveedores = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['PROVEEDOR', 'DONDE COMPRA', 'A QUIEN COMPRA', 'DISTRIBUIDOR']):
            col_proveedores = col
            break
    
    if col_proveedores and col_proveedores in df.columns:
        proveedores = df[col_proveedores].dropna()
        if not proveedores.empty:
            top_proveedores = proveedores.value_counts().head(8)
            respuesta_5 = "Principales proveedores y canales de compra mencionados:\n\n"
            for proveedor, count in top_proveedores.items():
                respuesta_5 += f"• {proveedor}: {count} menciones\n"
            
            pdf.chapter_body(respuesta_5)
    
    # Pregunta 6: Frecuencia de compra
    pdf.chapter_title('6. FRECUENCIA DE COMPRA')
    
    # Buscar columna de frecuencia
    col_frecuencia = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['FRECUENCIA', 'CADA CUANTO COMPRA', 'CON QUE FRECUENCIA']):
            col_frecuencia = col
            break
    
    if col_frecuencia and col_frecuencia in df.columns:
        frecuencia = df[col_frecuencia].value_counts()
        respuesta_6 = "Frecuencia de compra reportada por los establecimientos:\n\n"
        for freq, count in frecuencia.items():
            if pd.notna(freq):
                porcentaje = (count / frecuencia.sum()) * 100
                respuesta_6 += f"• {freq}: {count} establecimientos ({porcentaje:.1f}%)\n"
        
        pdf.chapter_body(respuesta_6)
    
    # Pregunta 7: Influencia del vendedor
    pdf.add_page()
    pdf.chapter_title('7. INFLUENCIA EN DECISIONES DE COMPRA')
    
    # Buscar columna de influencia
    col_influencia = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['INFLUYE', 'INFLUENCIA', 'RECOMENDACION', 'VENDEDOR']):
            col_influencia = col
            break
    
    if col_influencia and col_influencia in df.columns:
        influencia = df[col_influencia].value_counts()
        respuesta_7 = "Nivel de influencia percibida por los vendedores:\n\n"
        for nivel, count in influencia.items():
            if pd.notna(nivel):
                porcentaje = (count / influencia.sum()) * 100
                respuesta_7 += f"• {nivel}: {count} vendedores ({porcentaje:.1f}%)\n"
        
        pdf.chapter_body(respuesta_7)
    
    # Pregunta 8: Compra por aplicación digital
    pdf.chapter_title('8. COMPRA MEDIANTE APLICACIÓN DIGITAL')
    
    # Buscar columna de compra digital
    col_digital = None
    for col in df.columns:
        if any(keyword in col.upper() for keyword in ['APLICACION', 'DIGITAL', 'APP', 'PLATAFORMA']):
            col_digital = col
            break
    
    if col_digital and col_digital in df.columns:
        digital = df[col_digital].value_counts()
        respuesta_8 = "Uso de aplicaciones digitales para compras:\n\n"
        for respuesta, count in digital.items():
            if pd.notna(respuesta):
                porcentaje = (count / digital.sum()) * 100
                respuesta_8 += f"• {respuesta}: {count} establecimientos ({porcentaje:.1f}%)\n"
        
        # Nombres de aplicaciones específicas mencionadas
        if 'SI' in digital.index:
            # Buscar menciones de nombres de apps
            apps_mentions = []
            for respuesta in df[col_digital].dropna():
                if 'SI' in str(respuesta).upper():
                    # Extraer nombres de apps si se mencionan
                    if any(app_keyword in str(respuesta).upper() for app_keyword in ['PEDIDOS', 'APP', 'MERCADO', 'UBER', 'RAPPI']):
                        apps_mentions.append(str(respuesta))
            
            if apps_mentions:
                respuesta_8 += f"\nAplicaciones mencionadas: {', '.join(set(apps_mentions[:5]))}"
        
        pdf.chapter_body(respuesta_8)
    
    # 2. RECOMENDACIONES ESTRATÉGICAS
    pdf.add_page()
    pdf.chapter_title('RECOMENDACIONES ESTRATÉGICAS BASADAS EN LOS HALLAZGOS')
    
    recomendaciones = """
    1. ESTRATEGIA DE PENETRACIÓN DE MERCADO
    • Incrementar presencia de NIVEO en sectores con baja penetración
    • Desarrollar programas de incentivos para distribuidores
    • Implementar estrategias de visual merchandising en puntos de venta
    
    2. OPTIMIZACIÓN DE PRECIOS
    • Analizar competitividad de precios frente a marcas líderes
    • Desarrollar estrategias de valor agregado
    • Considerar promociones temporales en sectores competitivos
    
    3. FORTALECIMIENTO DE DISTRIBUCIÓN
    • Identificar y fortalecer relación con proveedores clave
    • Desarrollar programa de capacitación para vendedores
    • Implementar sistema de monitoreo de inventarios
    
    4. ESTRATEGIA DIGITAL
    • Evaluar oportunidades en plataformas digitales de pedidos
    • Desarrollar aplicación propia si es viable
    • Capacitar distribuidores en uso de tecnologías digitales
    
    5. CAPITALIZACIÓN DE INFLUENCIA
    • Desarrollar programa de incentivos para vendedores
    • Crear material de capacitación sobre beneficios de NIVEO
    • Implementar sistema de reconocimiento por ventas
    """
    
    pdf.chapter_body(recomendaciones)
    
    # 3. METODOLOGÍA
    pdf.add_page()
    pdf.chapter_title('METODOLOGÍA DEL ESTUDIO')
    
    metodologia = f"""
    ALCANCE DEL ESTUDIO:
    • Muestra: {total_establecimientos} establecimientos comerciales
    • Cobertura: Múltiples sectores geográficos
    • Método: Entrevistas presenciales con cuestionario estructurado
    • Periodo: {df['Timestamp'].dt.date.min() if 'Timestamp' in df.columns else 'N/A'} al {df['Timestamp'].dt.date.max() if 'Timestamp' in df.columns else 'N/A'}
    
    VARIABLES ANALIZADAS:
    • Presencia y distribución de marcas
    • Estructura de precios por marca
    • Percepción de rentabilidad y razones
    • Canales de distribución y proveedores
    • Frecuencia de compra
    • Influencia en punto de venta
    • Uso de plataformas digitales
    
    LIMITACIONES:
    • Los datos dependen de la veracidad de las respuestas
    • La muestra puede no ser representativa de todos los sectores
    • Variabilidad en la calidad de la información proporcionada
    """
    
    pdf.chapter_body(metodologia)
    
    # SOLUCIÓN PARA STREAMLIT CLOUD - Método alternativo
    try:
        # Método 1: Usar tempfile (más compatible)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, 'rb') as f:
                pdf_buffer = BytesIO(f.read())
        pdf_buffer.seek(0)
        return pdf_buffer
        
    except Exception as e:
        # Método 2: Alternativa de respaldo
        try:
            pdf_buffer = BytesIO()
            pdf_output = pdf.output(dest='S')  # Sin encode
            if isinstance(pdf_output, str):
                pdf_buffer.write(pdf_output.encode('latin-1'))
            else:
                pdf_buffer.write(pdf_output)
            pdf_buffer.seek(0)
            return pdf_buffer
        except:
            # Método 3: Último recurso
            pdf_buffer = BytesIO()
            pdf_output = pdf.output()
            pdf_buffer.write(pdf_output)
            pdf_buffer.seek(0)
            return pdf_buffer

# ----------------------------------------
# INTERFAZ PRINCIPAL CON 10 TABS
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Resumen General", 
    "🗺️ Análisis Geográfico", 
    "🏷️ Presencia de Marca", 
    "💰 Precios y Rentabilidad", 
    "🔁 Frecuencia de Compra",
    "💬 Influencia del Vendedor",
    "📦 Proveedores y Canales",
    "📝 Respuestas Abiertas",
    "📋 Resumen Ejecutivo"
])

# ----------------------------------------
# TAB 1: RESUMEN GENERAL
# ----------------------------------------
with tab1:
    st.header("📊 Resumen General del Mercado", divider="rainbow")
    
    # Métricas clave
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total levantamientos", df.shape[0])
    
    if 'Timestamp' in df.columns:
        ultima_fecha = df['Timestamp'].max().strftime('%d/%m/%Y')
        col2.metric("🕒 Último levantamiento", ultima_fecha)
    else:
        col2.metric("🕒 Último levantamiento", "No disponible")
    
    if 'CAMBIO DE NOMBRE?' in df.columns:
        cambios = df['CAMBIO DE NOMBRE?'].value_counts(normalize=True)
        if 'SI' in cambios.index:
            porcentaje_cambio = cambios['SI'] * 100
            col3.metric("🔄 % Cambio de nombre", f"{porcentaje_cambio:.1f}%")
        else:
            col3.metric("🔄 % Cambio de nombre", "0%")
    else:
        col3.metric("🔄 % Cambio de nombre", "No disponible")
    
    if not marcas_explotadas.empty:
        total_marcas = marcas_explotadas['MARCAS_LISTA'].nunique()
        col4.metric("🏷️ Marcas identificadas", total_marcas)
    else:
        col4.metric("🏷️ Marcas identificadas", "No disponible")
    
    style_metric_cards()
    
    # Mapa de calor por sector
    st.subheader("🌡️ Distribución por Sector", divider="gray")
    fig_mapa_calor = generar_mapa_calor_sectores(df)
    if fig_mapa_calor:
        st.plotly_chart(fig_mapa_calor, use_container_width=True, key='chart_0')
    
    # Distribución por tipo de establecimiento
    st.subheader("🏪 Distribución por Tipo de Establecimiento", divider="gray")
    if 'TIPO DE COLMADO' in df.columns:
        tipo_dist = df['TIPO DE COLMADO'].value_counts().reset_index()
        tipo_dist.columns = ['Tipo', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                tipo_dist, 
                values='Cantidad', 
                names='Tipo',
                title="Distribución por Tipo",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_1')
        
        with col2:
            fig = px.bar(
                tipo_dist,
                x='Tipo',
                y='Cantidad',
                title="Cantidad por Tipo",
                color='Tipo'
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_2')
    
        # Top marcas
    st.subheader("🏷️ Top Marcas en el Mercado", divider="gray")
    if 'MARCAS_LISTA' in df.columns:
        # Explotar la lista de marcas
        marcas_explotadas = df.explode('MARCAS_LISTA')
        top_marcas = marcas_explotadas['MARCAS_LISTA'].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                top_marcas,
                x=top_marcas.index,
                y=top_marcas.values,
                title="Top 10 Marcas más Comunes",
                color=top_marcas.values,
                labels={'x': 'Marca', 'y': 'Cantidad'}
            )
            st.plotly_chart(fig, use_container_width=True, key="top_marcas_bar")
        
        with col2:
            # Análisis Pareto
            total_menciones = top_marcas.sum()
            pareto = (top_marcas.cumsum() / total_menciones * 100).reset_index()
            pareto.columns = ['Marca', 'Porcentaje Acumulado']
            
            fig = px.line(
                pareto,
                x='Marca',
                y='Porcentaje Acumulado',
                title="Análisis Pareto de Marcas",
                markers=True
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80%")
            st.plotly_chart(fig, use_container_width=True, key="pareto_marcas")

# ----------------------------------------
# TAB 2: ANÁLISIS GEOGRÁFICO
# ----------------------------------------
with tab2:
    st.header("🗺️ Análisis Geográfico", divider="rainbow")
    
    # Aplicar extracción mejorada
    if 'Geolocalizacion' in df.columns or 'Mapeado' in df.columns:
        df_geo = extract_coordinates(df)
        valid_locations = len(df_geo.dropna(subset=['Latitud', 'Longitud']))
        
        # Mostrar diagnóstico
        st.subheader("Diagnóstico de Coordenadas")
        col1, col2 = st.columns(2)
        col1.metric("Registros totales", len(df))
        col2.metric("Ubicaciones válidas", valid_locations)
        
        if valid_locations > 0:
            # Configuración del mapa con vista centrada
            avg_lat = df_geo['Latitud'].mean()
            avg_lon = df_geo['Longitud'].mean()
            
            view_state = pdk.ViewState(
                latitude=avg_lat,
                longitude=avg_lon,
                zoom=11,  # Zoom un poco más cercano
                pitch=40  # Inclinación moderada
            )
            
            # Capa del mapa con puntos ROJOS y más visibles
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_geo,
                get_position=['Longitud', 'Latitud'],
                get_color=[255, 0, 0, 200],  # Rojo sólido [R, G, B, Alpha]
                get_radius=100,  # Tamaño un poco mayor para mejor visibilidad
                pickable=True,
                stroked=True,
                filled=True,
                extruded=False,
                radius_min_pixels=3,  # Tamaño mínimo en píxeles
                radius_max_pixels=15   # Tamaño máximo en píxeles
            )
            
            # Tooltip mejorado
            tooltip = {
                "html": """
                    <div style="background-color: white; padding: 10px; border-radius: 5px;">
                        <b>Establecimiento:</b> {NOMBRE DEL ENCARGADO DEL NEGOCIO O DUENO}<br/>
                        <b>Ubicación:</b> {Latitud:.6f}, {Longitud:.6f}<br/>
                        <b>Sector:</b> {SELECCION BARRIO/SECTOR}<br/>
                        <b>Tipo:</b> {TIPO DE COLMADO}
                    </div>
                """,
                "style": {
                    "backgroundColor": "white",
                    "color": "black",
                    "fontFamily": '"Helvetica Neue", Arial'
                }
            }
            
            # Mostrar mapa con estilo claro y puntos rojos
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",  # Fondo claro para mejor contraste
                height=600          # Altura mayor para mejor visualización
            ))
            
            # # Mostrar estadísticas de distribución
            # st.subheader("Distribución Geográfica", divider="gray")
            # col1, col2 = st.columns(2)
            
            # with col1:
            #     st.markdown("**Coordenas extremas:**")
            #     st.write(f"📍 Norte: {df_geo['Latitud'].max():.4f}°")
            #     st.write(f"📍 Sur: {df_geo['Latitud'].min():.4f}°")
            #     st.write(f"📍 Este: {df_geo['Longitud'].max():.4f}°")
            #     st.write(f"📍 Oeste: {df_geo['Longitud'].min():.4f}°")
            
            # with col2:
            #     st.markdown("**Centro geográfico:**")
            #     st.write(f"📌 Latitud promedio: {avg_lat:.4f}°")
            #     st.write(f"📌 Longitud promedio: {avg_lon:.4f}°")
            #     st.write(f"📌 Total establecimientos: {valid_locations}")
            
            # Análisis por sector
            if 'SELECCION BARRIO/SECTOR' in df_geo.columns:
                st.subheader("Filtrar por Sector", divider="gray")
                if 'SELECCION BARRIO/SECTOR' in df_geo.columns:
                    sectores = [
                        str(x).strip() 
                        for x in df_geo['SELECCION BARRIO/SECTOR'].dropna().unique()
                        if str(x).strip() not in ['nan', 'None', '']
                    ]
                    sector = st.selectbox(
                        "Seleccionar sector para ver en el mapa",
                        ["Todos"] + sorted(sectores, key=lambda x: x.lower())
                    )
                
                if sector != "Todos":
                    df_sector = df_geo[df_geo['SELECCION BARRIO/SECTOR'] == sector]
                    
                    # Mapa solo del sector seleccionado
                    sector_view = pdk.ViewState(
                        latitude=df_sector['Latitud'].mean(),
                        longitude=df_sector['Longitud'].mean(),
                        zoom=13,  # Más zoom para el sector
                        pitch=45
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        layers=[pdk.Layer(
                            "ScatterplotLayer",
                            data=df_sector,
                            get_position=['Longitud', 'Latitud'],
                            get_color=[255, 0, 0, 200],  # Rojo
                            get_radius=120,  # Puntos más grandes
                            pickable=True
                        )],
                        initial_view_state=sector_view,
                        tooltip=tooltip,
                        map_style="light",
                        height=500
                    ))
                    
                    # Mostrar tabla de establecimientos en el sector
                    st.dataframe(
                        df_sector[['NOMBRE DEL ENCARGADO DEL NEGOCIO O DUENO', 
                                  'TIPO DE COLMADO', 
                                  'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO']],
                        height=300
                    )   
                
            # Análisis por sector
            if 'SELECCION BARRIO/SECTOR' in df_geo.columns:
                st.subheader("Distribución por Sector", divider="gray")
                
                # Gráfico de distribución
                sector_counts = df_geo['SELECCION BARRIO/SECTOR'].value_counts().reset_index()
                sector_counts.columns = ['Sector', 'Establecimientos']
                
                fig = px.bar(
                    sector_counts,
                    x='Sector',
                    y='Establecimientos',
                    title="Establecimientos por Sector",
                    color='Establecimientos',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True, key='chart_3')
        else:
            st.error("No se encontraron ubicaciones válidas para mostrar el mapa")
            
            # Panel de diagnóstico avanzado
            with st.expander("🔍 Diagnóstico detallado"):
                st.write("**Datos crudos de geolocalización (primeras 5 filas):**")
                st.dataframe(df[['Geolocalizacion', 'Mapeado']].head())
                
                st.write("**Problemas detectados:**")
                
                # Analizar problemas comunes
                if 'Geolocalizacion' in df.columns:
                    geoloc_vacios = df['Geolocalizacion'].isna().sum()
                    st.write(f"- Campos 'Geolocalizacion' vacíos: {geoloc_vacios}/{len(df)}")
                
                if 'Mapeado' in df.columns:
                    mapeado_vacios = df['Mapeado'].isna().sum()
                    st.write(f"- Campos 'Mapeado' vacíos: {mapeado_vacios}/{len(df)}")
                
                st.write("**Solución recomendada:**")
                st.markdown("""
                1. Verificar que los datos tengan el formato correcto:
                   - Ejemplo: `"Longitude: -70.5334011 Latitude: 19.4081105"`
                2. Si los datos vienen de un formulario, revisar la plantilla de recolección
                3. Para datos existentes, considerar una limpieza previa
                """)
    else:
        st.error("El dataset no contiene columnas de geolocalización ('Geolocalizacion' o 'Mapeado')")

# ----------------------------------------
# TAB 3: PRESENCIA DE MARCA
# ----------------------------------------
with tab3:
    st.header("🏷️ Presencia de Marca", divider="rainbow")
    
    if 'MARCAS_LISTA' in df.columns:
        marcas_explotadas = df.explode('MARCAS_LISTA')
        marcas_disponibles = sorted(marcas_explotadas['MARCAS_LISTA'].dropna().unique())
        
        # Selector de marcas para comparar
        marcas_seleccionadas = st.multiselect(
            "Seleccionar Marcas para Comparar", 
            marcas_disponibles,
            default=marcas_disponibles[:3] if len(marcas_disponibles) >= 3 else marcas_disponibles
        )
        
        if marcas_seleccionadas:
            # Presencia por sector
            st.subheader("📊 Presencia por Sector", divider="gray")
            
            # Filtrar solo las marcas seleccionadas
            df_filtrado = marcas_explotadas[marcas_explotadas['MARCAS_LISTA'].isin(marcas_seleccionadas)]
            
            # Crear tabla cruzada
            presencia_sector = pd.crosstab(
                df_filtrado['SELECCION BARRIO/SECTOR'],
                df_filtrado['MARCAS_LISTA']
            )
            
            if not presencia_sector.empty:
                fig = px.bar(
                    presencia_sector,
                    barmode='group',
                    title="Presencia de Marcas por Sector",
                    labels={'value': 'Cantidad', 'variable': 'Marca'}
                )
                st.plotly_chart(fig, use_container_width=True, key=f"presencia_sector_{'_'.join(marcas_seleccionadas)}")
            
            # Luego filtramos solo las marcas seleccionadas que existen en la tabla
            marcas_existentes = [marca for marca in marcas_seleccionadas if marca in presencia_sector.columns]
            
            if marcas_existentes:
                presencia_sector = presencia_sector[marcas_existentes]
                
                # Tabla dinámica
                st.subheader("📋 Tabla Dinámica de Presencia", divider="gray")
                st.dataframe(
                    presencia_sector.style.background_gradient(cmap='Blues'),
                    use_container_width=True
                )
            else:
                st.warning("Ninguna de las marcas seleccionadas se encontró en los datos")

            # Penetración de mercado
            st.subheader("📈 Penetración de Mercado", divider="gray")
            total_establecimientos = len(df)
            penetracion_data = []
            
            for marca in marcas_seleccionadas:
                presencia = (marcas_explotadas['MARCAS_LISTA'] == marca).sum()
                penetracion = presencia / total_establecimientos * 100
                penetracion_data.append({
                    'Marca': marca,
                    'Establecimientos': presencia,
                    'Penetración (%)': penetracion
                })
            
            df_penetracion = pd.DataFrame(penetracion_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_penetracion,
                    x='Marca',
                    y='Penetración (%)',
                    title="Penetración de Mercado",
                    color='Marca',
                    text='Penetración (%)'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True, key='chart_6')
            
            with col2:
                st.dataframe(
                    df_penetracion.style.format({'Penetración (%)': '{:.1f}%'}).background_gradient(
                        subset=['Penetración (%)'], 
                        cmap='Greens'
                    ),
                    use_container_width=True
                )
            
                # Mapa de presencia
                st.subheader("🗺️ Mapa de Presencia", divider="gray")
                if 'Latitud' in df.columns and 'Longitud' in df.columns and not df[['Latitud', 'Longitud']].dropna().empty:
                    marca_mapa = st.selectbox(
                        "Seleccionar marca para mapa",
                        marcas_seleccionadas,
                        index=0
                    )
                    
                    df_marca = df[df['MARCAS_LISTA'].apply(lambda x: marca_mapa in x if isinstance(x, list) else False)]
                    df_marca_geo = df_marca.dropna(subset=['Latitud', 'Longitud'])
                    
                    if not df_marca_geo.empty:
                        view_state_marca = pdk.ViewState(
                            latitude=df_marca_geo['Latitud'].mean(),
                            longitude=df_marca_geo['Longitud'].mean(),
                            zoom=12,
                            pitch=50
                        )
                        
                        layer_marca = pdk.Layer(
                            "ScatterplotLayer",
                            data=df_marca_geo,
                            get_position=['Longitud', 'Latitud'],
                            get_color='[0, 100, 200, 160]',
                            get_radius=150,
                            pickable=True
                        )
                        
                        st.pydeck_chart(pdk.Deck(
                            layers=[layer_marca],
                            initial_view_state=view_state_marca,
                            tooltip=tooltip
                        ))
                    else:
                        st.warning(f"No hay datos geográficos para {marca_mapa}")
    else:
        st.warning("No hay datos de marcas disponibles para el análisis")
        
        # Mostrar alternativa usando la columna original
        if 'CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO' in df.columns:
            with st.expander("Ver datos crudos de marcas"):
                st.write("Ejemplo de datos de marcas:")
                st.write(df['CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO'].head())

# ----------------------------------------
# TAB 4: PRECIOS Y RENTABILIDAD
# ----------------------------------------
with tab4:
    st.header("💰 Precios y Rentabilidad", divider="rainbow")
    
    # ------------------------------------------------------------
    # 1. DEFINICIÓN ROBUSTA DE COLUMNAS (CON RESPALDO POR ÍNDICES)
    # ------------------------------------------------------------
    try:
        # Intenta primero con búsqueda inteligente
        def encontrar_columna(df, nombre_buscado):
            nombre_limpio = re.sub(r'[^a-zA-Z0-9]', '', str(nombre_buscado).lower())
            for col in df.columns:
                col_limpia = re.sub(r'[^a-zA-Z0-9]', '', str(col).lower())
                if col_limpia == nombre_limpio:
                    return col
            return None

        columna_marcas = encontrar_columna(df, "CUALES MARCAS ESTAN PRESENTES EN EL ESTABLECIMIENTO") or df.columns[6]
        columna_precios = encontrar_columna(df, "A QUE PRECIO VENDE CADA UNA DE LAS MARCAS") or df.columns[8]
        
        # Verificación final
        if columna_marcas not in df.columns or columna_precios not in df.columns:
            raise KeyError("Columnas críticas no encontradas")

    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        st.error("Columnas disponibles:")
        st.write(list(df.columns))
        st.stop()

    # ------------------------------------------------------------
    # 2. FUNCIÓN DE PROCESAMIENTO MEJORADA (CON MANEJO DE ERRORES)
    # ------------------------------------------------------------
    def procesar_precios_definitivo(df, col_marcas, col_precios):
        resultados = []
        problemas = []
        
        for idx, row in df.iterrows():
            try:
                # Extracción segura de valores
                marcas_raw = str(row[col_marcas]) if pd.notna(row[col_marcas]) else ""
                precios_raw = str(row[col_precios]) if pd.notna(row[col_precios]) else ""
                
                if not marcas_raw or not precios_raw:
                    continue
                
                # Procesamiento de marcas
                marcas = [m.strip().upper() for m in re.split(r'[,;\n]', marcas_raw) if m.strip()]
                
                # Procesamiento de precios
                precios = []
                for p in re.split(r'[\n,;]', precios_raw):
                    try:
                        precio_limpio = re.sub(r'[^\d.]', '', p.replace(',', '.'))
                        if precio_limpio:
                            precios.append(float(precio_limpio))
                    except ValueError:
                        problemas.append(f"Fila {idx+1}: Precio inválido - '{p}'")
                
                # Emparejamiento
                for i in range(min(len(marcas), len(precios))):
                    resultados.append({
                        'ID': idx,
                        'Marca': marcas[i],
                        'Precio': precios[i],
                        'Sector': row.get('SELECCION BARRIO/SECTOR', ''),
                        'Tipo': row.get('TIPO DE COLMADO', '')
                    })
                    
            except Exception as e:
                problemas.append(f"Fila {idx+1}: Error - {str(e)}")
                continue
        
        # Reporte de problemas
        if problemas:
            with st.expander(f"⚠️ {len(problemas)} problemas de formato (click para ver)"):
                st.text("\n".join(problemas[:100]))  # Muestra primeros 100 errores
        
        return pd.DataFrame(resultados)

    # ------------------------------------------------------------
    # 3. INTERFAZ PRINCIPAL DEL TAB
    # ------------------------------------------------------------    
    with st.spinner("Procesando precios. Esto puede tomar unos segundos..."):
        try:
            df_precios = procesar_precios_definitivo(df, columna_marcas, columna_precios)
            
            if df_precios.empty:
                st.warning("No se encontraron datos de precios válidos")
                st.stop()
                
            # ------------------------------------------------------------
            # 4. VISUALIZACIÓN DE RESULTADOS
            # ------------------------------------------------------------
            st.subheader("📊 Análisis de Precios por Marca", divider="gray")
            
            # Resumen estadístico
            resumen = df_precios.groupby('Marca').agg({
                'Precio': ['mean', 'min', 'max', 'median', 'count'],
                'Sector': 'nunique'
            }).reset_index()
            
            resumen.columns = ['Marca', 'Promedio', 'Mínimo', 'Máximo', 'Mediana', 'Conteo', 'Sectores']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    resumen.sort_values('Promedio', ascending=False),
                    x='Marca',
                    y='Promedio',
                    error_y='Máximo',
                    error_y_minus='Mínimo',
                    title="Precio Promedio por Marca (RD$)",
                    color='Promedio',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True, key='chart_7')
            
            with col2:
                st.dataframe(
                    resumen.style.format({
                        'Promedio': 'RD${:.2f}',
                        'Mínimo': 'RD${:.2f}',
                        'Máximo': 'RD${:.2f}',
                        'Mediana': 'RD${:.2f}'
                    }).background_gradient(cmap='Greens'),
                    height=400
                )
            
            # Análisis adicional por sector
            st.subheader("📍 Distribución por Sector", divider="gray")
            sector_seleccionado = st.selectbox(
                "Filtrar por sector:",
                options=['Todos'] + sorted(df_precios['Sector'].unique().tolist()
            )
            )
                        
            if sector_seleccionado != 'Todos':
                df_filtrado = df_precios[df_precios['Sector'] == sector_seleccionado]
            else:
                df_filtrado = df_precios
                
            fig = px.box(
                df_filtrado,
                x='Marca',
                y='Precio',
                title=f"Distribución de Precios {'en ' + sector_seleccionado if sector_seleccionado != 'Todos' else ''}",
                color='Marca'
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_8')
            
        except Exception as e:
            st.error(f"Error durante el procesamiento: {str(e)}")
            st.stop()
    
    # # Análisis de rentabilidad
    # st.subheader("💵 Rentabilidad Percibida", divider="gray")
    # if 'MARCA_RENTABLE' in df.columns:
    #     rentabilidad = df['MARCA_RENTABLE'].value_counts().reset_index()
    #     rentabilidad.columns = ['Marca', 'Establecimientos']
        
    #     if not rentabilidad.empty:
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             fig = px.pie(
    #                 rentabilidad,
    #                 values='Establecimientos',
    #                 names='Marca',
    #                 title="Marcas con Mayor Beneficio Percibido",
    #                 hole=0.3
    #             )
    #             st.plotly_chart(fig, use_container_width=True, key='chart_9')
            
    #         with col2:
    #             fig = px.bar(
    #                 rentabilidad,
    #                 x='Marca',
    #                 y='Establecimientos',
    #                 title="Marcas más Rentables",
    #                 color='Marca',
    #                 text='Establecimientos'
    #             )
    #             st.plotly_chart(fig, use_container_width=True, key='chart_10')
            
    #         # Relación entre precios y rentabilidad
    #         if not df_precios.empty and 'MARCA_RENTABLE' in df.columns:
    #             st.subheader("🔗 Relación Precio-Rentabilidad", divider="gray")
                
    #             # Crear DataFrame combinado
    #             rentabilidad_precios = pd.merge(
    #                 rentabilidad,
    #                 df_precios,
    #                 left_on='Marca',
    #                 right_on='Marca',
    #                 how='left'
    #             ).dropna()
                
    #             if not rentabilidad_precios.empty:
    #                 fig = px.scatter(
    #                     rentabilidad_precios,
    #                     x='Promedio',
    #                     y='Establecimientos',
    #                     size='Conteo',
    #                     color='Marca',
    #                     title="Relación entre Precio Promedio y Rentabilidad Percibida",
    #                     labels={
    #                         'Promedio': 'Precio Promedio (RD$)',
    #                         'Establecimientos': 'Veces mencionada como más rentable'
    #                     },
    #                     hover_data=['Mínimo', 'Máximo']
    #                 )
    #                 st.plotly_chart(fig, use_container_width=True, key='chart_11')
    # else:
    #     st.warning("No hay datos de rentabilidad disponibles (columna 'MARCA_RENTABLE' no encontrada)")

# ----------------------------------------
# TAB 5: FRECUENCIA DE COMPRA
# ----------------------------------------
with tab5:
    st.header("🔁 Frecuencia de Compra", divider="rainbow")
    
    if 'CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.' in df.columns:
        # Distribución general
        st.subheader("📊 Distribución General", divider="gray")
        frecuencia = df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.'].value_counts().reset_index()
        frecuencia.columns = ['Frecuencia', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                frecuencia,
                values='Cantidad',
                names='Frecuencia',
                title="Frecuencia de Compra",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_12')
        
        with col2:
            fig = px.bar(
                frecuencia,
                x='Frecuencia',
                y='Cantidad',
                title="Frecuencia de Compra",
                color='Frecuencia'
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_13')
        
        # Por tipo de establecimiento
        st.subheader("🛒 Por Tipo de Establecimiento", divider="gray")
        if 'TIPO DE COLMADO' in df.columns:
            tabla_cruzada = pd.crosstab(
                df['TIPO DE COLMADO'],
                df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.']
            )
            fig = px.bar(
                tabla_cruzada,
                barmode='group',
                title="Frecuencia por Tipo de Establecimiento",
                labels={'value': 'Cantidad', 'variable': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_14')
        
        # Por marca rentable
        st.subheader("🏷️ Por Marca Rentable", divider="gray")
        if 'MARCA_RENTABLE' in df.columns:
            tabla_marca = pd.crosstab(
                df['MARCA_RENTABLE'],
                df['CON QUE FRECUENCIA COMPRA PAPEL HIGIENICO.']
            )
            
            fig = px.bar(
                tabla_marca,
                barmode='group',
                title="Frecuencia por Marca Rentable",
                labels={'value': 'Cantidad', 'variable': 'Frecuencia'}
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_15')
    else:
        st.warning("No hay datos de frecuencia de compra disponibles")

# ----------------------------------------
# TAB 6: INFLUENCIA DEL VENDEDOR
# ----------------------------------------
with tab6:
    st.header("💬 Influencia del Vendedor", divider="rainbow")
    
    if 'CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.' in df.columns:
        # Distribución general
        st.subheader("📊 Percepción de Influencia", divider="gray")
        influencia = df['CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.'].value_counts().reset_index()
        influencia.columns = ['Nivel', 'Cantidad']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                influencia,
                values='Cantidad',
                names='Nivel',
                title="Nivel de Influencia Percibida",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_16')
        
        with col2:
            fig = px.bar(
                influencia,
                y='Nivel',
                x='Cantidad',
                title="Influencia en la Decisión",
                color='Nivel',
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_17')
        
        # Por sector
        st.subheader("📍 Por Sector", divider="gray")
        if 'SELECCION BARRIO/SECTOR' in df.columns:
            # Convertir a numérico para calcular promedio
            niveles = {
                'Muy Poco': 1,
                'Poco': 2,
                'Regular': 3,
                'Mucho': 4,
                'Demasiado': 5
            }
            
            df['Influencia_Num'] = df['CUANTO CONSIDERA QUE USTED INFLUYE EN QUE EL CLIENTE PARA QUE SE LLEVE EL PRODUCTO QUE MAS LE INTERESE QUE COMPRE.'].map(niveles)
            
            influencia_sector = df.groupby('SELECCION BARRIO/SECTOR')['Influencia_Num'].mean().sort_values(ascending=False).reset_index()
            influencia_sector.columns = ['Sector', 'Influencia Promedio']
            
            fig = px.bar(
                influencia_sector,
                x='Sector',
                y='Influencia Promedio',
                title="Influencia Promedio por Sector",
                color='Influencia Promedio',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_18')
        
        # Relación con rentabilidad
        st.subheader("📈 Relación con Rentabilidad", divider="gray")
        if 'MARCA_RENTABLE' in df.columns and 'Influencia_Num' in df.columns:
            df_cruzado = df.dropna(subset=['MARCA_RENTABLE', 'Influencia_Num'])
            
            fig = px.box(
                df_cruzado,
                x='MARCA_RENTABLE',
                y='Influencia_Num',
                title="Influencia vs Marca Rentable",
                labels={
                    'MARCA_RENTABLE': 'Marca Rentable',
                    'Influencia_Num': 'Nivel de Influencia'
                }
            )
            st.plotly_chart(fig, use_container_width=True, key='chart_19')
    else:
        st.warning("No hay datos de influencia del vendedor disponibles")

# ----------------------------------------
# TAB 7: PROVEEDORES Y CANALES
# ----------------------------------------
with tab7:
    st.header("📦 Proveedores y Canales", divider="rainbow")
    
    # ------------------------------------------------------------
    # 1. DEFINIR EL NOMBRE CORRECTO DE LA COLUMNA
    # ------------------------------------------------------------
    # Nombre exacto de la columna en tus datos
    columna_proveedores = 'DONDE COMPRA EL PAPEL HIGIENICO\xa0Y SI PUEDE MENCIONAR A QUIEN SE LO COMPRA. ( HACER ENFASIS EN NIVEO SI HAY EN EL ESTABLECIMIENTO)'
    
    # ------------------------------------------------------------
    # 2. VERIFICAR Y PROCESAR LOS DATOS
    # ------------------------------------------------------------
    if columna_proveedores in df.columns:
        # Limpiar los datos de proveedores
        df_proveedores = df[df[columna_proveedores].notna()].copy()
        df_proveedores['Proveedor'] = (
            df_proveedores[columna_proveedores]
            .str.upper()  # Convertir a mayúsculas
            .str.strip()  # Eliminar espacios en blanco
            .str.replace(r'\s+', ' ', regex=True)  # Unificar espacios múltiples
        )
        
        # ------------------------------------------------------------
        # 3. VISUALIZACIÓN DE PROVEEDORES
        # ------------------------------------------------------------
        st.subheader("🏭 Proveedores Mencionados", divider="gray")
        
        # Nube de palabras
        st.write("### Nube de Palabras de Proveedores")
        fig_nube = generar_nube_palabras(
            df_proveedores['Proveedor'],
            "Proveedores más mencionados"
        )
        if fig_nube:
            st.pyplot(fig_nube)
        else:
            st.warning("No se pudo generar la nube de palabras")
        
        # Top proveedores
        st.write("### Top Proveedores")
        top_proveedores = df_proveedores['Proveedor'].value_counts().reset_index()
        top_proveedores.columns = ['Proveedor', 'Cantidad']
        
        if not top_proveedores.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    top_proveedores.head(10),
                    x='Proveedor',
                    y='Cantidad',
                    title="Top 10 Proveedores",
                    color='Cantidad',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True, key="top_proveedores_chart")
            
            with col2:
                st.dataframe(
                    top_proveedores.style.format({'Cantidad': '{:,.0f}'})
                          .background_gradient(cmap='Blues'),
                    use_container_width=True,
                    height=400
                )
        else:
            st.warning("No se encontraron datos de proveedores después del filtrado")
    else:
        st.error(f"""
        **Error:** No se encontró la columna de proveedores.
        
        - Columna buscada: '{columna_proveedores}'
        - Columnas disponibles: {list(df.columns)}
        """)

# ----------------------------------------
# TAB 8: RESPUESTAS ABIERTAS
# ----------------------------------------
with tab8:
    st.header("📝 Respuestas Abiertas", divider="rainbow")
    
    # ------------------------------------------------------------
    # 1. DEFINICIÓN DE COLUMNAS REALES EN TUS DATOS
    # ------------------------------------------------------------
    # Mapeo de preguntas a columnas reales
    preguntas_columnas = {
        "¿Por qué considera que esa es la más rentable?": [
            'POR QUE?',
            'RAZON RENTABILIDAD',
            'MOTIVO RENTABILIDAD'
        ],
        "¿A quién le compra?": [
            'DONDE COMPRA EL PAPEL HIGIENICO\xa0Y SI PUEDE MENCIONAR A QUIEN SE LO COMPRA. ( HACER ENFASIS EN NIVEO SI HAY EN EL ESTABLECIMIENTO)',
            'PROVEEDOR',
            'A QUIEN LE COMPRA'
        ],
    }
    
    # ------------------------------------------------------------
    # 2. INTERFAZ DE USUARIO
    # ------------------------------------------------------------
    # Selector de pregunta
    pregunta_seleccionada = st.selectbox(
        "Seleccionar pregunta a analizar",
        list(preguntas_columnas.keys()),
        index=0
    )
    
    # Buscar la columna correspondiente
    columna_respuestas = None
    for posible_columna in preguntas_columnas[pregunta_seleccionada]:
        if posible_columna in df.columns:
            columna_respuestas = posible_columna
            break
    
    # ------------------------------------------------------------
    # 3. PROCESAMIENTO DE RESPUESTAS
    # ------------------------------------------------------------
    if columna_respuestas:
        st.subheader(f"📋 Respuestas para: {pregunta_seleccionada}", divider="gray")
        
        # Búsqueda de términos
        termino_busqueda = st.text_input("Buscar término específico en las respuestas")
        
        # Filtrar respuestas
        if termino_busqueda:
            mask = df[columna_respuestas].str.contains(termino_busqueda, case=False, na=False)
            df_filtrado = df[mask]
        else:
            df_filtrado = df
        
        # Mostrar resultados
        if not df_filtrado.empty:
            # Seleccionar columnas relevantes para mostrar
            columnas_mostrar = [columna_respuestas, 'SELECCION BARRIO/SECTOR', 'TIPO DE COLMADO']
            columnas_disponibles = [col for col in columnas_mostrar if col in df_filtrado.columns]
            
            st.dataframe(
                df_filtrado[columnas_disponibles].dropna(subset=[columna_respuestas]),
                use_container_width=True,
                height=400
            )
            
            # Análisis de sentimiento básico (solo para pregunta de rentabilidad)
            if "rentable" in pregunta_seleccionada.lower():
                st.subheader("🧠 Análisis de Sentimiento", divider="gray")
                
                # Palabras clave para análisis
                palabras_positivas = ['bueno', 'excelente', 'mejor', 'beneficio', 'calidad', 'buena', 'bonito', 'rápido', 'confiable']
                palabras_negativas = ['malo', 'mal', 'peor', 'problema', 'caro', 'feo', 'difícil', 'lento', 'deficiente']
                
                conteo_positivo = 0
                conteo_negativo = 0
                ejemplos_positivos = []
                ejemplos_negativos = []
                
                for texto in df_filtrado[columna_respuestas].dropna():
                    texto = str(texto).lower()
                    if any(palabra in texto for palabra in palabras_positivas):
                        conteo_positivo += 1
                        if len(ejemplos_positivos) < 3:
                            ejemplos_positivos.append(texto[:200] + "..." if len(texto) > 200 else texto)
                    if any(palabra in texto for palabra in palabras_negativas):
                        conteo_negativo += 1
                        if len(ejemplos_negativos) < 3:
                            ejemplos_negativos.append(texto[:200] + "..." if len(texto) > 200 else texto)
                
                # Mostrar resultados
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Respuestas positivas", conteo_positivo)
                    if ejemplos_positivos:
                        st.write("**Ejemplos positivos:**")
                        for ejemplo in ejemplos_positivos:
                            st.success(f"📄 {ejemplo}")
                
                with col2:
                    st.metric("Respuestas negativas", conteo_negativo)
                    if ejemplos_negativos:
                        st.write("**Ejemplos negativos:**")
                        for ejemplo in ejemplos_negativos:
                            st.error(f"📄 {ejemplo}")
        else:
            st.warning("No se encontraron respuestas con los filtros aplicados")
    else:
        st.error(f"""
        **No se encontró la columna para:** {pregunta_seleccionada}
        
        **Columnas buscadas:** {preguntas_columnas[pregunta_seleccionada]}
        **Columnas disponibles:** {list(df.columns)}
        """)
        
        # Mostrar columnas similares
        st.write("### Columnas similares disponibles:")
        for col in df.columns:
            if any(keyword.lower() in col.lower() for keyword in ['por que', 'razon', 'motivo', 'proveedor', 'compra', 'observacion', 'comentario']):
                st.write(f"- {col}")

# # ----------------------------------------
# # TAB 9: EXPORTACIÓN DE REPORTES (CORREGIDO - AHORA GENERA PDF)
# # ----------------------------------------
# with tab9:
#     st.header("📊 Análisis General", divider="rainbow")
    
#     # Contenido del análisis general
#     st.subheader("Resumen Completo del Levantamiento")
    
#     # Métricas principales
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Establecimientos", len(df))
    
#     if 'SELECCION BARRIO/SECTOR' in df.columns:
#         sectores = df['SELECCION BARRIO/SECTOR'].nunique()
#         col2.metric("Sectores Cubiertos", sectores)
    
#     if not marcas_explotadas.empty:
#         marcas_count = marcas_explotadas['MARCAS_LISTA'].nunique()
#         col3.metric("Marcas Identificadas", marcas_count)
    
#     # Función CORREGIDA para generar PDF
#     def generar_reporte_completo_pdf(df, marcas_explotadas):
#         """Genera un reporte completo en PDF con todos los análisis"""
#         pdf = PDFReport()
#         pdf.add_page()
        
#         # Portada
#         pdf.set_font('Arial', 'B', 20)
#         pdf.cell(0, 40, 'REPORTE COMPLETO DE LEVANTAMIENTO', 0, 1, 'C')
#         pdf.set_font('Arial', 'B', 16)
#         pdf.cell(0, 20, 'Industrias Nigua - Mercado Papel Higiénico', 0, 1, 'C')
#         pdf.set_font('Arial', '', 12)
#         pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
#         pdf.ln(30)
        
#         # 1. RESUMEN EJECUTIVO
#         pdf.chapter_title('1. RESUMEN EJECUTIVO')
#         total_establecimientos = len(df)
        
#         contenido_resumen = f"""
#         Este reporte presenta los hallazgos del levantamiento de mercado realizado en {total_establecimientos} 
#         establecimientos comerciales. El estudio abarca múltiples sectores y proporciona insights clave sobre:
        
#         - Presencia y distribución de marcas en el mercado
#         - Estrategias de precios y competitividad
#         - Comportamiento de compra y frecuencia
#         - Oportunidades de crecimiento para Industrias Nigua
        
#         Principales hallazgos:
#         • Penetración de mercado de las principales marcas
#         • Análisis comparativo de precios
#         • Distribución geográfica de la presencia
#         • Recomendaciones estratégicas específicas
#         """
#         pdf.chapter_body(contenido_resumen)
        
#         # 2. METODOLOGÍA
#         pdf.add_page()
#         pdf.chapter_title('2. METODOLOGÍA')
#         pdf.chapter_body(f"""
#         Método de recolección: Entrevistas presenciales con cuestionario estructurado
#         Muestra: {total_establecimientos} establecimientos comerciales
#         Alcance: Múltiples sectores geográficos
#         Periodo: {df['Timestamp'].min().strftime('%d/%m/%Y') if 'Timestamp' in df.columns else 'N/A'} al {df['Timestamp'].max().strftime('%d/%m/%Y') if 'Timestamp' in df.columns else 'N/A'}
#         Instrumento: Formulario digital con validación en tiempo real
#         """)
        
#         # 3. ANÁLISIS DE MARCAS
#         pdf.chapter_title('3. ANÁLISIS DE PRESENCIA DE MARCAS')
        
#         if not marcas_explotadas.empty:
#             top_marcas = marcas_explotadas['MARCAS_LISTA'].value_counts().head(10)
#             analisis_marcas = "Top 10 marcas por presencia:\n\n"
#             for marca, count in top_marcas.items():
#                 porcentaje = (count / total_establecimientos) * 100
#                 analisis_marcas += f"• {marca}: {count} establecimientos ({porcentaje:.1f}%)\n"
            
#             # Presencia de Niveo específicamente
#             niveo_presente = 'NIVEO' in marcas_explotadas['MARCAS_LISTA'].values
#             if niveo_presente:
#                 niveo_count = (marcas_explotadas['MARCAS_LISTA'] == 'NIVEO').sum()
#                 niveo_porcentaje = (niveo_count / total_establecimientos) * 100
#                 analisis_marcas += f"\nPresencia específica de NIVEO: {niveo_count} establecimientos ({niveo_porcentaje:.1f}%)"
            
#             pdf.chapter_body(analisis_marcas)
        
#         # 4. ANÁLISIS DE PRECIOS
#         pdf.add_page()
#         pdf.chapter_title('4. ANÁLISIS DE PRECIOS')
        
#         precio_cols = [col for col in df.columns if col.startswith('PRECIO_')]
#         if precio_cols:
#             precios_info = "Precios promedio por marca:\n\n"
#             for col in precio_cols:
#                 marca = col.replace('PRECIO_', '')
#                 precios = df[col].dropna()
#                 if not precios.empty:
#                     precios_info += f"• {marca}: ${precios.mean():.2f} (rango: ${precios.min():.2f} - ${precios.max():.2f})\n"
            
#             pdf.chapter_body(precios_info)
        
#         # 5. RECOMENDACIONES ESTRATÉGICAS
#         pdf.chapter_title('5. RECOMENDACIONES ESTRATÉGICAS')
#         recomendaciones = """
#         1. EXPANSIÓN DE COBERTURA
#         • Identificar sectores con baja penetración de Niveo
#         • Establecer alianzas con distribuidores locales
#         • Desarrollar programas de incentivos para puntos de venta
        
#         2. ESTRATEGIA DE PRECIOS
#         • Analizar estructura competitiva de precios
#         • Desarrollar promociones estratégicas
#         • Crear bundles de productos para mayor valor percibido
        
#         3. FORTALECIMIENTO DE MARCA
#         • Campañas de merchandising en punto de venta
#         • Programas de fidelización para distribuidores
#         • Monitoreo continuo del mercado
#         """
#         pdf.chapter_body(recomendaciones)
        
#         # Guardar PDF
#         pdf_buffer = BytesIO()
#         pdf_output = pdf.output(dest='S').encode('latin-1')
#         pdf_buffer.write(pdf_output)
#         pdf_buffer.seek(0)
        
#         return pdf_buffer

#     # Botón para generar reporte completo en PDF
#     if st.button("📄 Generar Reporte Completo en PDF", type="primary"):
#         with st.spinner("Generando reporte completo en PDF..."):
#             try:
#                 pdf_buffer = generar_reporte_completo_pdf(df, marcas_explotadas)
                
#                 # Crear botón de descarga
#                 b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
#                 href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_completo_niveo_{datetime.now().strftime("%Y%m%d")}.pdf">⬇️ Descargar Reporte Completo (PDF)</a>'
                
#                 st.success("✅ Reporte generado exitosamente en formato PDF!")
#                 st.markdown(href, unsafe_allow_html=True)
                
#             except Exception as e:
#                 st.error(f"Error al generar el reporte PDF: {str(e)}")

# ----------------------------------------
# TAB 9: RESUMEN EJECUTIVO
# ----------------------------------------
with tab9:
    st.header("📋 Resumen Ejecutivo Completo", divider="rainbow")
    
    st.markdown("""
    ### Generar Resumen Ejecutivo con Respuestas al Cuestionario
    
    Este reporte responde específicamente a las preguntas del levantamiento de mercado
    y proporciona recomendaciones estratégicas basadas en los hallazgos.
    """)
    
    if st.button("📊 Generar Resumen Ejecutivo Completo", type="primary"):
        with st.spinner("Generando resumen ejecutivo completo..."):
            try:
                pdf_buffer = generar_resumen_ejecutivo_completo(df, marcas_explotadas)
                
                # Crear botón de descarga
                b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="resumen_ejecutivo_niveo_{datetime.now().strftime("%Y%m%d")}.pdf">⬇️ Descargar Resumen Ejecutivo Completo</a>'
                
                st.success("✅ Resumen ejecutivo generado exitosamente!")
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error al generar el resumen ejecutivo: {str(e)}")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>📅 Última actualización: {}</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
