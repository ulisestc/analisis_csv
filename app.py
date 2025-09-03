import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# --- Funciones Genéricas para Generar Gráficas ---

def fig_to_base64(fig):
    """Convierte una figura de Matplotlib a una cadena base64 para HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def generar_heatmap_nulos(df):
    """Genera un mapa de calor de valores nulos."""
    try:
        if df.isnull().sum().sum() == 0:
            return None # No hay nulos
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title('Mapa de Calor de Valores Nulos', fontsize=16)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Error generando heatmap de nulos: {e}")
        return None

def generar_heatmap_correlacion(df):
    """Genera un mapa de calor de correlación para variables numéricas."""
    try:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return None # No hay suficientes columnas numéricas para correlación
        
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title('Mapa de Calor de Correlación Numérica', fontsize=16)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Error generando heatmap de correlación: {e}")
        return None

def generar_histogramas_numericos(df):
    """Genera histogramas para las primeras 9 variables numéricas."""
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            return None

        # Limitar a un máximo de 9 para que sea legible
        numeric_cols_to_plot = numeric_cols[:9]
        n_plots = len(numeric_cols_to_plot)
        n_cols = 3
        n_rows = (n_plots - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() # Aplanar para facilitar la iteración

        for i, col in enumerate(numeric_cols_to_plot):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribución de {col}', fontsize=12)
        
        # Ocultar ejes no utilizados
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
            
        fig.suptitle('Distribución de Variables Numéricas', fontsize=20, y=1.03)
        fig.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Error generando histogramas: {e}")
        return None

def generar_barras_categoricas(df):
    """Genera gráficos de barras para las primeras 6 variables categóricas."""
    try:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        # Filtrar columnas con demasiadas categorías únicas para que sea legible
        suitable_cols = [col for col in categorical_cols if df[col].nunique() < 20]

        if len(suitable_cols) == 0:
            return None

        # Limitar a un máximo de 6
        cols_to_plot = suitable_cols[:6]
        n_plots = len(cols_to_plot)
        n_cols = 2
        n_rows = (n_plots - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols_to_plot):
            order = df[col].value_counts().index[:10] # Top 10 categorías
            sns.countplot(y=df[col], ax=axes[i], order=order, palette='viridis')
            axes[i].set_title(f'Frecuencia en {col}', fontsize=12)
            axes[i].set_xlabel('Conteo')
            axes[i].set_ylabel('')

        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
            
        fig.suptitle('Frecuencia de Variables Categóricas', fontsize=20, y=1.03)
        fig.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Error generando gráficos de barras: {e}")
        return None

# --- Rutas de la Aplicación Flask ---

@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Recibe el archivo CSV, lo analiza y devuelve los resultados en formato JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'})

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

            # 1. Estadísticas básicas
            num_filas, num_columnas = df.shape
            
            # 2. Información de tipos de datos y nulos (texto)
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            # 3. Conteo de duplicados
            num_duplicados = int(df.duplicated().sum())

            # 4. Estadísticas descriptivas (HTML)
            desc_numericas_html = df.describe(include='number').round(2).to_html(classes='table table-striped table-bordered', border=0)
            desc_categoricas_html = df.describe(include=['object', 'category']).to_html(classes='table table-striped table-bordered', border=0)

            # 5. Generación de gráficas genéricas
            graficas = {
                'null_heatmap': generar_heatmap_nulos(df),
                'correlation_heatmap': generar_heatmap_correlacion(df),
                'numeric_distributions': generar_histogramas_numericos(df),
                'categorical_distributions': generar_barras_categoricas(df)
            }

            return jsonify({
                'nombre_archivo': file.filename,
                'num_filas': num_filas,
                'num_columnas': num_columnas,
                'info_columnas': info_str,
                'num_duplicados': num_duplicados,
                'desc_numericas': desc_numericas_html if not df.select_dtypes(include='number').empty else "<p>No hay columnas numéricas.</p>",
                'desc_categoricas': desc_categoricas_html if not df.select_dtypes(include=['object', 'category']).empty else "<p>No hay columnas categóricas.</p>",
                'graficas': graficas
            })

        except Exception as e:
            return jsonify({'error': f'Hubo un error al procesar el archivo: {str(e)}'})

    return jsonify({'error': 'Formato de archivo no válido. Por favor, sube un archivo .csv'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

