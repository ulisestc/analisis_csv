import os
import io
import base64
import pandas as pd
from flask import Flask, request, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Configuración para que Matplotlib funcione sin entorno gráfico
import matplotlib.pyplot as plt
import seaborn as sns

# Inicialización de la aplicación Flask
app = Flask(__name__)

def generar_grafica_nulos(df):
    """Genera una gráfica de barras de valores nulos y la devuelve como una imagen en base64."""
    try:
        # Contar nulos y filtrar solo columnas que tengan nulos
        nulos = df.isnull().sum()
        nulos = nulos[nulos > 0]

        if nulos.empty:
            return None # No hay nulos, no se genera gráfica

        plt.figure(figsize=(10, 6))
        sns.barplot(x=nulos.index, y=nulos.values, palette='viridis')
        plt.ylabel('Cantidad de valores nulos')
        plt.xlabel('Columnas')
        plt.title('Valores Nulos por Columna')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Guardar la gráfica en un buffer de memoria
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Codificar la imagen en base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close() # Cerrar la figura para liberar memoria
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error generando gráfica de nulos: {e}")
        return None

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
            # Leer el archivo CSV en un DataFrame de Pandas
            df = pd.read_csv(file)

            # 1. Estadísticas básicas
            num_filas, num_columnas = df.shape
            
            # 2. Información de tipos de datos y nulos
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            # 3. Conteo de duplicados
            num_duplicados = int(df.duplicated().sum())

            # 4. Estadísticas descriptivas para columnas numéricas
            desc_numericas = df.describe(include='number').round(2).to_html(classes='table table-striped table-bordered', border=0)
            
            # 5. Estadísticas descriptivas para columnas categóricas
            desc_categoricas = df.describe(include=['object', 'category']).to_html(classes='table table-striped table-bordered', border=0)
            
            # 6. Generar gráfica de nulos
            grafica_nulos_b64 = generar_grafica_nulos(df)

            # Construir la respuesta
            return jsonify({
                'nombre_archivo': file.filename,
                'num_filas': num_filas,
                'num_columnas': num_columnas,
                'info_columnas': info_str,
                'num_duplicados': num_duplicados,
                'desc_numericas': desc_numericas if not df.select_dtypes(include='number').empty else "<p>No hay columnas numéricas.</p>",
                'desc_categoricas': desc_categoricas if not df.select_dtypes(include=['object', 'category']).empty else "<p>No hay columnas categóricas.</p>",
                'grafica_nulos': grafica_nulos_b64
            })

        except Exception as e:
            return jsonify({'error': f'Hubo un error al procesar el archivo: {str(e)}'})

    return jsonify({'error': 'Formato de archivo no válido. Por favor, sube un archivo .csv'})

if __name__ == '__main__':
    # Usar el puerto que Render nos asigne, o 5000 para desarrollo local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
