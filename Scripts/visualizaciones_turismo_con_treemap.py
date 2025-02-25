#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualización de Datos Turísticos mediante Treemaps.

Este script genera visualizaciones interactivas tipo treemap para analizar la 
distribución de turistas en España según diferentes dimensiones. Crea dos visualizaciones
complementarias:
1. Un treemap que muestra la distribución de turistas por país de origen y región geográfica,
   junto con indicadores de satisfacción y condiciones climáticas.
2. Un treemap que presenta la distribución por temporada turística y medio de acceso,
   permitiendo identificar patrones estacionales.

Las visualizaciones generadas son interactivas (HTML) y permiten explorar diferentes
niveles de jerarquía, facilitando tanto el análisis general como detallado de los
patrones turísticos.

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos estructurados.
  Utilizada para cargar el dataset, realizar agrupaciones, cálculos estadísticos
  y preparar los datos para la visualización. 
  Funciones clave: read_csv, groupby, agg, apply, reset_index.

- plotly.express (5.13.0+): Módulo de alto nivel de Plotly para creación de gráficos.
  Proporciona una API simplificada para crear visualizaciones complejas e interactivas.
  Específicamente utilizado para generar treemaps con múltiples niveles jerárquicos
  y escalas de color basadas en variables cuantitativas.
  Funciones clave: treemap.

- plotly.io (5.13.0+): Módulo de entrada/salida de Plotly.
  Permite guardar gráficos interactivos en diversos formatos, incluyendo HTML.
  Funciones clave: write_html.

Autor: Bernardino Chancusig
Fecha: 25/02/2025
Versión: 1.0
"""

import pandas as pd                # Para manipulación y análisis de datos
import plotly.express as px        # Para creación de visualizaciones interactivas 
import plotly.io as pio            # Para exportación de gráficos

def create_tourism_treemaps(df_path='predicciones_finales_2019_2028.csv'):
    """
    Crea dos visualizaciones treemap interactivas a partir de datos turísticos.
    
    Esta función genera dos treemaps complementarios que permiten analizar la
    distribución de turistas desde diferentes perspectivas: por origen geográfico
    y por patrones temporales/modalidad de acceso.
    
    Parameters
    ----------
    df_path : str, optional
        Ruta al archivo CSV con los datos turísticos procesados, 
        por defecto 'predicciones_finales_2019_2028.csv'.
        
    Returns
    -------
    tuple
        Tupla con dos objetos plotly.graph_objects.Figure correspondientes a:
        - Treemap por país y región (fig1)
        - Treemap por temporada y medio de acceso (fig2)
        
    Notes
    -----
    La función realiza los siguientes pasos:
    1. Carga y procesa los datos del CSV
    2. Agrega los datos por país y calcula estadísticas relevantes
    3. Asigna regiones geográficas a los países
    4. Crea el primer treemap (por país y región)
    5. Agrega los datos por temporada y medio de acceso
    6. Crea el segundo treemap (por temporada y medio acceso)
    7. Configura formato y opciones de visualización
    8. Guarda ambos treemaps como archivos HTML interactivos
    
    Los archivos generados son:
    - tourism_region_treemap.html: Distribución por país y región
    - tourism_seasonal_treemap.html: Distribución por temporada y medio de acceso
    
    Raises
    ------
    FileNotFoundError
        Si el archivo CSV especificado no existe.
    pd.errors.EmptyDataError
        Si el archivo CSV está vacío o no contiene datos válidos.
    Exception
        Cualquier otro error durante el procesamiento o generación de gráficos.
    """
    # Leer los datos
    print(f"Cargando datos desde: {df_path}")
    df = pd.read_csv(df_path)
    print(f"Datos cargados correctamente. {len(df)} registros encontrados.")
    
    # 1. TREEMAP POR PAÍS Y REGIÓN
    print("Generando treemap por país y región...")
    
    # Agrupar datos por país y calcular métricas
    country_summary = df.groupby('Pais').agg({
        'Num_Turistas': 'sum',          # Total de turistas
        'Satisfaccion': 'mean',         # Satisfacción promedio
        'Temperatura_Media': 'mean',    # Temperatura media
        'Precipitacion_Media': 'mean'   # Precipitación media
    }).reset_index()
    
    # Calcular el porcentaje de turistas por país
    total_tourists = country_summary['Num_Turistas'].sum()
    country_summary['Porcentaje_Turistas'] = (country_summary['Num_Turistas'] / total_tourists) * 100
    
    # Definir regiones geográficas para agrupar países
    regions = {
        'Europa Occidental': ['Francia', 'Alemania', 'Reino Unido', 'Países Bajos', 'Bélgica', 'Suiza', 'Austria', 'Luxemburgo'],
        'Europa del Norte': ['Suecia', 'Noruega', 'Finlandia', 'Dinamarca'],
        'Europa del Sur': ['Italia', 'Portugal'],
        'Europa del Este': ['Polonia', 'República Checa', 'Rusia'],
        'América': ['EE.UU.', 'Canadá', 'México', 'Brasil', 'Argentina'],
        'Asia': ['China', 'Japón', 'Corea del Sur']
    }
    
    # Asignar región a cada país mediante función lambda
    country_summary['Region'] = country_summary['Pais'].apply(
        lambda x: next((k for k, v in regions.items() if x in v), 'Otros')
    )
    
    # Crear primer Treemap (País y Región)
    fig1 = px.treemap(
        country_summary,
        path=['Region', 'Pais'],         # Jerarquía: Región > País
        values='Num_Turistas',           # Tamaño basado en número de turistas
        color='Satisfaccion',            # Color basado en satisfacción
        hover_data=['Porcentaje_Turistas', 'Temperatura_Media', 'Precipitacion_Media'],  # Datos adicionales al hover
        color_continuous_scale='RdYlBu', # Escala de color: rojo (bajo) a azul (alto)
        title='Distribución de Turistas por País y Región'
    )
    
    # 2. TREEMAP POR TEMPORADA Y MEDIO DE ACCESO
    print("Generando treemap por temporada y medio de acceso...")
    
    # Agrupar datos por temporada y medio de acceso
    seasonal_summary = df.groupby(['Temporada', 'Medio_Acceso']).agg({
        'Num_Turistas': 'sum',      # Total de turistas
        'Satisfaccion': 'mean'      # Satisfacción promedio
    }).reset_index()
    
    # Calcular porcentaje de turistas por combinación temporada/medio
    seasonal_total = seasonal_summary['Num_Turistas'].sum()
    seasonal_summary['Porcentaje_Turistas'] = (seasonal_summary['Num_Turistas'] / seasonal_total) * 100
    
    # Crear segundo Treemap (Temporada y Medio de Acceso)
    fig2 = px.treemap(
        seasonal_summary,
        path=['Temporada', 'Medio_Acceso'],  # Jerarquía: Temporada > Medio de Acceso
        values='Num_Turistas',               # Tamaño basado en número de turistas
        color='Satisfaccion',                # Color basado en satisfacción
        hover_data=['Porcentaje_Turistas'],  # Datos adicionales al hover
        color_continuous_scale='RdYlBu',     # Escala de color: rojo (bajo) a azul (alto)
        title='Distribución de Turistas por Temporada y Medio de Acceso'
    )
    
    # Configurar diseño para ambos gráficos
    for fig in [fig1, fig2]:
        fig.update_layout(
            title_x=0.5,               # Centrar título
            width=1000,                # Ancho en píxeles
            height=600,                # Alto en píxeles
            margin=dict(t=50, l=25, r=25, b=25)  # Márgenes
        )
    
    # Actualizar formato hover para primer Treemap
    fig1.update_traces(
        hovertemplate="""
        <b>%{label}</b><br>
        Turistas: %{value:,.0f}<br>
        Porcentaje: %{customdata[0]:.1f}%<br>
        Satisfacción: %{color:.1f}<br>
        Temperatura Media: %{customdata[1]:.1f}°C<br>
        Precipitación Media: %{customdata[2]:.1f}mm<br>
        <extra></extra>
        """
    )
    
    # Actualizar formato hover para segundo Treemap
    fig2.update_traces(
        hovertemplate="""
        <b>%{label}</b><br>
        Turistas: %{value:,.0f}<br>
        Porcentaje: %{customdata[0]:.1f}%<br>
        Satisfacción: %{color:.1f}<br>
        <extra></extra>
        """
    )
    
    # Guardar los gráficos como archivos HTML interactivos
    output_file1 = 'tourism_region_treemap.html'
    output_file2 = 'tourism_seasonal_treemap.html'
    
    pio.write_html(fig1, output_file1)
    pio.write_html(fig2, output_file2)
    
    print("\nGráficos guardados correctamente:")
    print(f"- {output_file1} (Distribución por País y Región)")
    print(f"- {output_file2} (Distribución por Temporada y Medio de Acceso)")
    
    return fig1, fig2

def main():
    """
    Función principal que ejecuta la generación de visualizaciones treemap.
    
    Esta función coordina el proceso de creación de treemaps y maneja posibles errores,
    proporcionando mensajes informativos durante el proceso.
    """
    print("Iniciando generación de visualizaciones treemap para datos turísticos...")
    
    try:
        # Ejecutar la función principal
        fig1, fig2 = create_tourism_treemaps()
        print("\nProceso completado exitosamente.")
        
        # Sugerencia de uso
        print("\nPara visualizar los treemaps, abra los archivos HTML generados en su navegador.")
        print("Estos gráficos son interactivos: puede hacer clic en secciones para profundizar en los datos")
        print("y pasar el cursor sobre las áreas para ver información detallada.")
        
    except FileNotFoundError as e:
        print(f"\nError: No se encontró el archivo de datos. {str(e)}")
        print("Verifique que el archivo existe en la ruta especificada.")
    
    except pd.errors.EmptyDataError:
        print("\nError: El archivo de datos está vacío o no contiene datos válidos.")
    
    except Exception as e:
        print(f"\nError al crear los treemaps: {str(e)}")
        print("Verifique los datos de entrada y las dependencias necesarias.")


if __name__ == "__main__":
    main()