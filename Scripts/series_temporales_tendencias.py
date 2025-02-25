#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: series_temporales_tendencias.py
Visualización Mejorada de Series Temporales de Tendencias Turísticas.

Este módulo genera visualizaciones avanzadas de las tendencias turísticas en España
entre 2019 y 2023, proporcionando tanto gráficos unificados como visualizaciones
individuales para cada destino principal. La funcionalidad incluye elementos interactivos
para exploración de datos, contexto temporal para eventos significativos y análisis
visual de patrones estacionales.

El script está diseñado para proporcionar insights claros sobre:
- Evolución temporal del número de turistas por destino
- Impacto de la pandemia COVID-19 y posterior recuperación
- Estacionalidad en los diferentes destinos turísticos
- Tendencias a largo plazo mediante análisis de regresión polinómica

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca fundamental para manipulación y análisis de datos tabulares.
  Utilizada para cargar, transformar y agregar datos turísticos, manejar series temporales,
  y preparar datos para visualización.
  Componentes clave: read_csv, to_datetime, groupby, sort_values, DataFrame, Series.

- numpy (1.22.0+): Biblioteca para computación numérica.
  Utilizada principalmente para cálculos matemáticos como ajustes polinómicos (polyfit)
  para líneas de tendencia y operaciones con arrays.
  Componentes clave: arange, polyfit, poly1d.

- matplotlib (3.5.0+): Biblioteca principal para visualización estática.
  Proporciona la base para todas las visualizaciones generadas en este script.
  Componentes específicos:
  - pyplot: Interfaz principal para creación de gráficos.
  - dates: Manejo especializado de formatos de fechas en ejes.
  - ticker.FuncFormatter: Personalización de etiquetas en ejes numéricos.
  - gridspec: Creación de layouts complejos con múltiples subgráficos.

- seaborn (0.11.0+): Biblioteca para visualización estadística y estética mejorada.
  Proporciona paletas de colores accesibles y estilos visuales modernos.
  Componentes clave: color_palette, set_palette.

- ipywidgets (8.0.0+): Biblioteca para componentes interactivos en Jupyter.
  Permite la creación de controles interactivos para exploración de datos.
  Componentes utilizados: Checkbox, Button, RadioButtons, Output, VBox.

- IPython.display: Interfaz para la visualización interactiva en notebooks.
  Componentes clave: display, clear_output.

- datetime: Módulo estándar para manipulación de fechas y horas.
  Importado para soporte en procesamiento temporal.

- os: Módulo estándar para interacción con el sistema operativo.
  Utilizado para gestión de directorios y rutas de archivos.

Autor: Bernardino Chancusig
Fecha: 25/02/2025
Versión: 1.0
"""

# Importación de librerías
import pandas as pd               # Para manipulación y análisis de datos
import numpy as np                # Para operaciones numéricas y cálculos
import matplotlib.pyplot as plt   # Para creación de gráficos
import matplotlib.dates as mdates # Para manejo de fechas en ejes
from matplotlib.ticker import FuncFormatter  # Para formato personalizado de etiquetas
import seaborn as sns             # Para paletas de colores y estilos visuales
from datetime import datetime     # Para manipulación de fechas
import os                         # Para manejo de directorios y archivos
import ipywidgets as widgets      # Para componentes interactivos (en Jupyter)
from IPython.display import display, clear_output  # Para visualización en notebooks
import matplotlib.gridspec as gridspec  # Para layouts complejos de gráficos

# Configuración para guardar archivos
OUTPUT_DIR = 'resultados_visualizacion'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')  # Estilo base con cuadrícula
sns.set_palette("colorblind")            # Paleta accesible para daltónicos
plt.rcParams['font.family'] = 'DejaVu Sans'  # Fuente legible y multiplataforma
plt.rcParams['figure.figsize'] = (14, 8)     # Tamaño predeterminado de figuras
plt.rcParams['font.size'] = 12               # Tamaño de fuente base

def cargar_datos(filepath='turismo_procesado_2019_2023.csv'):
    """
    Carga y preprocesa los datos de turismo desde un archivo CSV.
    
    Parameters
    ----------
    filepath : str, optional
        Ruta al archivo CSV con los datos de turismo, por defecto 'turismo_procesado_2019_2023.csv'.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con los datos preprocesados y listos para visualización.
    
    Notes
    -----
    El preprocesamiento incluye:
    - Conversión de la columna 'Fecha' a formato datetime
    - Ordenación cronológica de los registros
    - Creación de una columna 'Año_Mes' para facilitar agrupaciones mensuales
    
    Raises
    ------
    FileNotFoundError
        Si el archivo especificado no existe.
    pd.errors.EmptyDataError
        Si el archivo está vacío o no contiene datos válidos.
    """
    print(f"Cargando datos desde {filepath}...")
    
    # Cargar el CSV
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: El archivo {filepath} no existe.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo {filepath} está vacío o no contiene datos válidos.")
        raise
    
    # Convertir la columna de fecha a formato datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%m/%d/%Y')
    
    # Ordenar por fecha
    df = df.sort_values('Fecha')
    
    # Crear columna de año-mes para agrupaciones mensuales
    df['Año_Mes'] = df['Fecha'].dt.strftime('%Y-%m')
    
    print(f"Datos cargados correctamente. Registros totales: {len(df)}")
    
    return df

def obtener_top_destinos(df, num_destinos=5, destinos_fijos=None):
    """
    Identifica los principales destinos turísticos por volumen total de visitantes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos de turismo.
    num_destinos : int, optional
        Número de destinos principales a identificar, por defecto 5.
    destinos_fijos : list, optional
        Lista de destinos que deben incluirse independientemente de su ranking.
    
    Returns
    -------
    list
        Lista de nombres de los principales destinos ordenados por volumen.
    
    Notes
    -----
    La función permite especificar destinos que deben incluirse obligatoriamente
    (como 'Comunidad de Madrid'), complementando la lista con los destinos de mayor
    volumen hasta alcanzar el número total especificado.
    """
    # Agrupar por destino y calcular el total de turistas
    totales_destino = df.groupby('Destino_Principal')['Num_Turistas'].sum()
    
    # Destinos que deben incluirse obligatoriamente
    if destinos_fijos is None:
        destinos_fijos = []
    
    # Asegurar que 'Comunidad de Madrid' esté incluida si no se especificó en destinos_fijos
    if 'Comunidad de Madrid' not in destinos_fijos:
        destinos_fijos.append('Comunidad de Madrid')
    
    # Obtener los top destinos excluyendo los fijos
    destinos_restantes = [d for d in totales_destino.index if d not in destinos_fijos]
    top_restantes = totales_destino[destinos_restantes].nlargest(num_destinos - len(destinos_fijos)).index.tolist()
    
    # Combinar destinos fijos con top restantes
    top_destinos = destinos_fijos + top_restantes
    
    print(f"Destinos identificados: {', '.join(top_destinos)}")
    
    return top_destinos

def preparar_datos_mensuales(df, top_destinos):
    """
    Prepara los datos agrupados mensualmente para cada destino principal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos de turismo.
    top_destinos : list
        Lista de nombres de los principales destinos.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    
    Notes
    -----
    La función realiza:
    1. Filtrado de los destinos principales
    2. Agrupación mensual de los datos
    3. Conversión del formato año-mes a fecha completa
    4. Ordenación cronológica y por destino
    """
    # Filtrar solo los destinos principales
    df_top = df[df['Destino_Principal'].isin(top_destinos)]
    
    # Agrupar por fecha y destino para sumar turistas
    df_mensual = df_top.groupby(['Año_Mes', 'Destino_Principal'])['Num_Turistas'].sum().reset_index()
    
    # Convertir Año_Mes a datetime para ordenar correctamente
    df_mensual['Fecha'] = pd.to_datetime(df_mensual['Año_Mes'] + '-01')
    
    # Ordenar por fecha y destino
    df_mensual = df_mensual.sort_values(['Fecha', 'Destino_Principal'])
    
    return df_mensual

def formato_miles(x, pos):
    """
    Formatea números grandes en miles (K) o millones (M) para mejorar la legibilidad.
    
    Parameters
    ----------
    x : float
        Valor numérico a formatear.
    pos : int
        Posición (requerido por matplotlib.ticker.FuncFormatter).
    
    Returns
    -------
    str
        Cadena formateada con sufijos K o M según corresponda.
    
    Notes
    -----
    Formatos aplicados:
    - Millones: 1.1M (un decimal)
    - Miles: 100K (sin decimales)
    - Números menores: valor entero
    """
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'  # Millones con un decimal
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'  # Miles sin decimales
    else:
        return f'{x:.0f}'        # Números menores como enteros

def calcular_tendencia(datos):
    """
    Calcula la línea de tendencia utilizando regresión polinómica.
    
    Parameters
    ----------
    datos : pandas.DataFrame
        DataFrame con los datos para un destino específico.
    
    Returns
    -------
    numpy.ndarray
        Valores Y de la línea de tendencia.
    
    Notes
    -----
    Utiliza un polinomio de grado 3 para capturar adecuadamente:
    - Tendencia inicial
    - Caída por COVID-19
    - Recuperación posterior
    - Tendencia actual
    
    Un polinomio de grado 3 es ideal para modelar estos cambios de dirección
    en la serie temporal.
    """
    x = np.arange(len(datos))
    y = datos['Num_Turistas'].values
    
    # Polinomio de grado 3 para capturar la caída por COVID y recuperación
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    
    return p(x)

def generar_grafico_unificado(df_mensual, top_destinos, destinos_seleccionados=None):
    """
    Genera un gráfico unificado con las tendencias de todos los destinos seleccionados.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    top_destinos : list
        Lista de nombres de los principales destinos.
    destinos_seleccionados : list, optional
        Lista de destinos a incluir en el gráfico. Si es None, se incluyen todos los top_destinos.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figura con el gráfico unificado de todos los destinos.
    
    Notes
    -----
    Este gráfico muestra:
    - Líneas temporales para cada destino seleccionado
    - Áreas sombreadas para períodos clave (COVID-19, recuperación)
    - Marcadores para los datos reales
    - Escala ajustada para valores en miles/millones
    - Anotaciones explicativas para contexto
    """
    if destinos_seleccionados is None:
        destinos_seleccionados = top_destinos
    
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Definir colores consistentes para los destinos principales
    colores_fijos = {
        'Cataluña': '#1f77b4',       # Azul
        'Baleares': '#ff7f0e',       # Naranja
        'Andalucía': '#2ca02c',      # Verde
        'Canarias': '#d62728',       # Rojo
        'Comunidad Valenciana': '#9467bd',  # Morado
        'Comunidad de Madrid': '#8c564b'    # Marrón
    }
    
    # Crear paleta de colores, priorizando los colores fijos para consistencia visual
    colores = {}
    paleta_restante = sns.color_palette("colorblind", n_colors=10)
    color_idx = 0
    
    for destino in top_destinos:
        if destino in colores_fijos:
            colores[destino] = colores_fijos[destino]
        else:
            colores[destino] = paleta_restante[color_idx]
            color_idx += 1
    
    # Graficar cada destino seleccionado
    for destino in destinos_seleccionados:
        datos_destino = df_mensual[df_mensual['Destino_Principal'] == destino]
        
        # Ordenar por fecha
        datos_destino = datos_destino.sort_values('Fecha')
        
        # Graficar línea principal
        ax.plot(datos_destino['Fecha'], datos_destino['Num_Turistas'], 
               label=destino, color=colores[destino], linewidth=2.5)
        
        # Añadir puntos para los datos reales
        ax.scatter(datos_destino['Fecha'], datos_destino['Num_Turistas'], 
                 color=colores[destino], s=30, zorder=3)
    
    # Añadir áreas sombreadas para eventos importantes
    # Período COVID-19
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'), 
              alpha=0.2, color='gray', label='Periodo COVID-19')
    
    # Añadir texto explicativo para COVID-19
    ax.text(pd.Timestamp('2020-06-15'), ax.get_ylim()[1]*0.9, 
           "Impacto\nCOVID-19", 
           ha='center', va='center', fontsize=12, color='dimgray',
           bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.7))
    
    # Período de recuperación post-pandemia
    ax.axvspan(pd.Timestamp('2021-06-01'), pd.Timestamp('2022-06-01'), 
              alpha=0.1, color='green')
    
    # Añadir texto explicativo para recuperación
    ax.text(pd.Timestamp('2021-09-15'), ax.get_ylim()[1]*0.8, 
           "Recuperación\npost-pandemia", 
           ha='center', va='center', fontsize=12, color='darkgreen',
           bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='green', alpha=0.7))
    
    # Configurar gráfico
    ax.set_title('Series Temporales de Tendencias Turísticas (2019-2023)', fontsize=18, pad=20)
    ax.set_xlabel('Fecha', fontsize=14)
    ax.set_ylabel('Número de Turistas', fontsize=14)
    
    # Formatear eje Y para mostrar números en miles/millones
    ax.yaxis.set_major_formatter(FuncFormatter(formato_miles))
    
    # Configurar eje X para mostrar fechas cada 6 meses
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))  # Formato: mes abreviado + año
    
    # Añadir cuadrícula
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir leyenda
    ax.legend(title="Destinos Principales", loc='upper left', framealpha=0.9, fontsize=12)
    
    # Añadir información de fuente en el pie de gráfico
    plt.figtext(0.01, 0.02, "Fuente: Datos de turismo 2019-2023", 
               ha="left", fontsize=10, fontstyle='italic')
    
    # Añadir marcas de agua / info adicional
    plt.figtext(0.99, 0.02, "Series Temporales de Tendencias Turísticas", 
               ha="right", fontsize=10, color='gray')
    
    # Ajustar layout
    plt.tight_layout()
    
    return fig

def generar_graficos_individuales(df_mensual, top_destinos, destinos_seleccionados=None):
    """
    Genera gráficos individuales para cada destino seleccionado.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    top_destinos : list
        Lista de nombres de los principales destinos.
    destinos_seleccionados : list, optional
        Lista de destinos para los que generar gráficos individuales.
        Si es None, se generan para todos los top_destinos.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figura con los gráficos individuales de cada destino.
    
    Notes
    -----
    Cada gráfico individual muestra:
    - Línea temporal de datos reales
    - Línea punteada de tendencia (regresión polinómica)
    - Áreas sombreadas para períodos clave (COVID-19, recuperación)
    - Título específico para cada destino
    
    Los gráficos se organizan en una cuadrícula con un máximo de 3 columnas.
    """
    if destinos_seleccionados is None:
        destinos_seleccionados = top_destinos
    
    # Determinar la disposición óptima de la cuadrícula
    n_graficos = len(destinos_seleccionados)
    n_filas = (n_graficos + 2) // 3  # Redondeo hacia arriba para filas (máximo 3 columnas)
    n_columnas = min(3, n_graficos)  # Máximo 3 columnas para mejor visualización
    
    # Crear figura principal
    fig = plt.figure(figsize=(16, n_filas * 5))
    gs = gridspec.GridSpec(n_filas, n_columnas, figure=fig)
    
    # Definir colores para cada destino
    paleta = sns.color_palette("colorblind", n_colors=len(top_destinos))
    colores = {destino: paleta[i] for i, destino in enumerate(top_destinos)}
    
    # Generar un gráfico para cada destino seleccionado
    for i, destino in enumerate(destinos_seleccionados):
        fila = i // n_columnas
        columna = i % n_columnas
        
        # Crear subplot para este destino
        ax = fig.add_subplot(gs[fila, columna])
        
        # Filtrar datos del destino y ordenarlos cronológicamente
        datos_destino = df_mensual[df_mensual['Destino_Principal'] == destino].sort_values('Fecha')
        
        # Graficar línea principal con datos reales
        ax.plot(datos_destino['Fecha'], datos_destino['Num_Turistas'], 
               color=colores[destino], linewidth=2.5)
        
        # Añadir puntos para los datos reales
        ax.scatter(datos_destino['Fecha'], datos_destino['Num_Turistas'], 
                 color=colores[destino], s=30, zorder=3)
        
        # Calcular y graficar línea de tendencia (regresión polinómica)
        tendencia = calcular_tendencia(datos_destino)
        ax.plot(datos_destino['Fecha'], tendencia, '--', 
               color=colores[destino], linewidth=1.5, alpha=0.7)
        
        # Añadir áreas sombreadas para eventos importantes
        # Período COVID-19 (marzo-diciembre 2020)
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'), 
                  alpha=0.2, color='gray')
        
        # Período de recuperación (junio 2021-junio 2022)
        ax.axvspan(pd.Timestamp('2021-06-01'), pd.Timestamp('2022-06-01'), 
                  alpha=0.1, color='green')
        
        # Configurar título y etiquetas
        ax.set_title(f'Tendencia Turística: {destino}', fontsize=14)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Número de Turistas', fontsize=12)
        
        # Formatear eje Y para mostrar números en miles/millones
        ax.yaxis.set_major_formatter(FuncFormatter(formato_miles))
        
        # Configurar eje X para mostrar fechas
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Cada 6 meses
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))  # Mes y año
        
        # Añadir cuadrícula
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir información de fuente en el pie de gráfico
    plt.figtext(0.01, 0.02, "Fuente: Datos de turismo 2019-2023", 
               ha="left", fontsize=10, fontstyle='italic')
    
    # Ajustar layout para evitar superposiciones
    plt.tight_layout()
    
    return fig

def interfaz_interactiva(df_mensual, top_destinos):
    """
    Crea una interfaz interactiva para explorar las tendencias turísticas.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    top_destinos : list
        Lista de nombres de los principales destinos.
    
    Notes
    -----
    Esta función utiliza ipywidgets y solo funciona en entornos Jupyter Notebook/Lab.
    Proporciona:
    - Checkboxes para seleccionar destinos específicos
    - Botones de radio para alternar entre vista unificada y vistas individuales
    - Botón para actualizar la visualización
    - Área de salida para mostrar el gráfico generado
    
    Al ejecutar esta función fuera de un entorno Jupyter, aparecerá un mensaje
    indicando que no es posible crear la interfaz interactiva.
    """
    # Crear checkboxes para seleccionar destinos
    checkboxes = {}
    for destino in top_destinos:
        checkboxes[destino] = widgets.Checkbox(
            value=True,
            description=destino,
            disabled=False
        )
    
    # Crear botón de radio para seleccionar tipo de vista
    vista_selector = widgets.RadioButtons(
        options=['Gráfico Unificado', 'Gráficos Individuales'],
        value='Gráfico Unificado',
        description='Tipo de Vista:',
        disabled=False
    )
    
    # Crear botón para actualizar gráfico
    actualizar_btn = widgets.Button(
        description='Actualizar Gráfico',
        button_style='primary', 
        tooltip='Haga clic para actualizar la visualización'
    )
    
    # Layout para organizar checkboxes
    checkbox_layout = widgets.VBox([widgets.Label("Seleccione Destinos:"), 
                                   *list(checkboxes.values())])
    
    # Layout principal con todos los controles
    controles = widgets.VBox([vista_selector, checkbox_layout, actualizar_btn])
    
    # Área de salida para mostrar el gráfico
    output = widgets.Output()
    
    # Función para actualizar gráfico cuando se presiona el botón
    def actualizar_grafico(b):
        with output:
            clear_output(wait=True)
            
            # Obtener destinos seleccionados mediante los checkboxes
            destinos_seleccionados = [d for d, cb in checkboxes.items() if cb.value]
            
            # Verificar que al menos un destino está seleccionado
            if not destinos_seleccionados:
                print("Por favor, seleccione al menos un destino.")
                return
            
            # Generar el gráfico correspondiente según el tipo de vista seleccionado
            if vista_selector.value == 'Gráfico Unificado':
                fig = generar_grafico_unificado(df_mensual, top_destinos, destinos_seleccionados)
            else:
                fig = generar_graficos_individuales(df_mensual, top_destinos, destinos_seleccionados)
            
            # Mostrar el gráfico generado
            plt.show()
    
    # Asignar callback al botón
    actualizar_btn.on_click(actualizar_grafico)
    
    # Mostrar interfaz completa
    display(widgets.VBox([controles, output]))
    
    # Generar gráfico inicial
    actualizar_grafico(None)

def guardar_visualizaciones(df_mensual, top_destinos, output_dir=OUTPUT_DIR):
    """
    Guarda automáticamente todas las visualizaciones en el directorio especificado.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    top_destinos : list
        Lista de nombres de los principales destinos.
    output_dir : str, optional
        Directorio donde guardar las visualizaciones, por defecto OUTPUT_DIR.
    
    Notes
    -----
    Genera y guarda:
    - Gráfico unificado con todos los destinos (PNG y PDF)
    - Gráfico con paneles individuales para cada destino (PNG y PDF)
    
    Los archivos se guardan con alta resolución (300 dpi) para PNG y con
    márgenes optimizados (bbox_inches='tight') para ambos formatos.
    """
    # Crear gráfico unificado
    fig_unificado = generar_grafico_unificado(df_mensual, top_destinos)
    
    # Guardar gráfico unificado en formatos PNG y PDF
    fig_unificado.savefig(os.path.join(output_dir, 'tendencias_turisticas_unificado.png'), 
                          dpi=300, bbox_inches='tight')
    fig_unificado.savefig(os.path.join(output_dir, 'tendencias_turisticas_unificado.pdf'), 
                          bbox_inches='tight')
    
    # Crear gráficos individuales
    fig_individuales = generar_graficos_individuales(df_mensual, top_destinos)
    
    # Guardar gráficos individuales en formatos PNG y PDF
    fig_individuales.savefig(os.path.join(output_dir, 'tendencias_turisticas_individuales.png'), 
                             dpi=300, bbox_inches='tight')
    fig_individuales.savefig(os.path.join(output_dir, 'tendencias_turisticas_individuales.pdf'), 
                             bbox_inches='tight')
    
    print(f"Visualizaciones guardadas en el directorio: {output_dir}")
    print(f"Archivos generados: \n- tendencias_turisticas_unificado.png/.pdf\n- tendencias_turisticas_individuales.png/.pdf")

def main():
    """
    Función principal que ejecuta el flujo completo de análisis y visualización.
    
    Esta función coordina todo el proceso:
    1. Carga y preprocesa los datos turísticos
    2. Identifica los destinos principales a analizar
    3. Prepara los datos mensuales agregados
    4. Detecta el entorno de ejecución (interactivo o no)
    5. Genera visualizaciones interactivas o estáticas según corresponda
    
    Es el punto de entrada principal del script cuando se ejecuta directamente.
    """
    print("Iniciando análisis de tendencias turísticas...")
    
    # Cargar datos
    df = cargar_datos()
    
    # Definir destinos que deben incluirse obligatoriamente
    destinos_obligatorios = ['Cataluña', 'Baleares', 'Andalucía', 'Canarias', 'Comunidad Valenciana', 'Comunidad de Madrid']
    
    # Obtener top destinos incluyendo los obligatorios
    top_destinos = obtener_top_destinos(df, num_destinos=6, destinos_fijos=destinos_obligatorios)
    
    # Preparar datos mensuales
    df_mensual = preparar_datos_mensuales(df, top_destinos)
    
    # Intentar detectar si estamos en un entorno Jupyter
    try:
        # Verificar si estamos en un entorno interactivo compatible con widgets
        get_ipython()
        is_interactive = True
    except NameError:
        is_interactive = False
    
    if is_interactive:
        print("Entorno interactivo detectado. Creando interfaz de usuario...")
        interfaz_interactiva(df_mensual, top_destinos)
    else:
        print("Entorno no interactivo. Generando y guardando visualizaciones estáticas...")
        guardar_visualizaciones(df_mensual, top_destinos)
    
    print("Análisis de tendencias turísticas completado.")
    print(f"Resultados guardados en: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()