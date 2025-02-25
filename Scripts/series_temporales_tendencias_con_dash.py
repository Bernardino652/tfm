#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script: series_temporales_tendencias_dash.py
Visualización Interactiva de Series Temporales de Tendencias Turísticas con Dash.

Este módulo genera una aplicación web interactiva para visualizar las tendencias 
turísticas en España entre 2019 y 2023. Utiliza Dash (basado en Flask) para crear
una interfaz donde el usuario puede explorar dinámicamente los datos, seleccionar
destinos específicos y alternar entre diferentes tipos de visualización.

Características:
- Aplicación web accesible desde cualquier navegador
- Panel interactivo con controles para filtrar destinos
- Alternancia entre vista unificada y vistas individuales
- Contexto temporal (periodos COVID y recuperación)
- Visualización dinámica de la estacionalidad

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos tabulares.
  Utilizada para cargar, transformar y agregar datos turísticos, manejar series
  temporales, y preparar datos para visualización.

- numpy (1.22.0+): Biblioteca para computación numérica.
  Utilizada principalmente para cálculos matemáticos como ajustes polinómicos
  para líneas de tendencia.

- dash (2.9.0+): Framework para crear aplicaciones web analíticas.
  Proporciona la base para la aplicación interactiva y la UI.
  Componentes principales utilizados: app, html, dcc.

- dash-bootstrap-components (1.0.0+): Componentes Bootstrap para Dash.
  Proporciona un diseño responsive y estético para la aplicación.

- plotly (5.13.0+): Biblioteca para gráficos interactivos.
  Utilizada para crear visualizaciones manipulables en el navegador.

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
import dash                       # Para crear la aplicación web interactiva
from dash import dcc, html        # Componentes de Dash para contenido e inputs
from dash.dependencies import Input, Output  # Para callbacks
import dash_bootstrap_components as dbc  # Componentes Bootstrap
import plotly.graph_objs as go    # Para crear gráficos interactivos
import plotly.express as px       # Para gráficos expresivos más sencillos
from datetime import datetime     # Para manipulación de fechas
import os                         # Para manejo de directorios y archivos

# Configuración para guardar archivos
OUTPUT_DIR = 'resultados_visualizacion'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar la aplicación Dash con tema Bootstrap
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                           'content': 'width=device-width, initial-scale=1.0'}])

# Título de la página
app.title = "Tendencias Turísticas España (2019-2023)"

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
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
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
    """
    x = np.arange(len(datos))
    y = datos['Num_Turistas'].values
    
    # Polinomio de grado 3 para capturar la caída por COVID y recuperación
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    
    return p(x)

def crear_grafico_unificado(df_mensual, destinos_seleccionados):
    """
    Crea un gráfico interactivo unificado con Plotly para visualizar tendencias de varios destinos.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    destinos_seleccionados : list
        Lista de destinos a incluir en el gráfico.
        
    Returns
    -------
    plotly.graph_objs.Figure
        Figura interactiva de Plotly con las tendencias turísticas.
    """
    # Definir colores para cada destino
    colores = {
        'Cataluña': '#1f77b4',       # Azul
        'Baleares': '#ff7f0e',       # Naranja
        'Andalucía': '#2ca02c',      # Verde
        'Canarias': '#d62728',       # Rojo
        'Comunidad Valenciana': '#9467bd',  # Morado
        'Comunidad de Madrid': '#8c564b'    # Marrón
    }
    
    # Crear figura base
    fig = go.Figure()
    
    # Añadir líneas para cada destino seleccionado
    for destino in destinos_seleccionados:
        datos_destino = df_mensual[df_mensual['Destino_Principal'] == destino]
        
        # Añadir línea principal
        fig.add_trace(go.Scatter(
            x=datos_destino['Fecha'],
            y=datos_destino['Num_Turistas'],
            name=destino,
            mode='lines+markers',
            line=dict(color=colores.get(destino, '#333333'), width=2),
            marker=dict(size=6)
        ))
    
    # Añadir áreas sombreadas para eventos importantes
    # Período COVID-19
    fig.add_vrect(
        x0="2020-03-01", 
        x1="2020-12-31",
        fillcolor="gray", 
        opacity=0.2,
        layer="below", 
        line_width=0,
        annotation_text="Período COVID-19",
        annotation_position="top left"
    )
    
    # Período de recuperación
    fig.add_vrect(
        x0="2021-06-01", 
        x1="2022-06-01",
        fillcolor="green", 
        opacity=0.1,
        layer="below", 
        line_width=0,
        annotation_text="Recuperación",
        annotation_position="top left"
    )
    
    # Configurar el layout
    fig.update_layout(
        title={
            'text': 'Series Temporales de Tendencias Turísticas (2019-2023)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title='Fecha',
        yaxis_title='Número de Turistas',
        legend_title='Destinos',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    # Formato para los valores del eje Y
    fig.update_yaxes(
        tickformat=",",
        separatethousands=True
    )
    
    # Añadir marca de agua
    fig.add_annotation(
        text="Fuente: Datos de turismo 2019-2023",
        xref="paper", yref="paper",
        x=0.01, y=-0.05,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    return fig

def crear_graficos_individuales(df_mensual, destinos_seleccionados):
    """
    Crea subplots individuales para cada destino seleccionado.
    
    Parameters
    ----------
    df_mensual : pandas.DataFrame
        DataFrame con datos mensuales agregados por destino.
    destinos_seleccionados : list
        Lista de destinos para los que generar gráficos individuales.
        
    Returns
    -------
    plotly.graph_objs.Figure
        Figura interactiva de Plotly con subplots para cada destino.
    """
    # Determinar número de filas y columnas para subplots
    n_graficos = len(destinos_seleccionados)
    n_rows = (n_graficos + 1) // 2  # Redondeo hacia arriba
    n_cols = min(2, n_graficos)     # Máximo 2 columnas
    
    # Definir colores para cada destino
    colores = {
        'Cataluña': '#1f77b4',       # Azul
        'Baleares': '#ff7f0e',       # Naranja
        'Andalucía': '#2ca02c',      # Verde
        'Canarias': '#d62728',       # Rojo
        'Comunidad Valenciana': '#9467bd',  # Morado
        'Comunidad de Madrid': '#8c564b'    # Marrón
    }
    
    # Crear subplots
    fig = go.Figure()
    
    # Para cada destino, crear un subplot
    for i, destino in enumerate(destinos_seleccionados):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Filtrar datos del destino
        datos_destino = df_mensual[df_mensual['Destino_Principal'] == destino].sort_values('Fecha')
        
        # Calcular tendencia
        x_indices = np.arange(len(datos_destino))
        tendencia = calcular_tendencia(datos_destino)
        
        # Añadir línea principal con datos reales
        fig.add_trace(
            go.Scatter(
                x=datos_destino['Fecha'],
                y=datos_destino['Num_Turistas'],
                name=f"{destino} - Datos",
                mode='lines+markers',
                line=dict(color=colores.get(destino, '#333333'), width=2),
                marker=dict(size=6),
                legendgroup=destino,
                showlegend=(col==1)  # Mostrar en leyenda solo una vez
            )
        )
        
        # Añadir línea de tendencia
        fig.add_trace(
            go.Scatter(
                x=datos_destino['Fecha'],
                y=tendencia,
                name=f"{destino} - Tendencia",
                mode='lines',
                line=dict(color=colores.get(destino, '#333333'), width=1, dash='dash'),
                legendgroup=destino,
                showlegend=False
            )
        )
    
    # Crear layout para subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols, 
        subplot_titles=[f"Tendencia Turística: {d}" for d in destinos_seleccionados]
    )
    
    # Añadir líneas a los subplots
    subplot_idx = 0
    for destino in destinos_seleccionados:
        row = subplot_idx // n_cols + 1
        col = subplot_idx % n_cols + 1
        subplot_idx += 1
        
        # Filtrar datos del destino
        datos_destino = df_mensual[df_mensual['Destino_Principal'] == destino].sort_values('Fecha')
        
        # Calcular tendencia
        tendencia = calcular_tendencia(datos_destino)
        
        # Añadir línea principal con datos reales
        fig.add_trace(
            go.Scatter(
                x=datos_destino['Fecha'],
                y=datos_destino['Num_Turistas'],
                name=destino,
                mode='lines+markers',
                line=dict(color=colores.get(destino, '#333333'), width=2),
                marker=dict(size=6),
                legendgroup=destino
            ),
            row=row, col=col
        )
        
        # Añadir línea de tendencia
        fig.add_trace(
            go.Scatter(
                x=datos_destino['Fecha'],
                y=tendencia,
                name=f"{destino} - Tendencia",
                mode='lines',
                line=dict(color=colores.get(destino, '#333333'), width=1, dash='dash'),
                legendgroup=destino,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Añadir áreas sombreadas para COVID-19
        fig.add_vrect(
            x0="2020-03-01", 
            x1="2020-12-31",
            fillcolor="gray", 
            opacity=0.2,
            layer="below", 
            line_width=0,
            row=row, col=col
        )
        
        # Añadir áreas sombreadas para recuperación
        fig.add_vrect(
            x0="2021-06-01", 
            x1="2022-06-01",
            fillcolor="green", 
            opacity=0.1,
            layer="below", 
            line_width=0,
            row=row, col=col
        )
    
    # Configurar layout general
    fig.update_layout(
        title_text="Tendencias Turísticas por Destino (2019-2023)",
        height=300 * n_rows,
        template='plotly_white',
        showlegend=True
    )
    
    # Configurar ejes X e Y para todos los subplots
    fig.update_xaxes(title_text="Fecha", rangeselector=dict(
        buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ))
    
    fig.update_yaxes(title_text="Número de Turistas", tickformat=",", separatethousands=True)
    
    return fig

# Cargar datos
df = cargar_datos()

# Definir destinos obligatorios
destinos_obligatorios = ['Cataluña', 'Baleares', 'Andalucía', 'Canarias', 'Comunidad Valenciana', 'Comunidad de Madrid']

# Obtener top destinos
top_destinos = obtener_top_destinos(df, num_destinos=6, destinos_fijos=destinos_obligatorios)

# Preparar datos mensuales
df_mensual = preparar_datos_mensuales(df, top_destinos)

# Importar aquí para evitar error de referencia circular
from plotly.subplots import make_subplots

# Diseño de la aplicación
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Tendencias Turísticas España (2019-2023)", 
                   className="text-center mb-4 mt-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Seleccione destinos:"),
            dbc.Card([
                dbc.CardBody([
                    # Checkboxes para seleccionar destinos
                    dbc.Checklist(
                        id="destinos-checklist",
                        options=[{"label": destino, "value": destino} for destino in top_destinos],
                        value=top_destinos,  # Todos seleccionados por defecto
                        inline=False
                    )
                ])
            ]),
            
            html.H4("Tipo de visualización:", className="mt-4"),
            dbc.Card([
                dbc.CardBody([
                    # Radio para tipo de visualización
                    dbc.RadioItems(
                        id="tipo-visualizacion",
                        options=[
                            {"label": "Gráfico Unificado", "value": "unificado"},
                            {"label": "Gráficos Individuales", "value": "individuales"}
                        ],
                        value="unificado",  # Valor por defecto
                        inline=False
                    )
                ])
            ]),
            
            html.Div([
                html.P("Eventos significativos:", className="mt-4 font-weight-bold"),
                html.Ul([
                    html.Li([
                        "Área gris: ", 
                        html.Span("Periodo COVID-19 (Mar-Dic 2020)", className="text-muted")
                    ]),
                    html.Li([
                        "Área verde: ", 
                        html.Span("Recuperación post-pandemia (Jun 2021-Jun 2022)", className="text-muted")
                    ])
                ])
            ], className="mt-3"),
            
            html.P("Fuente: Datos de turismo 2019-2023", className="text-muted small mt-5")
            
        ], md=3),
        
        dbc.Col([
            # Tarjeta para contener el gráfico
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-graph",
                        type="circle",
                        children=[
                            dcc.Graph(id="grafico-principal", style={"height": "70vh"})
                        ]
                    )
                ])
            ]),
            
            # Tarjeta para información general y análisis
            dbc.Card([
                dbc.CardBody([
                    html.H5("Análisis de Tendencias Turísticas", className="card-title"),
                    html.P([
                        "Este dashboard muestra la evolución del turismo en España entre 2019 y 2023, ",
                        "incluyendo el impacto de la pandemia COVID-19 y la posterior recuperación del sector. ",
                        "Seleccione destinos específicos y el tipo de visualización para analizar los patrones ",
                        "de estacionalidad y tendencias a largo plazo."
                    ]),
                    html.P([
                        "Desarrollado con ", 
                        html.A("Dash", href="https://dash.plotly.com/", target="_blank"),
                        " y ",
                        html.A("Plotly", href="https://plotly.com/python/", target="_blank"),
                        " utilizando datos procesados del turismo español."
                    ], className="text-muted small")
                ])
            ], className="mt-3")
            
        ], md=9)
    ], className="mt-4"),
    
], fluid=True)

# Callback para actualizar el gráfico
@app.callback(
    Output("grafico-principal", "figure"),
    [Input("destinos-checklist", "value"),
     Input("tipo-visualizacion", "value")]
)
def actualizar_grafico(destinos_seleccionados, tipo_visualizacion):
    """
    Callback para actualizar el gráfico basado en los destinos seleccionados
    y el tipo de visualización elegido.
    
    Parameters
    ----------
    destinos_seleccionados : list
        Lista de destinos seleccionados en los checkboxes.
    tipo_visualizacion : str
        Tipo de visualización seleccionada ('unificado' o 'individuales').
        
    Returns
    -------
    plotly.graph_objs.Figure
        Figura actualizada según las selecciones.
    """
    # Verificar que hay destinos seleccionados
    if not destinos_seleccionados:
        # Devolver figura vacía con mensaje
        fig = go.Figure()
        fig.add_annotation(
            text="Por favor, seleccione al menos un destino",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Generar el gráfico según el tipo de visualización
    if tipo_visualizacion == "unificado":
        return crear_grafico_unificado(df_mensual, destinos_seleccionados)
    else:
        return crear_graficos_individuales(df_mensual, destinos_seleccionados)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)