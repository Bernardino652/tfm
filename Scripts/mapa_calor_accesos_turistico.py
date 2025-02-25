#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: mapa_calor_accesos_turistico.py
Generación de Mapas de Calor de Accesos Turísticos por Comunidad Autónoma.

Este script genera visualizaciones en forma de mapas de calor que muestran la distribución
porcentual de los diferentes medios de transporte (aeropuerto, carretera, puerto y tren)
utilizados por los turistas para acceder a cada comunidad autónoma de España. El resultado
es una representación visual geográfica que permite identificar fácilmente los patrones
de acceso turístico según la región y el tipo de transporte preferido.

El proceso completo incluye:
- Carga de datos turísticos procesados (2019-2023)
- Análisis y agrupación de datos por destino y medio de acceso
- Cálculo de distribuciones porcentuales
- Generación de una visualización con cuatro paneles (uno por cada medio de transporte)
- Presentación de resultados en un formato de mapa esquemático de España

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos estructurados.
  Utilizada para cargar el dataset CSV, realizar agrupaciones, calcular estadísticas
  y crear tablas pivote con distribuciones porcentuales.
  Funciones clave: read_csv, groupby, pivot_table, reset_index.

- matplotlib.pyplot (3.5.0+): Biblioteca principal para visualización.
  Utilizada para crear la estructura general del mapa, los rectángulos que representan
  cada comunidad autónoma, las escalas de color y la gestión de textos y títulos.
  Funciones clave: figure, GridSpec, subplot, Rectangle, colorbar, savefig.

- numpy (1.22.0+): Biblioteca para operaciones numéricas.
  Aunque importada, su uso directo en este script es limitado, pero proporciona
  soporte para operaciones matriciales que podrían ser necesarias en ampliaciones.

- os: Módulo estándar para interactuar con el sistema operativo.
  Utilizado para gestionar la creación de directorios y rutas de archivos
  para guardar las visualizaciones generadas.
  Funciones clave: makedirs, path.join.

Autor: Bernardino Chancusig Es[pin
Fecha: 25/02/2025
Versión: 1.1
"""

# Importación de librerías
import pandas as pd             # Para manipulación y análisis de datos
import matplotlib.pyplot as plt  # Para visualizaciones
import numpy as np              # Para operaciones numéricas (uso limitado)
import os                       # Para manejo de directorios y archivos

# Configuración para guardar archivos
OUTPUT_DIR = 'resultados_visualizacion'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cargar_datos(filepath='turismo_procesado_2019_2023.csv'):
    """
    Carga los datos turísticos desde un archivo CSV.
    
    Parameters
    ----------
    filepath : str, optional
        Ruta al archivo CSV con los datos procesados,
        por defecto 'turismo_procesado_2019_2023.csv'.
        
    Returns
    -------
    pandas.DataFrame o None
        DataFrame con los datos turísticos si la carga es exitosa, 
        None si ocurre un error.
        
    Notes
    -----
    La función maneja excepciones y muestra mensajes informativos
    sobre el proceso de carga y posibles errores.
    """
    try:
        print(f"Cargando datos desde {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Datos cargados correctamente. {len(df)} registros encontrados.")
        return df
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None


def analizar_destinos(df):
    """
    Analiza todos los destinos en el dataset y sus distribuciones de medios de acceso.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos turísticos.
        
    Returns
    -------
    pandas.Series
        Serie con los destinos ordenados por número total de turistas.
        
    Notes
    -----
    Genera dos informes en la consola:
    1. Listado de destinos ordenados por volumen de turistas
    2. Distribución porcentual de medios de acceso para cada destino
    """
    # Agrupar por destino y sumar el número de turistas
    destinos = df.groupby('Destino_Principal')['Num_Turistas'].sum().sort_values(ascending=False)
    
    # Mostrar listado de destinos
    print(f"\n===== DESTINOS PARA MAPAS DE CALOR DE ACCESOS TURÍSTICOS =====")
    for i, (destino, turistas) in enumerate(destinos.items(), 1):
        print(f"{i}. {destino}: {turistas:,} turistas")
    
    # Mostrar distribución por medio de acceso para cada destino
    print("\n===== DISTRIBUCIÓN POR MEDIO DE ACCESO =====")
    for destino in destinos.index:
        print(f"\nDestino: {destino}")
        destino_df = df[df['Destino_Principal'] == destino]
        medios = destino_df.groupby('Medio_Acceso')['Num_Turistas'].sum()
        total = medios.sum()
        
        # Calcular y mostrar porcentajes para cada medio de acceso
        for medio, num in sorted(medios.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (num / total) * 100
            print(f"  - {medio}: {porcentaje:.1f}%")
    
    return destinos


def crear_mapas_calor_accesos_turisticos(df):
    """
    Genera los mapas de calor que muestran la distribución de medios de acceso por comunidad autónoma.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos turísticos.
        
    Notes
    -----
    Crea una visualización con cuatro paneles, uno para cada medio de acceso
    (Aeropuerto, Carretera, Puerto, Tren). Cada panel muestra un mapa esquemático
    de España con las comunidades autónomas representadas como rectángulos,
    coloreados según el porcentaje de turistas que utilizan ese medio de acceso.
    
    La visualización se guarda como un archivo PNG en el directorio de salida.
    """
    # Obtener todos los destinos del dataset
    destinos_datos = df.groupby('Destino_Principal')['Num_Turistas'].sum().sort_values(ascending=False)
    destinos_lista = destinos_datos.index.tolist()
    
    print(f"Destinos detectados para los mapas de calor: {', '.join(destinos_lista)}")
    
    # Normalizar los nombres para asegurar consistencia en la visualización
    mapping_nombres = {
        'Comunidad de Madrid': 'Madrid',
        'Comunidad Valenciana': 'C. Valenciana',
        'Castilla y León': 'Castilla y León',
        'Castilla-La Mancha': 'C. La Mancha'
    }
    
    # Calcular porcentajes por medio de acceso para cada destino
    regiones_acceso = df.groupby(['Destino_Principal', 'Medio_Acceso'])['Num_Turistas'].sum().reset_index()
    pivot_table = regiones_acceso.pivot_table(
        values='Num_Turistas', 
        index='Destino_Principal', 
        columns='Medio_Acceso'
    ).fillna(0)
    
    # Normalizar por fila (destino) - calcular porcentajes
    porcentajes = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
    
    # Definir posiciones para la cuadrícula, simulando la geografía de España
    # Formato: (x, y) donde x es posición horizontal e y es posición vertical
    posiciones = {
        # Fila 1 (espacio para título)
        'Galicia': (0, 2),           # Noroeste
        'Asturias': (1, 2),
        'Cantabria': (2, 2),
        'País Vasco': (3, 2),        # Norte
        # Fila 2
        'Castilla y León': (0, 3),
        'La Rioja': (1, 3),
        'Navarra': (2, 3),
        'Aragón': (3, 3),            # Noreste
        # Fila 3
        'Madrid': (0, 4),            # Centro
        'C. La Mancha': (1, 4),
        'C. Valenciana': (2, 4),
        'Cataluña': (3, 4),          # Este
        # Fila 4
        'Extremadura': (0, 5),       # Suroeste
        'Andalucía': (1, 5),
        'Murcia': (2, 5),
        'Baleares': (3, 5),          # Islas Mediterráneo
        # Fila 5
        'Ceuta': (0, 6),             # Ciudades autónomas
        'Melilla': (1, 6),
        'Canarias': (2, 6)           # Islas Atlántico
    }
    
    # Crear figura principal
    plt.figure(figsize=(20, 18))
    
    # Título principal
    plt.suptitle('Mapas de calor de accesos turísticos', 
                 fontsize=22, y=0.98, fontweight='bold')
    
    # Subtítulo
    plt.figtext(0.5, 0.95, 
                "Porcentaje de turistas según medio de transporte utilizado para acceder a cada región", 
                ha='center', fontsize=16)
    
    # Crear estructura de subplots (2x2)
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                     left=0.05, right=0.95, bottom=0.05, top=0.9, 
                     wspace=0.2, hspace=0.3)
    
    # Crear los cuatro paneles para cada medio de acceso
    axes = []
    for i in range(4):
        row, col = divmod(i, 2)
        axes.append(plt.subplot(gs[row, col]))
    
    # Definir medios de acceso a representar
    medios_acceso = ['Aeropuerto', 'Carretera', 'Puerto', 'Tren']
    
    # Generar cada panel (uno por medio de acceso)
    for i, medio in enumerate(medios_acceso):
        ax = axes[i]
        
        # Configurar el área de dibujo con espacio para el título
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(1.0, 7.0)
        
        # Elegir esquema de color según el medio de acceso
        if medio == 'Aeropuerto':
            cmap = plt.cm.Blues
            title = 'Acceso por Aeropuerto (%)'
        elif medio == 'Carretera':
            cmap = plt.cm.Greens
            title = 'Acceso por Carretera (%)'
        elif medio == 'Puerto':
            cmap = plt.cm.Purples
            title = 'Acceso por Puerto (%)'
        else:  # Tren
            cmap = plt.cm.Oranges
            title = 'Acceso por Tren (%)'
        
        # Encontrar valores para la escala de color (mínimo y máximo)
        valores_validos = []
        
        for region_original in destinos_lista:
            # Aplicar mapeo de nombres si es necesario
            region = mapping_nombres.get(region_original, region_original)
            
            if region_original in porcentajes.index and medio in porcentajes.columns:
                valores_validos.append(porcentajes.loc[region_original, medio])
        
        vmin = min(valores_validos) if valores_validos else 0
        vmax = max(valores_validos) if valores_validos else 100
        
        # Dibujar cada comunidad autónoma como un rectángulo
        for region, (x, y) in posiciones.items():
            # Encontrar el nombre original si fuera necesario (desmapeo)
            region_original = region
            for orig, mapeado in mapping_nombres.items():
                if mapeado == region:
                    region_original = orig
                    break
            
            # Determinar si esta región existe en el dataset
            es_region_del_dataset = region_original in destinos_lista or region in destinos_lista
            
            # Determinar el color según el valor porcentual
            if region_original in porcentajes.index and medio in porcentajes.columns:
                valor = porcentajes.loc[region_original, medio]
                # Normalizar valor entre 0 y 1 para el mapa de color
                norm_valor = (valor - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = cmap(norm_valor)
            else:
                # Intentar con región sin mapeo
                if region in porcentajes.index and medio in porcentajes.columns:
                    valor = porcentajes.loc[region, medio]
                    norm_valor = (valor - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    color = cmap(norm_valor)
                else:
                    # Gris para regiones sin datos
                    color = '#AAAAAA' if es_region_del_dataset else 'lightgrey'
                    valor = None
            
            # Determinar estilo de borde según si existe en el dataset
            borde = 'red' if es_region_del_dataset else 'black'
            grosor = 2 if es_region_del_dataset else 1
            
            # Dibujar el rectángulo para la comunidad autónoma
            rect = plt.Rectangle((x, y), 0.9, 0.9, 
                                facecolor=color, 
                                edgecolor=borde, 
                                linewidth=grosor)
            ax.add_patch(rect)
            
            # Añadir nombre de la región
            ax.text(x + 0.45, y + 0.7, region, 
                   ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Añadir valor porcentual (si existe)
            if valor is not None:
                ax.text(x + 0.45, y + 0.35, f"{valor:.1f}%", 
                       ha='center', va='center', fontsize=7)
            elif es_region_del_dataset:
                ax.text(x + 0.45, y + 0.35, "Sin datos", 
                       ha='center', va='center', fontsize=6, style='italic')
        
        # Configurar aspecto final del panel
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Título bien visible con fondo
        title_x = 0.5
        title_y = 1.5
        
        # Colocar un rectángulo blanco detrás del título para hacerlo destacar
        ax.text(title_x, title_y, title, 
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='gray', pad=5.0))
        
        # Línea separadora entre título y cuadros
        ax.axhline(y=1.5, xmin=0.1, xmax=0.9, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Añadir barra de color para interpretar los valores
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('%')
    
    # Guardar la visualización final
    mapa_path = os.path.join(OUTPUT_DIR, 'mapas_calor_accesos_turisticos.png')
    plt.savefig(mapa_path, dpi=300, bbox_inches='tight')
    print(f"Mapas de calor de accesos turísticos guardados en {mapa_path}")
    plt.close()


def main():
    """
    Función principal que ejecuta el proceso completo de generación de mapas de calor.
    
    Esta función coordina la carga de datos, el análisis de destinos y la generación
    de los mapas de calor, mostrando mensajes informativos durante el proceso.
    """
    print("Iniciando generación de Mapas de calor de accesos turísticos...")
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        print("No se pudieron cargar los datos. Terminando programa.")
        return
    
    # Analizar destinos para obtener estadísticas
    destinos = analizar_destinos(df)
    
    # Crear los mapas de calor
    crear_mapas_calor_accesos_turisticos(df)
    
    print("\nProceso completado exitosamente.")
    print(f"Visualización guardada en: {os.path.abspath(os.path.join(OUTPUT_DIR, 'mapas_calor_accesos_turisticos.png'))}")
    print("\nRevisa la carpeta 'resultados_visualizacion' para ver los mapas de calor generados.")


if __name__ == "__main__":
    main()