#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: LimpiezaTransformaciondeDatos.py
Limpieza y Transformación de Datos Turísticos (2019-2023).

Este script realiza el procesamiento, limpieza y transformación de un conjunto de datos 
turísticos para el período 2019-2023. El proceso incluye la carga del dataset original, 
limpieza de valores anómalos, conversión de tipos de datos, validación de la integridad
del conjunto, enriquecimiento con características adicionales, detección de anomalías
y generación de resúmenes estadísticos.

El script procesa el archivo 'turismo_completo_2019_2023.csv' y genera un dataset limpio
y enriquecido llamado 'turismo_procesado_2019_2023.csv' que puede utilizarse para análisis
posteriores y visualizaciones de tendencias turísticas.

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca principal para manipulación y análisis de datos tabulares.
  Utilizada para cargar, transformar, analizar y exportar el dataset turístico.
  Funciones clave: read_csv, to_csv, groupby, apply, to_datetime.

- numpy (1.22.0+): Biblioteca para cálculo numérico y operaciones vectorizadas.
  Utilizada principalmente para la función np.select que permite crear categorías
  condicionales de manera eficiente para la clasificación de temporadas turísticas.

- datetime: Módulo estándar de Python para manipulación de fechas y horas.
  Importado para facilitar operaciones con los campos de fecha aunque su uso
  directo es mínimo ya que se utilizan las funcionalidades de datetime de pandas.

Autor: Bernardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.1
"""

import pandas as pd  # Para manipulación de datos y análisis
import numpy as np   # Para operaciones numéricas y condicionales vectorizadas
from datetime import datetime  # Para manipulación de fechas y tiempos

def cargar_datos(archivo):
    """
    Carga el dataset desde un archivo CSV y realiza conversiones iniciales.
    
    Parameters
    ----------
    archivo : str
        Ruta al archivo CSV que contiene los datos turísticos.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con los datos cargados y columnas innecesarias eliminadas.
    
    Notes
    -----
    Se eliminan las columnas relacionadas con vuelos ya que no son necesarias
    para el análisis principal.
    """
    df = pd.read_csv(archivo, encoding='utf-8')
    columns_to_drop = ['Num_Vuelos', 'Num_Asientos', 'Num_Pasajeros']
    df = df.drop(columns=columns_to_drop)
    return df

def limpiar_datos(df):
    """
    Realiza la limpieza general del dataset, incluyendo conversión de tipos,
    normalización de strings y manejo de valores nulos.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame original con los datos cargados.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame limpio con tipos de datos consistentes y valores normalizados.
    
    Notes
    -----
    - Las conversiones numéricas utilizan coerce para convertir valores no válidos en NaN
    - Los strings son normalizados eliminando espacios al inicio y final
    """
    df_limpio = df.copy()
    
    # Asegurar que Num_Turistas sea numérico
    df_limpio['Num_Turistas'] = pd.to_numeric(df_limpio['Num_Turistas'], errors='coerce')
    
    # Normalizar strings
    columnas_texto = ['Pais', 'Destino_Principal', 'Medio_Acceso', 'Organizacion_Viaje', 'Estacionalidad']
    for columna in columnas_texto:
        df_limpio[columna] = df_limpio[columna].str.strip()
    
    # Asegurar que las columnas numéricas sean float
    df_limpio['Temperatura'] = pd.to_numeric(df_limpio['Temperatura_Media'], errors='coerce')
    df_limpio['Precipitacion'] = pd.to_numeric(df_limpio['Precipitacion_Media'], errors='coerce')
    df_limpio['Satisfaccion'] = pd.to_numeric(df_limpio['Satisfaccion'], errors='coerce')
    
    return df_limpio

def validar_datos(df):
    """
    Realiza validaciones básicas del dataset para verificar su integridad y calidad.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame limpio para validar.
    
    Returns
    -------
    dict
        Diccionario con los resultados de las validaciones, incluyendo:
        - Número total de registros
        - Conteo de valores nulos por columna
        - Número de países únicos
        - Número de destinos únicos
        - Rango de años en el dataset
        - Rango de valores de satisfacción
    """
    validaciones = {
        'registros_totales': len(df),
        'valores_nulos': df.isnull().sum().to_dict(),
        'paises_unicos': df['Pais'].nunique(),
        'destinos_unicos': df['Destino_Principal'].nunique(),
        'rango_anos': f"{df['Año'].min()} - {df['Año'].max()}",
        'rango_satisfaccion': f"{df['Satisfaccion'].min():.2f} - {df['Satisfaccion'].max():.2f}"
    }
    return validaciones

def agregar_caracteristicas(df):
    """
    Enriquece el dataset con características adicionales derivadas de los datos existentes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame limpio para enriquecer.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame enriquecido con nuevas características:
        - Fecha completa (YYYY-MM-DD)
        - Trimestre
        - Temporada (Alta, Media-Alta, Media, Baja)
    
    Notes
    -----
    La temporada se determina según el mes:
    - Alta: julio y agosto
    - Media-Alta: junio y septiembre
    - Media: abril, mayo y octubre
    - Baja: enero, febrero, marzo, noviembre y diciembre
    """
    df_nuevo = df.copy()
    
    # Crear fecha completa
    df_nuevo['Fecha'] = df_nuevo.apply(lambda x: f"{x['Año']}-{x['mes']:02d}-01", axis=1)
    df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'])
    
    # Agregar trimestre
    df_nuevo['Trimestre'] = df_nuevo['Fecha'].dt.quarter
    
    # Categorizar temporada
    condiciones = [
        df_nuevo['mes'].isin([7, 8]),                # Alta temporada (verano)
        df_nuevo['mes'].isin([6, 9]),                # Media-Alta (principio y fin de verano)
        df_nuevo['mes'].isin([4, 5, 10]),            # Media (primavera y otoño)
        df_nuevo['mes'].isin([1, 2, 3, 11, 12])      # Baja (invierno)
    ]
    valores = ['Alta', 'Media-Alta', 'Media', 'Baja']
    df_nuevo['Temporada'] = np.select(condiciones, valores, default='No Definida')
    
    return df_nuevo

def detectar_anomalias(df):
    """
    Detecta posibles anomalías en los datos que podrían requerir atención.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame enriquecido para analizar.
    
    Returns
    -------
    dict
        Diccionario con diferentes tipos de anomalías detectadas:
        - Registros con cero turistas
        - Valores de satisfacción fuera de rango esperado (5-10)
        - Temperaturas extremas (<0°C o >45°C)
        - Precipitaciones extremas (>100mm)
    
    Notes
    -----
    Estas anomalías no se eliminan automáticamente, pero se reportan para su
    posible revisión manual o tratamiento posterior.
    """
    anomalias = {
        'turistas_cero': df[df['Num_Turistas'] == 0]['Pais'].value_counts().to_dict(),
        'satisfaccion_extrema': df[
            (df['Satisfaccion'] < 5) | (df['Satisfaccion'] > 10)
        ][['Pais', 'Satisfaccion']].values.tolist(),
        'temperatura_extrema': df[
            (df['Temperatura'] < 0) | (df['Temperatura'] > 45)
        ][['Pais', 'Fecha', 'Temperatura']].values.tolist(),
        'precipitacion_extrema': df[
            df['Precipitacion'] > 100
        ][['Pais', 'Fecha', 'Precipitacion']].values.tolist()
    }
    return anomalias

def generar_resumen(df):
    """
    Genera un resumen estadístico del dataset para obtener insights rápidos.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame final procesado.
    
    Returns
    -------
    dict
        Diccionario con diversos resúmenes estadísticos:
        - Total de turistas por año
        - Distribución porcentual por medio de acceso
        - Satisfacción promedio por país
        - Temperatura promedio por temporada
        - Distribución porcentual por tipo de organización
    """
    resumen = {
        'total_turistas_por_ano': df.groupby('Año')['Num_Turistas'].sum().to_dict(),
        'distribucion_medios_acceso': df.groupby('Medio_Acceso')['Num_Turistas'].sum().div(df['Num_Turistas'].sum()) * 100,
        'satisfaccion_promedio_por_pais': df.groupby('Pais')['Satisfaccion'].mean().round(2).to_dict(),
        'temperatura_promedio_por_temporada': df.groupby('Temporada')['Temperatura'].mean().round(2).to_dict(),
        'distribucion_organizacion': df.groupby('Organizacion_Viaje')['Num_Turistas'].sum().div(df['Num_Turistas'].sum()) * 100
    }
    return resumen

def main():
    """
    Función principal que ejecuta el flujo completo de limpieza y transformación.
    
    Esta función coordina todas las etapas del procesamiento de datos, desde la carga
    inicial hasta la generación del archivo final, proporcionando mensajes informativos
    durante el proceso.
    
    Raises
    ------
    Exception
        Si ocurre algún error durante el proceso, se captura y se muestra un mensaje
        descriptivo antes de re-lanzar la excepción.
    """
    try:
        # Cargar datos
        print("Cargando datos...")
        df_original = cargar_datos('turismo_completo_2019_2023.csv')
        
        # Proceso de limpieza
        print("Limpiando datos...")
        df_limpio = limpiar_datos(df_original)
        
        # Validaciones
        print("\nRealizando validaciones básicas:")
        validaciones = validar_datos(df_limpio)
        for k, v in validaciones.items():
            print(f"{k}: {v}")
        
        # Agregar características
        print("\nAgregando características adicionales...")
        df_final = agregar_caracteristicas(df_limpio)
        
        # Detectar anomalías
        print("\nDetectando anomalías:")
        anomalias = detectar_anomalias(df_final)
        for k, v in anomalias.items():
            if v:  # Solo mostrar si hay anomalías
                print(f"\n{k}:")
                print(v)
        
        # Generar resumen
        print("\nGenerando resumen estadístico:")
        resumen = generar_resumen(df_final)
        for k, v in resumen.items():
            print(f"\n{k}:")
            print(v)
        
        # Guardar dataset procesado
        print("\nGuardando dataset procesado...")
        df_final.to_csv('turismo_procesado_2019_2023.csv', index=False, encoding='utf-8')
        print("Proceso completado con éxito!")
        print(df_final)
        print(df_final.columns)
        
    except Exception as e:
        print(f"Error en el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()