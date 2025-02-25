#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: generar_dataset_histórico_predictivo.py
Generador de Dataset Histórico-Predictivo para Turismo (2019-2028).

Este script genera un conjunto de datos extendido para análisis y predicción turística,
combinando datos históricos reales (2019-2023) con proyecciones futuras (2024-2028)
basadas en patrones de crecimiento, estacionalidad y tendencias climáticas observadas.
El objetivo es crear un dataset completo para entrenar y probar modelos predictivos
que puedan anticipar la evolución del turismo en España.

El proceso incluye:
- Carga de datos históricos procesados
- Cálculo de patrones de crecimiento específicos por país
- Análisis de estacionalidad mensual por país emisor
- Proyección de tendencias climáticas por temporada
- Generación de datos futuros con ajustes por país, temporada y medio de acceso

El dataset resultante mantiene la misma estructura que los datos históricos,
pero incluye una columna adicional que identifica el origen de cada registro
('Histórico' o 'Datos Generados').

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos estructurados.
  Utilizada para cargar datos históricos, realizar agregaciones y transformaciones,
  manipular series temporales y exportar el dataset combinado resultante.
  Funciones clave: read_csv, to_datetime, groupby, concat.

- numpy (1.22.0+): Biblioteca para computación numérica.
  Utilizada para cálculos matemáticos como potencias y operaciones con arrays.

- datetime: Módulo estándar de Python para manipulación de fechas y horas.
  Importado para facilitar operaciones con los campos de fecha aunque su uso
  directo es mínimo, ya que se utilizan principalmente las funcionalidades
  de datetime incluidas en pandas.

- sklearn.preprocessing.LabelEncoder: Componente de scikit-learn para codificación
  de variables categóricas. Importado para mantener consistencia con posibles
  requisitos del pipeline de procesamiento, aunque no se utiliza directamente
  en este script.

- joblib: Biblioteca para serialización y paralelización en Python.
  Importada para posibles operaciones de serialización, aunque no se utiliza
  directamente en esta versión del script.

- os: Módulo estándar para interactuar con el sistema operativo.
  Importado para posible manipulación de rutas de archivos, aunque no se utiliza
  directamente en esta versión del script.

Autor: Bernardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.1
"""

# Importación de librerías
import pandas as pd                     # Para manipulación y análisis de datos
import numpy as np                      # Para cálculos numéricos
from datetime import datetime           # Para manipulación de fechas
from sklearn.preprocessing import LabelEncoder  # Para codificación de variables categóricas
import joblib                           # Para serialización de objetos
import os                               # Para manejo de directorios y archivos


class TourismDataGenerator:
    """
    Clase para generar datos turísticos proyectados a futuro basados en patrones históricos.
    
    Esta clase encapsula toda la lógica necesaria para analizar patrones en datos
    históricos de turismo y proyectar tendencias futuras, manteniendo la consistencia
    con patrones de estacionalidad, crecimiento y variaciones climáticas.
    
    Attributes
    ----------
    historical_data : pandas.DataFrame
        DataFrame con los datos históricos cargados del CSV.
    growth_patterns : dict
        Diccionario con tasas de crecimiento por país.
    seasonal_patterns : dict
        Diccionario con patrones estacionales por país y mes.
    climate_trends : dict
        Diccionario con tendencias climáticas por temporada.
    """
    
    def __init__(self, historical_data_path='turismo_procesado_2019_2023.csv'):
        """
        Inicializa el generador de datos turísticos cargando datos históricos
        y calculando patrones de crecimiento, estacionalidad y tendencias climáticas.
        
        Parameters
        ----------
        historical_data_path : str, optional
            Ruta al archivo CSV con datos históricos procesados,
            por defecto 'turismo_procesado_2019_2023.csv'.
            
        Notes
        -----
        Al inicializar, la clase calcula automáticamente los patrones base 
        que se utilizarán para las proyecciones futuras, incluyendo:
        - Tasas de crecimiento específicas por país
        - Patrones estacionales mensuales
        - Tendencias climáticas por temporada
        """
        self.historical_data = pd.read_csv(historical_data_path)
        self.historical_data['Fecha'] = pd.to_datetime(self.historical_data['Fecha'])
        
        # Calcular patrones básicos para proyecciones
        self.growth_patterns = self._calculate_growth_patterns()
        self.seasonal_patterns = self._calculate_seasonal_patterns()
        self.climate_trends = self._calculate_climate_trends()

    def _calculate_growth_patterns(self):
        """
        Calcula las tasas de crecimiento anual compuesto por país.
        
        Returns
        -------
        dict
            Diccionario con las tasas de crecimiento para cada país.
            
        Notes
        -----
        El cálculo se basa en la tasa de crecimiento anual compuesto (CAGR)
        desde el primer al último año de datos históricos. Adicionalmente,
        se aplican factores de ajuste según la proximidad geográfica:
        - Países cercanos (Francia, Portugal, Reino Unido): +10% de crecimiento
        - Países lejanos (China, Japón, Corea del Sur): -5% de crecimiento
        
        Para países con datos insuficientes, se asigna una tasa de crecimiento
        predeterminada del 5%.
        """
        patterns = {}
        for country in self.historical_data['Pais'].unique():
            country_data = self.historical_data[self.historical_data['Pais'] == country]
            yearly_tourists = country_data.groupby('Año')['Num_Turistas'].sum()
            
            # Calcular tasa de crecimiento compuesto
            if len(yearly_tourists) > 1:
                growth_rate = (yearly_tourists.iloc[-1] / yearly_tourists.iloc[0]) ** (1/(len(yearly_tourists)-1))
                # Ajustar crecimiento basado en distancia y conectividad
                if country in ['Francia', 'Portugal', 'Reino Unido']:
                    growth_rate *= 1.1  # Mayor crecimiento para países cercanos
                elif country in ['China', 'Japón', 'Corea del Sur']:
                    growth_rate *= 0.95  # Menor crecimiento para países lejanos
            else:
                growth_rate = 1.05  # Valor predeterminado para países con datos insuficientes
                
            patterns[country] = growth_rate
        return patterns

    def _calculate_seasonal_patterns(self):
        """
        Calcula patrones estacionales mensuales por país.
        
        Returns
        -------
        dict
            Diccionario con patrones estacionales (índices mensuales) por país.
            
        Notes
        -----
        Para cada país, calcula la distribución mensual normalizada del número de turistas.
        El índice estacional representa la proporción de turistas en cada mes respecto
        a la media mensual anual (un valor de 2.0 significa el doble del promedio).
        """
        patterns = {}
        for country in self.historical_data['Pais'].unique():
            country_data = self.historical_data[self.historical_data['Pais'] == country]
            monthly_avg = country_data.groupby('Mes')['Num_Turistas'].mean()
            yearly_avg = monthly_avg.mean()
            patterns[country] = monthly_avg / yearly_avg  # Índice estacional
        return patterns

    def _calculate_climate_trends(self):
        """
        Define tendencias climáticas por temporada para proyecciones futuras.
        
        Returns
        -------
        dict
            Diccionario con tendencias de temperatura y precipitación por temporada.
            
        Notes
        -----
        Establece tendencias climáticas simplificadas basadas en escenarios
        de cambio climático moderados:
        - Incremento anual de temperatura de 0.02°C para todas las temporadas
        - Reducción gradual de precipitaciones en verano (-5%)
        - Aumento gradual de precipitaciones en invierno (+5%)
        - Estabilidad en precipitaciones para primavera y otoño
        """
        # Definir temporadas según meses
        seasons = {
            'verano': [6, 7, 8],
            'invierno': [12, 1, 2],
            'primavera': [3, 4, 5],
            'otoño': [9, 10, 11]
        }
        
        # Definir tendencias climáticas
        trends = {}
        for season, months in seasons.items():
            season_data = self.historical_data[self.historical_data['Mes'].isin(months)]
            trends[season] = {
                'temp_trend': 0.02,  # Incremento anual de temperatura (°C)
                'precip_adjustment': {
                    'verano': 0.95,    # Menos precipitación en verano (-5% anual)
                    'invierno': 1.05,  # Más precipitación en invierno (+5% anual)
                    'primavera': 1.0,  # Sin cambio en primavera
                    'otoño': 1.0       # Sin cambio en otoño
                }[season]
            }
        return trends

    def generate_future_data(self, start_year=2024, end_year=2028):
        """
        Genera datos turísticos para años futuros basados en patrones históricos.
        
        Parameters
        ----------
        start_year : int, optional
            Año inicial para la generación de datos, por defecto 2024.
        end_year : int, optional
            Año final para la generación de datos, por defecto 2028.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame con los datos generados para el período futuro especificado.
            
        Notes
        -----
        Para cada combinación de país, mes y medio de acceso, genera registros futuros
        aplicando los patrones de crecimiento, estacionalidad y tendencias climáticas
        calculados previamente. Toma como base el último registro histórico disponible
        para cada combinación.
        """
        future_records = []
        
        # Para cada país en el conjunto de datos históricos
        for country in self.historical_data['Pais'].unique():
            growth_rate = self.growth_patterns[country]  # Tasa de crecimiento específica por país
            seasonal_pattern = self.seasonal_patterns[country]  # Patrón estacional por país
            
            # Para cada año en el rango futuro
            for year in range(start_year, end_year + 1):
                # Para cada mes del año
                for month in range(1, 13):
                    # Base: último registro histórico para este país/mes
                    base_data = self.historical_data[
                        (self.historical_data['Pais'] == country) &
                        (self.historical_data['Mes'] == month)
                    ].copy()
                    
                    # Verificar que hay datos históricos para esta combinación
                    if len(base_data) == 0:
                        continue
                    
                    # Para cada medio de acceso posible
                    for medio_acceso in ['Aeropuerto', 'Carretera', 'Puerto', 'Tren']:
                        # Seleccionar el último registro histórico como base
                        record = base_data[base_data['Medio_Acceso'] == medio_acceso].iloc[-1].copy()
                        
                        # Ajustar año y fecha para el registro futuro
                        record['Año'] = year
                        record['Fecha'] = pd.to_datetime(f'{year}-{month:02d}-01')
                        
                        # Proyectar número de turistas aplicando crecimiento y estacionalidad
                        years_diff = year - 2023  # Diferencia de años respecto al último año histórico
                        base_tourists = record['Num_Turistas']  # Número base de turistas
                        seasonal_factor = seasonal_pattern[month]  # Factor estacional del mes
                        
                        # Calcular proyección
                        record['Num_Turistas'] = int(base_tourists * (growth_rate ** years_diff) * seasonal_factor)
                        
                        # Ajustar variables climáticas según tendencias
                        season = self._get_season(month)  # Obtener temporada del mes
                        
                        # Proyectar temperatura
                        record['Temperatura_Media'] += self.climate_trends[season]['temp_trend'] * years_diff
                        
                        # Proyectar precipitación
                        record['Precipitacion_Media'] *= self.climate_trends[season]['precip_adjustment'] ** years_diff
                        
                        # Añadir registro a la lista de registros futuros
                        future_records.append(record)
        
        # Crear DataFrame con todos los registros futuros
        future_df = pd.DataFrame(future_records)
        future_df['Tipo_Dato'] = 'Datos Generados'  # Marcar como datos generados
        
        return future_df

    def _get_season(self, month):
        """
        Determina la temporada correspondiente a un mes.
        
        Parameters
        ----------
        month : int
            Número de mes (1-12).
            
        Returns
        -------
        str
            Nombre de la temporada ('invierno', 'primavera', 'verano', 'otoño').
        """
        if month in [12, 1, 2]: 
            return 'invierno'
        elif month in [3, 4, 5]: 
            return 'primavera'
        elif month in [6, 7, 8]: 
            return 'verano'
        else: 
            return 'otoño'

    def generate_complete_dataset(self):
        """
        Genera un dataset completo combinando datos históricos y proyecciones futuras.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame combinado con datos históricos y generados, ordenados cronológicamente.
            
        Notes
        -----
        El dataset resultante mantiene la misma estructura que los datos históricos,
        pero incluye una columna adicional 'Tipo_Dato' que identifica el origen
        de cada registro ('Histórico' o 'Datos Generados').
        """
        # Preparar datos históricos con etiqueta
        historical_df = self.historical_data.copy()
        historical_df['Tipo_Dato'] = 'Histórico'
        
        # Generar datos futuros
        future_df = self.generate_future_data()
        
        # Combinar datasets
        combined_df = pd.concat([historical_df, future_df], ignore_index=True)
        combined_df.sort_values(['Fecha', 'Pais'], inplace=True)
        
        # Mostrar resumen del dataset generado
        print("\nResumen del dataset generado:")
        print(f"Total registros: {len(combined_df)}")
        print(f"Período: {combined_df['Año'].min()} - {combined_df['Año'].max()}")
        print("\nDistribución por tipo de dato:")
        print(combined_df['Tipo_Dato'].value_counts())
        
        return combined_df


def main():
    """
    Función principal que ejecuta el proceso de generación del dataset completo.
    
    Esta función inicializa el generador de datos, crea el dataset combinado
    (histórico + futuro) y lo guarda en un archivo CSV.
    """
    # Inicializar generador de datos
    generator = TourismDataGenerator()
    
    # Generar dataset completo
    dataset_completo = generator.generate_complete_dataset()
    
    # Guardar dataset en archivo CSV
    output_file = 'dataset_completo_2019_2028.csv'
    dataset_completo.to_csv(output_file, index=False)
    print(f"\nDataset guardado exitosamente en: {output_file}")
    print(f"Dimensiones del dataset: {dataset_completo.shape[0]} filas x {dataset_completo.shape[1]} columnas")


if __name__ == "__main__":
    main()