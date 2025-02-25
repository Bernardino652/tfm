#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: comparacion_historicos_predicciones_con_analisis_confianza.py
Comparación de Datos Históricos vs Predicciones con Análisis de Confianza.

Este script analiza y compara datos turísticos históricos (2019-2023) con las
predicciones generadas por modelos de aprendizaje automático (2024-2028), 
centrándose en los medios de acceso utilizados por los turistas. El análisis incluye
comparaciones de distribución, evolución temporal y evaluación de la confianza de
las predicciones, generando tanto visualizaciones gráficas como tablas de datos
detalladas para uso en informes y dashboards.

El proceso completo incluye:
- Carga del dataset combinado (histórico + predicciones)
- Separación de datos históricos y predicciones
- Generación de visualizaciones comparativas
- Análisis de patrones y tendencias
- Evaluación de la confianza de las predicciones
- Generación de tablas detalladas para análisis adicional

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos estructurados.
  Utilizada para cargar datos, realizar agrupaciones, pivots y cálculos estadísticos, 
  además de exportar resultados en formato CSV.
  Funciones clave: read_csv, groupby, pivot_table, cut, value_counts.

- numpy (1.22.0+): Biblioteca para computación numérica.
  Utilizada para cálculos matemáticos y operaciones con arrays,
  aunque su uso directo es limitado en este script.

- matplotlib.pyplot (3.5.0+): Biblioteca principal para visualización.
  Utilizada para crear gráficos de barras, líneas temporales y establecer
  configuraciones generales de visualización.
  Funciones clave: figure, subplots, plot, savefig.

- seaborn (0.11.0+): Biblioteca para visualización estadística.
  Utilizada para crear visualizaciones estadísticas más avanzadas como
  boxplots para el análisis de confianza.
  Funciones clave: boxplot.

Autor: Bernardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.0
"""

# Importación de librerías
import pandas as pd             # Para manipulación y análisis de datos
import numpy as np              # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualizaciones
import seaborn as sns           # Para visualizaciones estadísticas avanzadas


class PredictionAnalyzer:
    """
    Clase para analizar y comparar datos históricos y predicciones de turismo.
    
    Esta clase encapsula todas las funcionalidades necesarias para cargar,
    procesar, visualizar y analizar datos turísticos históricos (2019-2023) 
    y predicciones futuras (2024-2028), con un enfoque especial en la 
    distribución de medios de acceso y la confianza de las predicciones.
    
    Attributes
    ----------
    data : pandas.DataFrame
        Dataset completo que contiene datos históricos y predicciones.
    historical : pandas.DataFrame
        Subconjunto de datos históricos (años <= 2023).
    predictions : pandas.DataFrame
        Subconjunto de predicciones (años > 2023).
    """
    
    def __init__(self, data_path='predicciones_finales_2019_2028.csv'):
        """
        Inicializa el analizador cargando los datos y separando históricos de predicciones.
        
        Parameters
        ----------
        data_path : str, optional
            Ruta al archivo CSV con datos históricos y predicciones,
            por defecto 'predicciones_finales_2019_2028.csv'.
            
        Notes
        -----
        Durante la inicialización, la clase:
        1. Carga el dataset desde el archivo CSV
        2. Convierte la columna 'Fecha' a formato datetime
        3. Separa los datos en dos subconjuntos: histórico (2019-2023) y 
           predicciones (2024-2028)
        """
        # Cargar datos
        self.data = pd.read_csv(data_path)
        self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
        
        # Separar datos históricos y predicciones
        self.historical = self.data[self.data['Año'] <= 2023]
        self.predictions = self.data[self.data['Año'] > 2023]

    def plot_distribution_comparison(self):
        """
        Genera un gráfico comparativo de la distribución de medios de acceso
        entre datos históricos y predicciones.
        
        Notes
        -----
        Crea un gráfico con dos paneles:
        - Panel izquierdo: Distribución porcentual de medios de acceso en datos históricos
        - Panel derecho: Distribución porcentual de medios de acceso en predicciones
        
        El gráfico se guarda como 'distribucion_comparacion.png'.
        """
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribución histórica (panel izquierdo)
        hist_dist = self.historical['Medio_Acceso'].value_counts(normalize=True)
        hist_dist.plot(kind='bar', ax=ax1, title='Distribución Histórica (2019-2023)')
        ax1.set_ylabel('Proporción')
        
        # Distribución predicha (panel derecho)
        pred_dist = self.predictions['Medio_Acceso_Predicho'].value_counts(normalize=True)
        pred_dist.plot(kind='bar', ax=ax2, title='Distribución Predicha (2024-2028)')
        ax2.set_ylabel('Proporción')
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig('distribucion_comparacion.png')
        plt.close()
        
        print(f"Gráfico de comparación de distribuciones guardado: distribucion_comparacion.png")

    def plot_temporal_evolution(self):
        """
        Visualiza la evolución temporal del número de turistas por medio de acceso,
        mostrando tanto datos históricos como predicciones.
        
        Notes
        -----
        Crea un gráfico de líneas que muestra la evolución anual del número
        de turistas para cada medio de acceso, con líneas continuas para datos
        históricos y líneas discontinuas para predicciones. Incluye una línea
        vertical que marca la separación entre datos históricos y predicciones.
        
        El gráfico se guarda como 'evolucion_temporal.png'.
        """
        # Crear figura
        plt.figure(figsize=(15, 8))
        
        # Preparar datos agrupados
        historical_grouped = self.historical.groupby(['Año', 'Medio_Acceso'])['Num_Turistas'].sum().reset_index()
        predictions_grouped = self.predictions.groupby(['Año', 'Medio_Acceso_Predicho'])['Num_Turistas'].sum().reset_index()
        
        # Graficar cada medio de acceso
        for medio in self.historical['Medio_Acceso'].unique():
            # Datos históricos
            hist_data = historical_grouped[historical_grouped['Medio_Acceso'] == medio]
            plt.plot(hist_data['Año'], hist_data['Num_Turistas'], 
                    marker='o', label=f'Histórico - {medio}')
            
            # Predicciones
            pred_data = predictions_grouped[predictions_grouped['Medio_Acceso_Predicho'] == medio]
            plt.plot(pred_data['Año'], pred_data['Num_Turistas'], 
                    linestyle='--', marker='x', label=f'Predicción - {medio}')
        
        # Añadir línea vertical en 2023 (separación histórico/predicción)
        plt.axvline(x=2023, color='gray', linestyle=':', label='Límite histórico/predicción')
        
        # Configurar gráfico
        plt.title('Evolución de Medios de Acceso (2019-2028)')
        plt.xlabel('Año')
        plt.ylabel('Número de Turistas')
        plt.legend()
        plt.grid(True)
        
        # Guardar y cerrar
        plt.savefig('evolucion_temporal.png')
        plt.close()
        
        print(f"Gráfico de evolución temporal guardado: evolucion_temporal.png")

    def plot_confidence_analysis(self):
        """
        Genera visualizaciones para analizar la confianza de las predicciones.
        
        Notes
        -----
        Crea un gráfico con dos paneles:
        - Panel izquierdo: Boxplot que muestra la distribución de confianza
          para cada medio de acceso predicho
        - Panel derecho: Gráfico de barras con la distribución de predicciones
          por nivel de confianza (Baja, Media, Alta, Muy Alta)
        
        El gráfico se guarda como 'analisis_confianza.png'.
        """
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot de confianza por medio de acceso (panel izquierdo)
        sns.boxplot(data=self.predictions, x='Medio_Acceso_Predicho', 
                   y='Confianza_Prediccion', ax=ax1)
        ax1.set_title('Distribución de Confianza por Medio de Acceso')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_xlabel('Medio de Acceso')
        ax1.set_ylabel('Nivel de Confianza')
        
        # Distribución de niveles de confianza (panel derecho)
        conf_levels = pd.cut(self.predictions['Confianza_Prediccion'], 
                           bins=[0, 0.25, 0.5, 0.75, 1.0], 
                           labels=['Baja', 'Media', 'Alta', 'Muy Alta'])
        conf_levels.value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title('Distribución de Niveles de Confianza')
        ax2.set_xlabel('Nivel de Confianza')
        ax2.set_ylabel('Número de Predicciones')
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig('analisis_confianza.png')
        plt.close()
        
        print(f"Gráfico de análisis de confianza guardado: analisis_confianza.png")

    def analyze_patterns(self):
        """
        Analiza patrones y genera métricas comparativas entre datos históricos y predicciones.
        
        Notes
        -----
        Realiza y muestra:
        1. Comparación de distribuciones porcentuales de medios de acceso
        2. Cálculo de tasas de crecimiento anual promedio por medio de acceso,
           tanto para datos históricos como para predicciones
        
        Los resultados se muestran en la consola.
        """
        print("\nAnálisis de Patrones:")
        
        # Distribución de medios de acceso
        print("\nDistribución de Medios de Acceso:")
        print("\nHistórico (2019-2023):")
        hist_dist = self.historical['Medio_Acceso'].value_counts(normalize=True)
        print(hist_dist)
        
        print("\nPredicciones (2024-2028):")
        pred_dist = self.predictions['Medio_Acceso_Predicho'].value_counts(normalize=True)
        print(pred_dist)
        
        # Crecimiento anual promedio por medio de acceso
        print("\nCrecimiento Anual Promedio por Medio de Acceso:")
        for medio in self.historical['Medio_Acceso'].unique():
            hist_growth = self.calculate_growth_rate(self.historical, medio, 'Medio_Acceso')
            pred_growth = self.calculate_growth_rate(self.predictions, medio, 'Medio_Acceso_Predicho')
            print(f"\n{medio}:")
            print(f"Histórico: {hist_growth:.2%}")
            print(f"Predicho: {pred_growth:.2%}")
            
            # Calcular diferencia porcentual
            if hist_growth != 0:
                diff = (pred_growth - hist_growth) / abs(hist_growth)
                print(f"Diferencia relativa: {diff:.2%}")

    def analyze_confidence(self):
        """
        Analiza en detalle la confianza de las predicciones generadas.
        
        Notes
        -----
        Realiza y muestra:
        1. Estadísticas de confianza (media, mínimo, máximo) por medio de acceso
        2. Distribución de predicciones por nivel de confianza (Baja, Media, Alta, Muy Alta)
        3. Evolución de la confianza promedio por año de predicción
        
        Los resultados se muestran en la consola.
        """
        print("\nAnálisis de Confianza en Predicciones (2024-2028):")
        
        # Confianza promedio por medio de acceso
        print("\nConfianza por medio de acceso (media, mínimo, máximo):")
        confidence_by_means = self.predictions.groupby('Medio_Acceso_Predicho')['Confianza_Prediccion'].agg(['mean', 'min', 'max'])
        print(confidence_by_means)
        
        # Distribución de niveles de confianza
        confidence_levels = pd.cut(self.predictions['Confianza_Prediccion'], 
                                 bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                 labels=['Baja', 'Media', 'Alta', 'Muy Alta'])
        print("\nDistribución de niveles de confianza:")
        level_counts = confidence_levels.value_counts()
        level_percent = confidence_levels.value_counts(normalize=True)
        
        # Combinar conteos y porcentajes
        level_stats = pd.DataFrame({
            'Conteo': level_counts,
            'Porcentaje': level_percent.map(lambda x: f"{x:.2%}")
        })
        print(level_stats)
        
        # Confianza por año
        print("\nConfianza promedio por año:")
        conf_by_year = self.predictions.groupby('Año')['Confianza_Prediccion'].mean()
        print(conf_by_year)
        
        # Análisis de tendencia de confianza
        trend = "creciente" if conf_by_year.iloc[-1] > conf_by_year.iloc[0] else "decreciente"
        print(f"\nTendencia de confianza: {trend} a lo largo del periodo de predicción")

    def calculate_growth_rate(self, data, medio, column_name):
        """
        Calcula la tasa de crecimiento anual compuesto para un medio de acceso específico.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame con los datos a analizar.
        medio : str
            Medio de acceso para el que se calculará la tasa.
        column_name : str
            Nombre de la columna que contiene el medio de acceso.
            
        Returns
        -------
        float
            Tasa de crecimiento anual compuesto.
            
        Notes
        -----
        Utiliza la fórmula de Tasa de Crecimiento Anual Compuesto (CAGR):
        CAGR = (Valor Final / Valor Inicial)^(1/n) - 1
        donde n es el número de años menos 1.
        """
        # Filtrar datos para el medio específico y agrupar por año
        yearly_data = data[data[column_name] == medio].groupby('Año')['Num_Turistas'].sum()
        
        # Calcular tasa si hay al menos 2 años de datos
        if len(yearly_data) >= 2:
            # CAGR = (Valor Final / Valor Inicial)^(1/n) - 1
            return (yearly_data.iloc[-1] / yearly_data.iloc[0]) ** (1/(len(yearly_data)-1)) - 1
        
        # Si no hay suficientes datos, retornar 0
        return 0
        
    def generate_detailed_tables(self):
        """
        Genera tablas detalladas de análisis para uso en informes y dashboards.
        
        Notes
        -----
        Crea y guarda tres archivos CSV:
        1. 'confianza_por_medio_Año.csv': Tabla pivote de confianza promedio por
           medio de acceso y año
        2. 'turistas_por_Año_pais_medio_2019_2028.csv': Agrupación de número de
           turistas por año, mes, país y medio de acceso
        3. 'turistas_por_Año_pais_medio_destino_2019_2028.csv': Agrupación que
           incluye también el destino principal
        
        Las tablas se utilizan para análisis más detallados y visualizaciones en
        herramientas de BI.
        """
        # Tabla de confianza por medio de acceso y año
        confidence_table = self.predictions.pivot_table(
            values='Confianza_Prediccion',
            index='Medio_Acceso_Predicho',
            columns='Año',
            aggfunc='mean'
        )
        confidence_table.to_csv('confianza_por_medio_Año.csv')
        print("\nConfianza promedio por Medio de Acceso y Año:")
        print(confidence_table)
        
        # Dataset agrupado por año, mes, país y medio de acceso
        grouping1 = self.data.groupby(['Año', 'Mes', 'Pais', 'Medio_Acceso_Predicho'])['Num_Turistas'].sum().reset_index()
        grouping1.to_csv('turistas_por_Año_pais_medio_2019_2028.csv', index=False)
        print("\nPrimeras filas del agrupamiento por año, país y medio de acceso:")
        print(grouping1.head())
        
        # Dataset agrupado incluyendo destino principal
        grouping2 = self.data.groupby(['Año', 'Mes', 'Pais', 'Medio_Acceso_Predicho', 'Destino_Principal'])['Num_Turistas'].sum().reset_index()
        grouping2.to_csv('turistas_por_Año_pais_medio_destino_2019_2028.csv', index=False)
        print("\nPrimeras filas del agrupamiento incluyendo destino principal:")
        print(grouping2.head())
        
        print("\nTablas detalladas generadas y guardadas:")
        print("- confianza_por_medio_Año.csv")
        print("- turistas_por_Año_pais_medio_2019_2028.csv")
        print("- turistas_por_Año_pais_medio_destino_2019_2028.csv")


def main():
    """
    Función principal que ejecuta el análisis completo de comparación entre
    datos históricos y predicciones.
    
    Esta función inicializa el analizador de predicciones y ejecuta todas las
    funciones de análisis y visualización secuencialmente.
    """
    print("Iniciando análisis de comparación de históricos vs predicciones...")
    
    # Crear instancia del analizador
    analyzer = PredictionAnalyzer()
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones comparativas...")
    analyzer.plot_distribution_comparison()
    analyzer.plot_temporal_evolution()
    analyzer.plot_confidence_analysis()
    
    # Realizar análisis estadísticos
    print("\nRealizando análisis estadísticos...")
    analyzer.analyze_patterns()
    analyzer.analyze_confidence()
    
    # Generar tablas detalladas
    print("\nGenerando tablas detalladas para análisis adicional...")
    analyzer.generate_detailed_tables()
    
    print("\nProceso de análisis completado exitosamente.")


if __name__ == "__main__":
    main()