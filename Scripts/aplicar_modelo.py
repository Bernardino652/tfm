#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: aplicar_modelo.py
Aplicación de Modelo para Predicción de Medios de Acceso Turístico.

Este script aplica un modelo de aprendizaje automático previamente entrenado para
predecir los medios de acceso (aeropuerto, carretera, tren, puerto) que utilizarán 
los turistas en futuras visitas a España, basándose en características como el país 
de origen, mes, año, número de turistas, condiciones climáticas y destino principal.

El proceso completo incluye:
- Carga del modelo entrenado (principalmente Random Forest)
- Carga del dataset completo (histórico + proyecciones futuras)
- Preparación de los codificadores para variables categóricas
- Aplicación del modelo para generar predicciones sobre datos futuros
- Combinación de datos históricos (reales) y proyecciones (predichas)
- Generación de archivo CSV con el conjunto completo de datos, incluyendo
  medio de acceso real (para datos históricos) y predicho (para datos futuros)

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos tabulares.
  Utilizada para cargar datos, procesar estructuras tabulares, manipular series 
  temporales y exportar resultados finales en formato CSV.
  Funciones clave: read_csv, to_datetime, concat.

- numpy (1.22.0+): Biblioteca para operaciones numéricas y manipulación de arrays.
  Utilizada principalmente para encontrar valores máximos en las matrices de probabilidad
  de predicción generadas por el modelo.

- sklearn.preprocessing.LabelEncoder: Componente de scikit-learn para codificación
  de variables categóricas. Utilizado para transformar variables categóricas (como país,
  estacionalidad, destino) en valores numéricos que el modelo pueda procesar, y
  posteriormente para convertir las predicciones numéricas de vuelta a categorías.

- joblib: Biblioteca para serialización y carga de objetos Python.
  Utilizada específicamente para cargar el modelo entrenado previamente guardado
  en formato .pkl, manteniendo toda su configuración y parámetros.

Autor: Analista de Datos Turísticos
Fecha: 25/02/2025
Versión: 1.0
"""

# Importación de librerías
import pandas as pd               # Para manipulación y análisis de datos
import numpy as np                # Para operaciones numéricas
from sklearn.preprocessing import LabelEncoder  # Para codificación de variables categóricas
import joblib                     # Para cargar el modelo entrenado


class TourismPredictor:
    """
    Clase para aplicar un modelo entrenado a datos turísticos y generar predicciones
    sobre medios de acceso.
    
    Esta clase encapsula toda la lógica necesaria para cargar un modelo entrenado,
    preparar los datos de entrada, generar predicciones y combinar los resultados
    con datos históricos para crear un conjunto completo de datos con predicciones.
    
    Attributes
    ----------
    model : sklearn estimator
        Modelo de aprendizaje automático cargado desde el archivo .pkl.
    data : pandas.DataFrame
        Dataset completo con datos históricos y proyecciones futuras.
    label_encoders : dict
        Diccionario con codificadores para variables categóricas.
    """
    
    def __init__(self, model_path='random_forest_model.pkl', 
                data_path='dataset_completo_2019_2028.csv'):
        """
        Inicializa el predictor cargando el modelo entrenado y el dataset completo.
        
        Parameters
        ----------
        model_path : str, optional
            Ruta al archivo del modelo entrenado (.pkl),
            por defecto 'random_forest_model.pkl'.
        data_path : str, optional
            Ruta al archivo CSV con datos históricos y futuros,
            por defecto 'dataset_completo_2019_2028.csv'.
            
        Notes
        -----
        Durante la inicialización, la clase:
        1. Carga el modelo entrenado
        2. Carga el dataset completo
        3. Convierte la columna 'Fecha' a formato datetime
        4. Prepara los codificadores para variables categóricas
        """
        # Cargar modelo entrenado
        self.model = joblib.load(model_path)
        
        # Cargar datos
        self.data = pd.read_csv(data_path)
        self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
        
        # Preparar codificadores
        self.label_encoders = self._prepare_encoders()

    def _prepare_encoders(self):
        """
        Prepara los codificadores para variables categóricas.
        
        Returns
        -------
        dict
            Diccionario con codificadores LabelEncoder para cada variable categórica.
            
        Notes
        -----
        Los codificadores se entrenan con los valores únicos de cada variable
        categórica en el dataset completo, para asegurar consistencia en la
        codificación de datos históricos y futuros.
        """
        encoders = {}
        categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal', 'Medio_Acceso']
        
        for feature in categorical_features:
            encoders[feature] = LabelEncoder()
            encoders[feature].fit(self.data[feature].unique())
            
        return encoders
        
    def predict_access(self, data):
        """
        Aplica el modelo para predecir el medio de acceso más probable.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame con los datos para los que se generarán predicciones.
            
        Returns
        -------
        tuple
            Tupla que contiene:
            - Array con las categorías predichas (medios de acceso)
            - Array con las probabilidades/confianza de cada predicción
            
        Notes
        -----
        Este método:
        1. Prepara las características de entrada codificando variables categóricas
        2. Aplica el modelo para obtener predicciones y probabilidades
        3. Convierte las predicciones numéricas a categorías
        4. Retorna las predicciones con sus niveles de confianza asociados
        """
        # Crear copia para no modificar los datos originales
        X = data.copy()
        
        # Asegurar tipo correcto para el año
        X['Año'] = X['Año']
        
        # Definir características para el modelo
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                'Destino_Principal']
        
        # Codificar variables categóricas
        for col in ['Pais', 'Estacionalidad', 'Destino_Principal']:
            X[col] = self.label_encoders[col].transform(X[col])
        
        # Aplicar modelo para obtener predicciones
        predictions = self.model.predict(X[features])
        
        # Obtener probabilidades/confianza de las predicciones
        probabilities = self.model.predict_proba(X[features])
        
        # Retornar categorías predichas y nivel de confianza
        return (self.label_encoders['Medio_Acceso'].inverse_transform(predictions), 
                np.max(probabilities, axis=1))

    def generate_predictions(self):
        """
        Genera predicciones para datos futuros y combina con datos históricos.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame completo con datos históricos y futuros, incluyendo
            medios de acceso reales (históricos) y predichos (futuros),
            junto con niveles de confianza de predicción.
            
        Notes
        -----
        Este método:
        1. Separa datos históricos y proyecciones futuras
        2. Aplica el modelo para predecir medios de acceso en datos futuros
        3. Para datos históricos, utiliza el medio de acceso real como "predicho"
        4. Combina ambos conjuntos de datos y los ordena cronológicamente
        5. Guarda los resultados en un nuevo archivo CSV
        6. Muestra un resumen de la distribución de medios de acceso
        """
        # Separar datos históricos y futuros
        historical = self.data[self.data['Tipo_Dato'] == 'Histórico'].copy()
        future = self.data[self.data['Tipo_Dato'] == 'Datos Generados'].copy()
        
        # Predecir medios de acceso para datos futuros
        predicted_access, confidence = self.predict_access(future)
        
        # Agregar predicciones a datos futuros
        future['Medio_Acceso_Predicho'] = predicted_access
        future['Confianza_Prediccion'] = confidence
        
        # Para datos históricos, usar el medio de acceso real como predicho
        historical['Medio_Acceso_Predicho'] = historical['Medio_Acceso']
        historical['Confianza_Prediccion'] = 1.0  # Confianza máxima para datos reales
        
        # Combinar datasets y ordenar cronológicamente
        result = pd.concat([historical, future], ignore_index=True)
        result.sort_values(['Fecha', 'Pais'], inplace=True)
        
        # Guardar resultados en CSV
        output_file = 'predicciones_finales_2019_2028.csv'
        result.to_csv(output_file, index=False)
        
        # Mostrar resumen comparativo
        print(f"\nPredicciones generadas y guardadas en: {output_file}")
        print("\nResumen de predicciones:")
        print("\nDatos históricos (2019-2023):")
        print(historical['Medio_Acceso'].value_counts())
        print("\nPredicciones (2024-2028):")
        print(future['Medio_Acceso_Predicho'].value_counts())
        
        return result


def main():
    """
    Función principal que ejecuta el proceso de predicción.
    
    Esta función inicializa el predictor turístico y genera predicciones
    para todo el conjunto de datos, produciendo el archivo final con
    proyecciones de medios de acceso.
    """
    # Crear instancia del predictor
    predictor = TourismPredictor()
    
    # Generar predicciones
    predictions = predictor.generate_predictions()
    
    print("\nProceso de predicción completado exitosamente.")
    print(f"Registros totales procesados: {len(predictions)}")


if __name__ == "__main__":
    main()