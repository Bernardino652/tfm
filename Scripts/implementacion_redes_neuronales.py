#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: implementacion_redes_neuronales.py
Implementación de Redes Neuronales para Predicción de Medios de Acceso Turístico.

Este script implementa un modelo de redes neuronales artificiales (ANN) para predecir
el medio de acceso utilizado por los turistas (aeropuerto, carretera, tren, puerto) 
en función de diversas características como el país de origen, mes, destino, condiciones
climáticas y nivel de satisfacción. El proceso completo incluye preparación de datos, 
construcción de la arquitectura de la red, entrenamiento con validación cruzada, evaluación
exhaustiva mediante diversas métricas y visualizaciones, y exportación de resultados
para análisis posterior.

El modelo utiliza una arquitectura feed-forward con capas densas, regularización mediante
dropout, y optimización con Adam. Se entrena con datos turísticos procesados del período
2019-2023 y genera diversos artefactos que permiten comprender su rendimiento y evaluar
su capacidad predictiva.

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca para manipulación y análisis de datos estructurados.
  Utilizada para cargar el dataset, manipular estructuras de datos y exportar resultados
  en formato CSV para dashboards.

- numpy (1.22.0+): Biblioteca para computación numérica.
  Utilizada para operaciones matriciales, transformación de datos y cálculo eficiente
  de métricas de rendimiento.

- scikit-learn (1.0.0+): Herramientas para aprendizaje automático.
  Componentes utilizados:
  - model_selection: Para división de datos en conjuntos de entrenamiento y prueba.
  - preprocessing: Para codificación de variables categóricas y estandarización de datos.
  - metrics: Para generación de métricas de evaluación y curvas de rendimiento.

- tensorflow (2.8.0+): Framework para aprendizaje profundo.
  Utilizado para construir, entrenar y evaluar la red neuronal artificial.
  Componentes clave:
  - keras.models: Definición de la arquitectura secuencial del modelo.
  - keras.layers: Componentes de la red (Dense, Dropout).
  - keras.optimizers: Algoritmo de optimización Adam.
  - keras.utils: Conversión de etiquetas a formato categórico.

- matplotlib (3.5.0+): Biblioteca de visualización.
  Utilizada para generar todas las visualizaciones de rendimiento y evaluación.
  Configurada con backend 'Agg' para funcionamiento sin interfaz gráfica.

- seaborn (0.11.0+): Biblioteca para visualización estadística basada en matplotlib.
  Utilizada para crear mapas de calor y mejorar la estética de las visualizaciones.

- os: Módulo estándar para interacción con el sistema operativo.
  Utilizado para gestión de directorios y archivos de salida.

- time: Módulo estándar para medición de tiempos.
  Utilizado para registrar el rendimiento temporal de las diferentes fases del proceso.

Autor: Bernardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.2
"""

# Importación de librerías
import pandas as pd                       # Para manipulación y análisis de datos
import numpy as np                        # Para operaciones matriciales y numéricas
from sklearn.model_selection import train_test_split  # Para división de conjuntos de datos
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Para preprocesamiento
from tensorflow.keras.models import Sequential  # Para definir arquitectura secuencial
from tensorflow.keras.layers import Dense, Dropout  # Capas de la red neuronal
from tensorflow.keras.optimizers import Adam  # Optimizador adaptativo
from tensorflow.keras.utils import to_categorical  # Para codificación one-hot
from sklearn.metrics import (                # Métricas de evaluación
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
import matplotlib.pyplot as plt           # Para visualizaciones
import seaborn as sns                     # Para visualizaciones estadísticas mejoradas
import matplotlib.gridspec as gridspec    # Para layouts complejos en visualizaciones
import matplotlib                         # Para configuración general de visualización
matplotlib.use('Agg')                     # Configurar backend no interactivo
import os                                 # Para manejo de directorios y archivos
import time                               # Para medición de tiempos de ejecución


class TourismANN:
    """
    Clase para implementar, entrenar y evaluar un modelo de Red Neuronal Artificial
    para la predicción del medio de acceso turístico.
    
    Esta clase encapsula todo el flujo de trabajo relacionado con el modelo:
    preparación de datos, construcción de la arquitectura, entrenamiento,
    evaluación, visualización y exportación de resultados.
    
    Attributes
    ----------
    model : tensorflow.keras.models.Sequential
        Modelo de red neuronal entrenado.
    label_encoders : dict
        Diccionario que almacena los codificadores para variables categóricas.
    scaler : sklearn.preprocessing.StandardScaler
        Escalador para normalizar las características numéricas.
    base_path : str
        Ruta base para almacenamiento de archivos generados.
    figures_path : str
        Ruta para almacenar visualizaciones generadas.
    dashboard_path : str
        Ruta para almacenar datos procesados para dashboards.
    """
    
    def __init__(self):
        """
        Inicializa la clase TourismANN y crea la estructura de directorios
        necesaria para almacenar modelos, figuras y datos de dashboard.
        """
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.base_path = os.path.abspath('modelos')
        self.figures_path = os.path.join(self.base_path, 'figures')
        self.dashboard_path = os.path.join(self.base_path, 'dashboard')
        
        # Crear directorios necesarios
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.dashboard_path, exist_ok=True)
    
    def prepare_data(self, data_path='data/processed/turismo_procesado_2019_2023.csv'):
        """
        Carga, preprocesa y divide los datos para el entrenamiento del modelo.
        
        Parameters
        ----------
        data_path : str, optional
            Ruta al archivo CSV con los datos turísticos procesados,
            por defecto 'data/processed/turismo_procesado_2019_2023.csv'.
            
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test: Conjuntos de datos divididos
            y preprocesados para entrenamiento y evaluación.
            
        Notes
        -----
        Realiza los siguientes procesos:
        - Carga de datos y transformación de fechas
        - Extracción de características temporales (mes, año)
        - Codificación de variables categóricas
        - Estandarización de características numéricas
        - Codificación one-hot de la variable objetivo
        - División en conjuntos de entrenamiento (80%) y prueba (20%)
        """
        print("Cargando y preparando datos...")
        df = pd.read_csv(data_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Extraer características temporales
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year
        
        # Seleccionar características y variable objetivo
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                    'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                    'Destino_Principal']
        target = 'Medio_Acceso'
        
        # Separar características y variable objetivo
        X = df[features]
        y = df[target]
        
        # Codificar variables categóricas
        categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            X[feature] = self.label_encoders[feature].fit_transform(X[feature])
        
        # Estandarizar características numéricas
        X = self.scaler.fit_transform(X)
        
        # Codificar variable objetivo
        self.label_encoders['target'] = LabelEncoder()
        y = self.label_encoders['target'].fit_transform(y)
        y = to_categorical(y)  # Convertir a formato one-hot
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """
        Construye la arquitectura de la red neuronal.
        
        Parameters
        ----------
        input_shape : int
            Número de características de entrada.
        num_classes : int
            Número de clases a predecir (categorías de salida).
            
        Returns
        -------
        tensorflow.keras.models.Sequential
            Modelo de red neuronal compilado pero no entrenado.
            
        Notes
        -----
        Arquitectura de la red:
        - Capa de entrada: 256 neuronas, activación ReLU
        - Dropout del 50% para regularización
        - Capa oculta: 128 neuronas, activación ReLU
        - Dropout del 40% para regularización
        - Capa oculta: 64 neuronas, activación ReLU
        - Capa de salida: num_classes neuronas, activación softmax
        
        Compilación:
        - Optimizador: Adam con tasa de aprendizaje de 0.001
        - Función de pérdida: Entropía cruzada categórica
        - Métrica: Precisión (accuracy)
        """
        print("Construyendo el modelo ANN...")
        
        # Definir arquitectura secuencial
        self.model = Sequential([
            # Capa de entrada
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),  # Regularización para prevenir sobreajuste
            
            # Primera capa oculta
            Dense(128, activation='relu'),
            Dropout(0.4),  # Regularización para prevenir sobreajuste
            
            # Segunda capa oculta
            Dense(64, activation='relu'),
            
            # Capa de salida (softmax para clasificación multiclase)
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar el modelo
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        """
        Entrena el modelo de red neuronal con el conjunto de datos proporcionado.
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Conjunto de características para entrenamiento.
        y_train : numpy.ndarray
            Etiquetas codificadas en one-hot para entrenamiento.
        epochs : int, optional
            Número de épocas de entrenamiento, por defecto 100.
        batch_size : int, optional
            Tamaño del lote para el entrenamiento, por defecto 32.
            
        Returns
        -------
        tensorflow.keras.callbacks.History
            Historial de entrenamiento con métricas por época.
            
        Notes
        -----
        - Utiliza el 20% de los datos de entrenamiento como conjunto de validación
        - Registra métricas de precisión y pérdida tanto en entrenamiento como validación
        - Estos registros se utilizan posteriormente para visualizar las curvas de aprendizaje
        """
        print("Entrenando el modelo ANN...")
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,  # 20% de los datos de entrenamiento para validación
            verbose=1  # Mostrar progreso
        )
        
        return history

    def plot_confusion_matrix_detailed(self, y_true_classes, y_pred_classes):
        """
        Genera una visualización detallada de la matriz de confusión con métricas.
        
        Parameters
        ----------
        y_true_classes : numpy.ndarray
            Clases reales (no en formato one-hot).
        y_pred_classes : numpy.ndarray
            Clases predichas por el modelo (no en formato one-hot).
            
        Notes
        -----
        Genera un gráfico con dos paneles:
        1. Matriz de confusión normalizada por filas (valores porcentuales)
        2. Métricas de rendimiento por clase (precisión, recall, F1-score)
        
        El gráfico se guarda en el directorio de figuras definido en la inicialización.
        """
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        class_names = self.label_encoders['target'].classes_
        
        # Crear figura con dos paneles
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
        
        # Panel 1: Matriz normalizada
        ax0 = plt.subplot(gs[0])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax0,
                    xticklabels=class_names, yticklabels=class_names)
        ax0.set_title('Matriz de Confusión Normalizada')
        ax0.set_xlabel('Predicción')
        ax0.set_ylabel('Real')
        
        # Panel 2: Métricas por clase
        ax1 = plt.subplot(gs[1])
        
        # Calcular métricas
        metrics = {
            'Precisión': np.diag(cm) / np.sum(cm, axis=0),
            'Recall': np.diag(cm) / np.sum(cm, axis=1),
            'F1-Score': 2 * (np.diag(cm) / np.sum(cm, axis=0) * np.diag(cm) / np.sum(cm, axis=1)) / 
                        (np.diag(cm) / np.sum(cm, axis=0) + np.diag(cm) / np.sum(cm, axis=1))
        }
        metrics_df = pd.DataFrame(metrics, index=class_names)
        
        # Visualizar métricas
        sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Métricas de Rendimiento por Clase')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'ann_confusion_matrix_detailed.png'))
        plt.close()

    def plot_confusion_matrix_simple(self, y_true_classes, y_pred_classes):
        """
        Genera una visualización simple de la matriz de confusión.
        
        Parameters
        ----------
        y_true_classes : numpy.ndarray
            Clases reales (no en formato one-hot).
        y_pred_classes : numpy.ndarray
            Clases predichas por el modelo (no en formato one-hot).
            
        Notes
        -----
        Genera un mapa de calor con los valores absolutos de la matriz de confusión.
        Útil para una visualización rápida de los aciertos y errores del modelo.
        
        El gráfico se guarda en el directorio de figuras definido en la inicialización.
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        class_names = self.label_encoders['target'].classes_
        
        # Visualizar matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        
        plt.title('Matriz de Confusión - ANN')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_path, 'ann_confusion_matrix_simple.png'))
        plt.close()

    def plot_roc_curves(self, X_test, y_test):
        """
        Genera curvas ROC (Receiver Operating Characteristic) para cada clase.
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Conjunto de características de prueba.
        y_test : numpy.ndarray
            Etiquetas codificadas en one-hot para prueba.
            
        Notes
        -----
        Para cada clase, calcula:
        - Tasa de verdaderos positivos (sensibilidad)
        - Tasa de falsos positivos (1 - especificidad)
        - Área bajo la curva ROC (AUC)
        
        El gráfico se guarda en el directorio de figuras definido en la inicialización.
        """
        # Obtener probabilidades de predicción
        y_pred_proba = self.model.predict(X_test)
        n_classes = y_test.shape[1]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Generar curva ROC para cada clase
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                    label=f'{self.label_encoders["target"].classes_[i]} (AUC = {roc_auc:.2f})')
        
        # Añadir línea diagonal de referencia (clasificador aleatorio)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC por Clase - ANN')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.savefig(os.path.join(self.figures_path, 'ann_roc_curves.png'))
        plt.close()

    def plot_precision_recall_curves(self, X_test, y_test):
        """
        Genera curvas de precisión-recall para cada clase.
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Conjunto de características de prueba.
        y_test : numpy.ndarray
            Etiquetas codificadas en one-hot para prueba.
            
        Notes
        -----
        Para cada clase, calcula:
        - Precisión (valor predictivo positivo)
        - Recall (sensibilidad)
        
        Útil para evaluar el rendimiento en conjuntos de datos desbalanceados.
        El gráfico se guarda en el directorio de figuras definido en la inicialización.
        """
        # Obtener probabilidades de predicción
        y_pred_proba = self.model.predict(X_test)
        n_classes = y_test.shape[1]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Generar curva precision-recall para cada clase
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, 
                    label=f'{self.label_encoders["target"].classes_[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall por Clase - ANN')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.savefig(os.path.join(self.figures_path, 'ann_pr_curves.png'))
        plt.close()

    def plot_learning_curves(self, history):
        """
        Visualiza las curvas de aprendizaje del modelo durante el entrenamiento.
        
        Parameters
        ----------
        history : tensorflow.keras.callbacks.History
            Historial de entrenamiento con métricas por época.
            
        Notes
        -----
        Genera un gráfico que muestra la evolución de la precisión en los conjuntos
        de entrenamiento y validación a lo largo de las épocas.
        
        Útil para detectar problemas de sobreajuste o subajuste.
        El gráfico se guarda en el directorio de figuras definido en la inicialización.
        """
        plt.figure(figsize=(12, 6))
        
        # Graficar precisión de entrenamiento y validación
        plt.plot(history.history['accuracy'], 'b-', label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], 'orange', label='Validación')
        
        plt.title('Curva de Aprendizaje del Modelo')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend(loc='upper right')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'ann_learning_curve.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def generate_dashboard_data(self, X_test, y_test, y_pred_classes, y_true_classes, history):
        """
        Genera DataFrames con métricas y resultados para su uso en dashboards.
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Conjunto de características de prueba.
        y_test : numpy.ndarray
            Etiquetas codificadas en one-hot para prueba.
        y_pred_classes : numpy.ndarray
            Clases predichas por el modelo (no en formato one-hot).
        y_true_classes : numpy.ndarray
            Clases reales (no en formato one-hot).
        history : tensorflow.keras.callbacks.History
            Historial de entrenamiento con métricas por época.
            
        Returns
        -------
        dict
            Diccionario con DataFrames para diferentes aspectos del rendimiento del modelo.
            
        Notes
        -----
        Genera y guarda seis archivos CSV diferentes:
        1. Métricas de precisión por clase
        2. Matriz de confusión
        3. Probabilidades de predicción por clase
        4. Curvas de aprendizaje (precisión y pérdida por época)
        5. Métricas ROC-AUC por clase
        6. Métricas globales del modelo
        
        Estos archivos están diseñados para ser importados en herramientas
        de visualización como Power BI.
        """
        # Obtener las métricas del reporte de clasificación
        report = classification_report(y_true_classes, y_pred_classes,
                                    target_names=self.label_encoders['target'].classes_,
                                    output_dict=True)
        
        # DataFrame de métricas de precisión por clase
        metricas_precision = pd.DataFrame({
            'Medio_Acceso': list(report.keys())[:-3],  # Excluir promedios
            'Precision': [report[key]['precision'] for key in list(report.keys())[:-3]],
            'Recall': [report[key]['recall'] for key in list(report.keys())[:-3]],
            'F1_Score': [report[key]['f1-score'] for key in list(report.keys())[:-3]],
            'Support': [report[key]['support'] for key in list(report.keys())[:-3]]
        })

        # DataFrame de matriz de confusión
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        matriz_confusion = pd.DataFrame(
            cm,
            index=self.label_encoders['target'].classes_,
            columns=self.label_encoders['target'].classes_
        )

        # DataFrame de predicciones por clase
        y_pred_proba = self.model.predict(X_test)
        predicciones_proba = pd.DataFrame(
            y_pred_proba,
            columns=[f'Prob_{clase}' for clase in self.label_encoders['target'].classes_]
        )

        # DataFrame de curvas de aprendizaje
        curvas_aprendizaje = pd.DataFrame({
            'Epoca': range(1, len(history.history['accuracy']) + 1),
            'Accuracy_Train': history.history['accuracy'],
            'Accuracy_Val': history.history['val_accuracy'],
            'Loss_Train': history.history['loss'],
            'Loss_Val': history.history['val_loss']
        })

        # DataFrame de métricas ROC-AUC
        metricas_roc = pd.DataFrame({
            'Medio_Acceso': self.label_encoders['target'].classes_,
            'AUC_Score': [auc(*roc_curve(y_test[:, i], y_pred_proba[:, i])[:2]) 
                        for i in range(len(self.label_encoders['target'].classes_))]
        })

        # Métricas globales del modelo
        metricas_globales = pd.DataFrame({
            'Metrica': ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1'],
            'Valor': [
                report['accuracy'],
                report['macro avg']['precision'],
                report['macro avg']['recall'],
                report['macro avg']['f1-score']
            ]
        })

        # Guardar todos los DataFrames
        metricas_precision.to_csv(os.path.join(self.dashboard_path, 'ann_metricas_precision.csv'), index=False)
        matriz_confusion.to_csv(os.path.join(self.dashboard_path, 'ann_matriz_confusion.csv'))
        predicciones_proba.to_csv(os.path.join(self.dashboard_path, 'ann_predicciones_probabilidad.csv'), index=False)
        curvas_aprendizaje.to_csv(os.path.join(self.dashboard_path, 'ann_curvas_aprendizaje.csv'), index=False)
        metricas_roc.to_csv(os.path.join(self.dashboard_path, 'ann_metricas_roc.csv'), index=False)
        metricas_globales.to_csv(os.path.join(self.dashboard_path, 'ann_metricas_globales.csv'), index=False)

        return {
            'metricas_precision': metricas_precision,
            'matriz_confusion': matriz_confusion,
            'predicciones_proba': predicciones_proba,
            'curvas_aprendizaje': curvas_aprendizaje,
            'metricas_roc': metricas_roc,
            'metricas_globales': metricas_globales
        }
    
    def evaluate_model(self, X_test, y_test, history):
        """
        Realiza una evaluación completa del modelo entrenado.
        
        Parameters
        ----------
        X_test : numpy.ndarray
            Conjunto de características de prueba.
        y_test : numpy.ndarray
            Etiquetas codificadas en one-hot para prueba.
        history : tensorflow.keras.callbacks.History
            Historial de entrenamiento con métricas por época.
            
        Returns
        -------
        float
            Precisión global del modelo en el conjunto de prueba.
            
        Notes
        -----
        - Calcula métricas de rendimiento (precisión, recall, F1)
        - Genera visualizaciones (matrices de confusión, curvas ROC, etc.)
        - Genera datos para dashboards
        - Guarda resultados en archivos de texto y CSV
        
        Este método centraliza todas las evaluaciones del modelo entrenado.
        """
        print("\nEvaluando modelo...")
        
        # Evaluar modelo y obtener métricas globales
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Precisión del modelo en datos de prueba: {accuracy*100:.2f}%')
        
        # Obtener predicciones y convertir a clases
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Mostrar reporte de clasificación
        print("\nReporte de Clasificación:")
        print(classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.label_encoders['target'].classes_
        ))
        
        # Generar todas las visualizaciones
        self.plot_confusion_matrix_simple(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix_detailed(y_true_classes, y_pred_classes)
        self.plot_roc_curves(X_test, y_test)
        self.plot_precision_recall_curves(X_test, y_test)
        self.plot_learning_curves(history)
        
        # Generar datos para dashboard
        dashboard_data = self.generate_dashboard_data(X_test, y_test, y_pred_classes, y_true_classes, history)
        print("\nDatos para dashboard generados en:", self.dashboard_path)
        print("Archivos generados:")
        for filename in os.listdir(self.dashboard_path):
            if filename.startswith('ann_'):
                print(f"- {filename}")
        
        # Guardar métricas detalladas
        metrics_path = os.path.join(self.base_path, 'ann_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Red Neuronal Artificial - Métricas de Evaluación\n")
            f.write("===========================================\n\n")
            f.write(f"Precisión global: {accuracy*100:.2f}%\n\n")
            f.write("Métricas detalladas por clase:\n")
            f.write(classification_report(
                y_true_classes,
                y_pred_classes,
                target_names=self.label_encoders['target'].classes_
            ))
        
        print(f"\nMétricas guardadas en: {metrics_path}")
        print("Gráficos guardados en:", self.figures_path)

        return accuracy

    def save_trained_model(self, model_path='modelos/trained_models/ann_model.h5'):
        """
        Guarda el modelo entrenado en formato .h5
        
        Args:
            model_path (str): Ruta donde se guardará el modelo
        """
        print("\nGuardando modelo entrenado...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Modelo guardado en: {model_path}")

def main():
    import time
    
    # Tiempo inicial total
    tiempo_inicio_total = time.time()
    
    print("Iniciando proceso de Red Neuronal Artificial...")
    ann_model = TourismANN()
    
    # Tiempo de preparación de datos
    print("\nIniciando preparación de datos...")
    tiempo_inicio = time.time()
    X_train, X_test, y_train, y_test = ann_model.prepare_data()
    tiempo_prep_datos = time.time() - tiempo_inicio
    print(f"✓ Preparación de datos completada en {tiempo_prep_datos:.2f} segundos")
    
    # Tiempo de construcción del modelo
    print("\nIniciando construcción del modelo...")
    tiempo_inicio = time.time()
    model = ann_model.build_model(input_shape=X_train.shape[1], 
                                num_classes=y_train.shape[1])
    tiempo_construccion = time.time() - tiempo_inicio
    print(f"✓ Construcción del modelo completada en {tiempo_construccion:.2f} segundos")
    
    # Tiempo de entrenamiento
    print("\nIniciando entrenamiento del modelo...")
    tiempo_inicio = time.time()
    history = ann_model.train_model(X_train, y_train)
    tiempo_entrenamiento = time.time() - tiempo_inicio
    print(f"✓ Entrenamiento completado en {tiempo_entrenamiento:.2f} segundos")
    
    # Tiempo de evaluación
    print("\nIniciando evaluación del modelo...")
    tiempo_inicio = time.time()
    accuracy = ann_model.evaluate_model(X_test, y_test, history)
    tiempo_evaluacion = time.time() - tiempo_inicio
    print(f"✓ Evaluación completada en {tiempo_evaluacion:.2f} segundos")
    
    # Tiempo de guardado
    print("\nGuardando modelo...")
    tiempo_inicio = time.time()
    ann_model.save_trained_model()
    tiempo_guardado = time.time() - tiempo_inicio
    print(f"✓ Modelo guardado en {tiempo_guardado:.2f} segundos")
    
    # Resumen de tiempos
    tiempo_total = time.time() - tiempo_inicio_total
    print("\n=== Resumen de Tiempos de Procesamiento ===")
    print(f"Preparación de datos: {tiempo_prep_datos:.2f} s")
    print(f"Construcción del modelo: {tiempo_construccion:.2f} s")
    print(f"Entrenamiento: {tiempo_entrenamiento:.2f} s")
    print(f"Evaluación: {tiempo_evaluacion:.2f} s")
    print(f"Guardado: {tiempo_guardado:.2f} s")
    print(f"Tiempo total: {tiempo_total:.2f} s ({tiempo_total/60:.2f} min)")
    
    print("\nProceso de ANN completado exitosamente")
    print(f"Precisión final: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()