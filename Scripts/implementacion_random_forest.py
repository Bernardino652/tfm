#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: implementacion_random_forest.py
Implementación de Modelo Random Forest para Predicción de Medios de Acceso Turístico.

Este script implementa un modelo de aprendizaje automático basado en Random Forest
para predecir el medio de acceso utilizado por los turistas (aeropuerto, carretera, etc.)
en función de diversas características como el país de origen, mes, destino y variables
climáticas. El proceso completo incluye preparación de datos, entrenamiento del modelo
con optimización de hiperparámetros, evaluación exhaustiva y generación de visualizaciones
y datos para su uso en dashboards.

El modelo se entrena con datos turísticos procesados del período 2019-2023 y genera
diversos artefactos de evaluación que permiten comprender su rendimiento y las
características más influyentes en la predicción.

Dependencias:
-----------
- pandas (1.5.0+): Biblioteca principal para manipulación de datos tabulares.
  Utilizada para cargar y procesar los datos turísticos, crear dataframes para 
  evaluación y exportar resultados.

- numpy (1.22.0+): Biblioteca para cálculos numéricos y operaciones matriciales.
  Utilizada para manipulaciones de arrays en el procesamiento de métricas y 
  visualizaciones.

- scikit-learn (1.0.0+): Conjunto de herramientas para aprendizaje automático.
  Componentes utilizados:
  - model_selection: Para división de datos (train_test_split) y búsqueda de 
    hiperparámetros (GridSearchCV).
  - ensemble: Implementación del clasificador RandomForestClassifier.
  - preprocessing: Para codificación de variables categóricas (LabelEncoder) y 
    binarización de etiquetas.
  - metrics: Funciones para evaluación del modelo (classification_report, 
    confusion_matrix, roc_curve, auc, precision_recall_curve).

- matplotlib (3.5.0+): Biblioteca de visualización.
  Utilizada para generar todas las gráficas de evaluación y rendimiento del modelo.
  Se configura para usar el backend 'Agg' para garantizar la generación de gráficos
  en entornos sin interfaz gráfica.

- seaborn (0.11.0+): Biblioteca para visualización estadística basada en matplotlib.
  Utilizada para crear mapas de calor, gráficos de barras y otras visualizaciones
  con mayor nivel estético y funcional.

- joblib: Utilizada para serializar y guardar el modelo entrenado.

- os: Módulo estándar para interactuar con el sistema operativo.
  Utilizado para gestionar directorios y rutas de archivos.

Autor: Bernmardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.2
"""

# Importación de librerías
import pandas as pd               # Para manipulación y análisis de datos
import numpy as np                # Para operaciones numéricas y matriciales
from sklearn.model_selection import train_test_split, GridSearchCV  # Para división de datos y optimización de hiperparámetros
from sklearn.ensemble import RandomForestClassifier     # Algoritmo de clasificación Random Forest
from sklearn.preprocessing import LabelEncoder           # Para codificar variables categóricas
from sklearn.metrics import (                            # Métricas de evaluación
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize        # Para binarizar etiquetas en evaluación multi-clase
import seaborn as sns                                   # Para visualizaciones estadísticas avanzadas
import matplotlib.pyplot as plt                         # Para creación de gráficos
import matplotlib.gridspec as gridspec                  # Para layouts complejos de gráficos
import matplotlib                                       # Para configuración general de visualización
matplotlib.use('Agg')                                   # Configurar backend no interactivo
plt.switch_backend('Agg')                               # Asegurar backend no interactivo
import os                                               # Para manipulación de directorios y archivos
import joblib                                           # Para serializar modelos (implícitamente importado)


class TourismRandomForest:
    """
    Clase que implementa un modelo de Random Forest para la predicción 
    de medios de acceso turístico.
    
    Esta clase encapsula todo el flujo de trabajo relacionado con el modelo:
    preparación de datos, entrenamiento, evaluación, generación de visualizaciones
    y exportación de resultados para su uso en dashboards.
    
    Attributes
    ----------
    label_encoders : dict
        Diccionario que almacena los codificadores para variables categóricas.
    model : RandomForestClassifier
        Modelo entrenado de Random Forest.
    feature_importance : array
        Importancia de cada característica en el modelo entrenado.
    base_path : str
        Ruta base para almacenamiento de archivos generados.
    figures_path : str
        Ruta para almacenar visualizaciones generadas.
    models_path : str
        Ruta para almacenar el modelo entrenado.
    dashboard_path : str
        Ruta para almacenar datos procesados para dashboards.
    """
    
    def __init__(self):
        """
        Inicializa la clase TourismRandomForest y crea la estructura de directorios
        necesaria para almacenar modelos, figuras y datos de dashboard.
        """
        self.label_encoders = {}
        self.model = None
        self.feature_importance = None
        self.base_path = os.path.abspath('modelos')
        self.figures_path = os.path.join(self.base_path, 'figures')
        self.models_path = os.path.join(self.base_path, 'trained_models')
        self.dashboard_path = os.path.join(self.base_path, 'dashboard')
        
        # Crear directorios si no existen
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.dashboard_path, exist_ok=True)

    def prepare_data(self, data_path='data/processed/turismo_procesado_2019_2023.csv'):
        """
        Carga y prepara los datos para el entrenamiento del modelo.
        
        Parameters
        ----------
        data_path : str, optional
            Ruta al archivo CSV con los datos turísticos procesados,
            por defecto 'data/processed/turismo_procesado_2019_2023.csv'.
            
        Returns
        -------
        tuple
            Tupla con X_train, X_test, y_train, y_test y la lista de características.
            
        Notes
        -----
        - Extrae mes y año de la columna Fecha
        - Codifica variables categóricas usando LabelEncoder
        - Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%)
        """
        print("Cargando datos desde:", data_path)
        df = pd.read_csv(data_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        # Extraer características temporales
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year
        
        # Definir características y variable objetivo
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                   'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                   'Destino_Principal']
        target = 'Medio_Acceso'
        
        # Codificar variables categóricas
        print("\nCodificando variables categóricas...")
        categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        # Separar características y variable objetivo
        X = df[features]
        y = df[target]
        
        # Codificar variable objetivo
        self.label_encoders['target'] = LabelEncoder()
        y = self.label_encoders['target'].fit_transform(y)
        
        # Dividir datos en entrenamiento y prueba
        print("Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, features

    def train_model(self, X_train, y_train):
        """
        Entrena el modelo Random Forest con optimización de hiperparámetros.
        
        Parameters
        ----------
        X_train : pandas.DataFrame
            Conjunto de características para entrenamiento.
        y_train : array-like
            Vector de etiquetas para entrenamiento.
            
        Returns
        -------
        RandomForestClassifier
            Modelo entrenado con los mejores hiperparámetros encontrados.
            
        Notes
        -----
        - Utiliza GridSearchCV para encontrar los mejores hiperparámetros
        - Guarda el modelo entrenado en formato .pkl para su uso posterior
        - Explora combinaciones de n_estimators, max_depth, min_samples_split
          y min_samples_leaf
        """
        print("\nIniciando entrenamiento del modelo Random Forest...")
        # Definir la cuadrícula de hiperparámetros a explorar
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Inicializar clasificador base
        rf = RandomForestClassifier(random_state=42)
        
        # Realizar búsqueda de hiperparámetros óptimos
        print("Realizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,                  # Validación cruzada con 5 pliegues
            n_jobs=-1,             # Usar todos los núcleos disponibles
            scoring='accuracy'     # Optimizar por precisión
        )
        
        # Ajustar modelo
        grid_search.fit(X_train, y_train)
        
        # Mostrar mejores parámetros
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        
        # Guardar mejor modelo
        self.model = grid_search.best_estimator_
        
        # Serializar modelo para uso posterior
        model_path = os.path.join(self.models_path, 'random_forest_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        return grid_search.best_estimator_

    def plot_confusion_matrix_simple(self, y_test, y_pred):
        """
        Genera una visualización simple de la matriz de confusión.
        
        Parameters
        ----------
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
        y_pred : array-like
            Predicciones del modelo.
            
        Notes
        -----
        - Crea un mapa de calor con los valores absolutos de la matriz de confusión
        - Utiliza la paleta YlOrRd para visualizar intensidades
        - Guarda la figura en formato PNG de alta resolución
        """
        cm = confusion_matrix(y_test, y_pred)
        class_names = self.label_encoders['target'].classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=class_names,
                    yticklabels=class_names)
        
        plt.title('Matriz de Confusión - Random Forest')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.figures_path, 'rf_confusion_matrix_simple.png'), 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close('all')

    def plot_confusion_matrix_detailed(self, y_test, y_pred):
        """
        Genera una visualización detallada de la matriz de confusión 
        con métricas adicionales.
        
        Parameters
        ----------
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
        y_pred : array-like
            Predicciones del modelo.
            
        Notes
        -----
        - Crea un panel con dos visualizaciones:
          1. Matriz de confusión normalizada (valores porcentuales)
          2. Métricas de precisión, recall y F1-score para cada clase
        - Utiliza diferentes paletas de colores para cada visualización
        - Guarda la figura en formato PNG de alta resolución
        """
        cm = confusion_matrix(y_test, y_pred)
        class_names = self.label_encoders['target'].classes_
        
        # Crear figura con dos subplots
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
        
        # Subplot 1: Matriz de confusión normalizada
        ax0 = plt.subplot(gs[0])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax0,
                    xticklabels=class_names, yticklabels=class_names)
        ax0.set_title('Matriz de Confusión Normalizada')
        ax0.set_xlabel('Predicción')
        ax0.set_ylabel('Real')
        
        # Subplot 2: Métricas por clase
        ax1 = plt.subplot(gs[1])
        metrics = {
            'Precisión': np.diag(cm) / np.sum(cm, axis=0),
            'Recall': np.diag(cm) / np.sum(cm, axis=1),
            'F1-Score': 2 * (np.diag(cm) / np.sum(cm, axis=0) * np.diag(cm) / np.sum(cm, axis=1)) / 
                        (np.diag(cm) / np.sum(cm, axis=0) + np.diag(cm) / np.sum(cm, axis=1))
        }
        metrics_df = pd.DataFrame(metrics, index=class_names)
        sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Métricas de Rendimiento por Clase')
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'rf_confusion_matrix_detailed.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_roc_curves(self, X_test, y_test):
        """
        Genera curvas ROC (Receiver Operating Characteristic) para cada clase.
        
        Parameters
        ----------
        X_test : pandas.DataFrame
            Conjunto de características de prueba.
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
            
        Notes
        -----
        - Genera curvas ROC separadas para cada clase en un problema multiclase
        - Calcula y muestra el área bajo la curva (AUC) para cada clase
        - Incluye la línea de referencia para clasificación aleatoria
        - Guarda la figura en formato PNG
        """
        plt.figure(figsize=(10, 8))
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.label_encoders['target'].classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Generar curva ROC para cada clase
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.label_encoders["target"].classes_[i]} (AUC = {roc_auc:.2f})')
        
        # Añadir línea de referencia (clasificador aleatorio)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC por Clase - Random Forest')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(self.figures_path, 'rf_roc_curves.png'), bbox_inches='tight')
        plt.close('all')

    def plot_precision_recall_curves(self, X_test, y_test):
        """
        Genera curvas de precisión-recall para cada clase.
        
        Parameters
        ----------
        X_test : pandas.DataFrame
            Conjunto de características de prueba.
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
            
        Notes
        -----
        - Genera curvas de precisión-recall separadas para cada clase
        - Útil para evaluar rendimiento en clases desbalanceadas
        - Guarda la figura en formato PNG
        """
        plt.figure(figsize=(10, 8))
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.label_encoders['target'].classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Generar curva precision-recall para cada clase
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, label=f'{self.label_encoders["target"].classes_[i]}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall por Clase - Random Forest')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(os.path.join(self.figures_path, 'rf_pr_curves.png'), bbox_inches='tight')
        plt.close('all')

    def plot_feature_importance_detailed(self, features):
        """
        Genera un gráfico detallado de la importancia de las características.
        
        Parameters
        ----------
        features : list
            Lista con los nombres de las características utilizadas en el modelo.
            
        Notes
        -----
        - Visualiza la importancia relativa de cada característica en el modelo
        - Ordena las características por importancia ascendente
        - Incluye los valores numéricos de importancia en cada barra
        - Guarda la figura en formato PNG
        """
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
               
        # Añadir etiquetas con valores exactos
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v, i, f'{v:.3f}', va='center')
        
        plt.title('Análisis Detallado de Importancia de Características')
        plt.xlabel('Puntuación de Importancia')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'rf_feature_importance_detailed.png'), bbox_inches='tight')
        plt.close('all')
        
    def plot_correlation_heatmap(self, X):
        """
        Genera un mapa de calor de correlaciones entre las variables predictoras.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Conjunto de características.
            
        Notes
        -----
        - Calcula y visualiza la matriz de correlación entre todas las variables
        - Utiliza una máscara triangular para mostrar solo la mitad inferior
        - Usa paleta divergente 'coolwarm' centrada en cero
        - Guarda la figura en formato PNG
        """
        correlation_matrix = X.corr()
        
        plt.figure(figsize=(12, 10))  
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   square=True,
                   linewidths=.5)
        
        plt.title('Mapa de Calor de Correlaciones entre Variables')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'correlation_heatmap.png'))
        plt.close()

    def generate_dashboard_data(self, X_test, y_test, y_pred, features):
        """
        Genera DataFrames con métricas y resultados para su uso en dashboards.
        
        Parameters
        ----------
        X_test : pandas.DataFrame
            Conjunto de características de prueba.
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
        y_pred : array-like
            Predicciones del modelo.
        features : list
            Lista con los nombres de las características.
            
        Returns
        -------
        dict
            Diccionario con DataFrames de métricas de precisión, importancia de
            características, matriz de confusión y probabilidades de predicción.
            
        Notes
        -----
        - Genera y guarda cuatro archivos CSV diferentes:
          1. Métricas de precisión por clase
          2. Importancia de características
          3. Matriz de confusión
          4. Probabilidades de predicción por clase
        - Estos archivos están diseñados para ser importados en herramientas
          de visualización como Power BI
        """
        # Obtener las métricas del reporte de clasificación
        report = classification_report(y_test, y_pred, 
                                 target_names=self.label_encoders['target'].classes_,
                                 output_dict=True)
    
        # DataFrame de métricas de precisión
        metricas_precision_rf = pd.DataFrame({
          'Medio_Acceso': list(report.keys())[:-3],  # Excluir promedios
          'Precision': [report[key]['precision'] for key in list(report.keys())[:-3]],
          'Recall': [report[key]['recall'] for key in list(report.keys())[:-3]],
          'F1_Score': [report[key]['f1-score'] for key in list(report.keys())[:-3]],
          'Support': [report[key]['support'] for key in list(report.keys())[:-3]]
        })

        # DataFrame de importancia de características
        importancia_caracteristicas = pd.DataFrame({
          'Caracteristica': features,
          'Importancia': self.model.feature_importances_
        }).sort_values('Importancia', ascending=False)

        # DataFrame de matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        matriz_confusion = pd.DataFrame(
           cm,
           index=self.label_encoders['target'].classes_,
           columns=self.label_encoders['target'].classes_
        )

        # DataFrame de predicciones por clase
        y_pred_proba = self.model.predict_proba(X_test)
        predicciones_proba = pd.DataFrame(
        y_pred_proba,
        columns=[f'Prob_{clase}' for clase in self.label_encoders['target'].classes_]
        )

        # Guardar DataFrames
        os.makedirs('data/dashboard', exist_ok=True)
        metricas_precision_rf.to_csv(os.path.join(self.dashboard_path, 'rf_metricas_precision.csv'), index=False)
        importancia_caracteristicas.to_csv(os.path.join(self.dashboard_path, 'rf_importancia_caracteristicas.csv'), index=False)
        matriz_confusion.to_csv(os.path.join(self.dashboard_path, 'rf_matriz_confusion.csv'))
        predicciones_proba.to_csv(os.path.join(self.dashboard_path, 'rf_predicciones_probabilidad.csv'), index=False)

        return {
          'metricas_precision': metricas_precision_rf,
          'importancia_caracteristicas': importancia_caracteristicas,
          'matriz_confusion': matriz_confusion,
          'predicciones_proba': predicciones_proba
        }    

    def evaluate_model(self, X_test, y_test, features):
        """
        Realiza una evaluación completa del modelo entrenado.
        
        Parameters
        ----------
        X_test : pandas.DataFrame
            Conjunto de características de prueba.
        y_test : array-like
            Etiquetas reales del conjunto de prueba.
        features : list
            Lista con los nombres de las características.
            
        Returns
        -------
        array
            Importancia de las características en el modelo.
            
        Notes
        -----
        - Genera el reporte de clasificación con métricas por clase
        - Crea todas las visualizaciones (matrices de confusión, curvas ROC, etc.)
        - Genera los datos para dashboard
        - Guarda todas las métricas en un archivo de texto
        """
        print("\nEvaluando modelo...")
        y_pred = self.model.predict(X_test)
    
        print("\nReporte de Clasificación:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoders['target'].classes_
        ))
    
        # Generar visualizaciones
        self.plot_confusion_matrix_simple(y_test, y_pred)
        self.plot_confusion_matrix_detailed(y_test, y_pred)
        self.plot_feature_importance_detailed(features)
        self.plot_roc_curves(X_test, y_test)
        self.plot_precision_recall_curves(X_test, y_test)
        self.plot_correlation_heatmap(X_test)
    
        # Generar datos para dashboard
        dashboard_data = self.generate_dashboard_data(X_test, y_test, y_pred, features)
        print("\nDatos para dashboard generados en:", self.dashboard_path)
        print("Archivos generados:")
        for filename in os.listdir(self.dashboard_path):
              if filename.startswith('rf_'):
               print(f"- {filename}")
    
        # Guardar métricas en archivo de texto
        metrics_path = os.path.join(self.base_path, 'rf_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Random Forest - Métricas de Evaluación\n")
            f.write("=====================================\n\n")
            f.write(classification_report(
                y_test,
                y_pred,
             target_names=self.label_encoders['target'].classes_
            ))
        print(f"Métricas guardadas en: {metrics_path}")
        print("Gráficos guardados en:", self.figures_path)
    
        return self.feature_importance


def main():
    import time
    
    # Tiempo inicial total
    tiempo_inicio_total = time.time()
    """
    Función principal que ejecuta el flujo completo de entrenamiento y evaluación
    del modelo Random Forest para predicción de medios de acceso turístico.
    
    Esta función coordina todas las etapas: preparación de datos, entrenamiento
    del modelo y evaluación completa.
    """


    print("Iniciando proceso de Random Forest...")
    
    # Inicializar clase
    # Tiempo Inicializar clase
    print("\nIniciando inicializar clase...")
    tiempo_inicio = time.time()
    rf_model = TourismRandomForest()
    tiempo_inicio_clase = time.time() - tiempo_inicio
    print(f"✓ Inicializar clase completada {tiempo_inicio_clase:.2f} segundos")

    # Preparar datos
    # Tiempo preparar datos
    print("\nIniciando preparar datos...")
    tiempo_inicio = time.time()
    X_train, X_test, y_train, y_test, features = rf_model.prepare_data()
    
    tiempo_prep_datos = time.time() - tiempo_inicio
    print(f"✓ Preparar datos completada {tiempo_prep_datos:.2f} segundos")


    # Entrenar modelo

     # Tiempo entrenar modelo
    print("\nIniciando entrenar modelo...")
    tiempo_inicio = time.time()
    rf_model.train_model(X_train, y_train)
    
    tiempo_ent_modelo = time.time() - tiempo_inicio
    print(f"✓ Entrenar modelo completada {tiempo_ent_modelo:.2f} segundos")

    # Evaluar modelo
    # Tiempo evaluar modelo
    print("\nIniciando evalura modelo...")
    tiempo_inicio = time.time()

    feature_importance = rf_model.evaluate_model(X_test, y_test, features)
    
    tiempo_eva_modelo = time.time() - tiempo_inicio
    print(f"✓ Evaluar modelo completada {tiempo_eva_modelo:.2f} segundos")

# Resumen de tiempos
    tiempo_total = time.time() - tiempo_inicio_total
    print("\n=== Resumen de Tiempos de Procesamiento ===")
    print(f"Inicializar clase: {tiempo_inicio_clase:.2f} s")
    print(f"Preparar datos: {tiempo_prep_datos:.2f} s")
    print(f"Entrenar modelo: {tiempo_ent_modelo:.2f} s")
    print(f"Evaluar modelo: {tiempo_eva_modelo:.2f} s")
    print(f"Tiempo total: {tiempo_total:.2f} s ({tiempo_total/60:.2f} min)")

    print("\nProceso completado exitosamente")


if __name__ == "__main__":
    main()