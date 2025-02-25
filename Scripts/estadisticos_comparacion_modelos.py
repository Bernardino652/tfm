"""
script: estadisticos_comparacion_modelos.py
Script para realizar análisis estadístico comparativo entre modelos Random Forest y ANN.
Incluye:

Prueba de McNemar para comparar rendimiento de clasificadores
Intervalos de confianza para métricas de evaluación
Visualizaciones comparativas de resultados

Autor: Bernardino Chancusig
Fecha: 23/02/2025
"""
"""
Script para realizar análisis estadístico comparativo entre modelos Random Forest y ANN.
Incluye:
- Prueba de McNemar para comparar rendimiento de clasificadores
- Intervalos de confianza para métricas de evaluación
- Visualizaciones comparativas de resultados

Autor: Bernardino Chancusig
Fecha: 23/02/2025

Bibliotecas estándar
import os
import time
Bibliotecas de análisis de datos
import numpy as np  # Para operaciones numéricas eficientes
import pandas as pd  # Para manipulación y análisis de datos tabulares
Bibliotecas de visualización
import matplotlib.pyplot as plt  # Para creación de gráficos
import seaborn as sns  # Para visualizaciones estadísticas mejoradas
Bibliotecas estadísticas
from scipy import stats  # Para funciones estadísticas
from statsmodels.stats.contingency_tables import mcnemar  # Para prueba de McNemar
Bibliotecas de machine learning
from sklearn.model_selection import train_test_split  # Para división de datos
from sklearn.preprocessing import LabelEncoder  # Para codificación de variables categóricas
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # Métricas de evaluación
import joblib  # Para cargar/guardar modelos sklearn
from tensorflow.keras.models import load_model  # Para cargar modelos de redes neuronales

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from tensorflow.keras.models import load_model
import os
import time

class ModelComparison:
    """
    Clase para realizar comparaciones estadísticas entre modelos de machine learning.
    Implementa pruebas estadísticas e intervalos de confianza para validar
    diferencias significativas en el rendimiento.
    """
    def __init__(self):
        """
        Inicializa el comparador de modelos con las variables necesarias
        """
        self.label_encoders = {}
        self.rf_model = None
        self.ann_model = None
        self.X_test = None
        self.y_test = None
        
        # Crear directorio para guardar resultados
        self.results_dir = 'resultados_comparacion'
        os.makedirs(self.results_dir, exist_ok=True)

    def prepare_data(self, data_path='data/processed/turismo_procesado_2019_2023.csv'):
        """
        Prepara los datos para la comparación, incluyendo codificación y división
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos
            
        Returns:
            tuple: X_test, y_test preparados para la evaluación
        """
        print("\nCargando y preparando datos...")
        start_time = time.time()
        
        try:
            # Cargar datos
            df = pd.read_csv(data_path)
            print(f"✓ Datos cargados: {df.shape[0]} registros con {df.shape[1]} columnas")
            
            # Preparar características
            features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                       'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                       'Destino_Principal']
            target = 'Medio_Acceso'
            
            # Codificar variables categóricas
            categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal']
            for feature in categorical_features:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
            
            # Preparar X e y
            X = df[features]
            y = df[target]
            
            # Codificar variable objetivo
            self.label_encoders['target'] = LabelEncoder()
            y = self.label_encoders['target'].fit_transform(y)
            
            # División de datos
            _, self.X_test, _, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"✓ Datos preparados en {time.time() - start_time:.2f} segundos")
            return self.X_test, self.y_test
            
        except Exception as e:
            print(f"\nError en la preparación de datos: {e}")
            raise

    def load_models(self, rf_path='modelos/trained_models/random_forest_model.pkl', 
                   ann_path='modelos/trained_models/ann_model.h5'):
        """
        Carga los modelos previamente entrenados
        
        Args:
            rf_path (str): Ruta al modelo Random Forest guardado
            ann_path (str): Ruta al modelo ANN guardado
        """
        print("\nCargando modelos...")
        try:
            # Cargar Random Forest
            print(f"Intentando cargar Random Forest desde: {rf_path}")
            self.rf_model = joblib.load(rf_path)
            print("✓ Modelo Random Forest cargado exitosamente")
            
            # Cargar ANN
            print(f"\nIntentando cargar ANN desde: {ann_path}")
            self.ann_model = load_model(ann_path)
            print("✓ Modelo ANN cargado exitosamente")
            
        except FileNotFoundError as e:
            print(f"\nError: No se pudo encontrar el archivo del modelo: {e}")
            raise
        except Exception as e:
            print(f"\nError al cargar los modelos: {e}")
            raise

    def mcnemar_test(self, y_true, rf_pred, ann_pred):
        """
        Realiza la prueba de McNemar para comparar los modelos
        
        Args:
            y_true: Etiquetas verdaderas
            rf_pred: Predicciones de Random Forest
            ann_pred: Predicciones de ANN
            
        Returns:
            tuple: (estadístico de prueba, valor p)
        """
        # Crear tabla de contingencia para errores
        rf_incorrect = rf_pred != y_true
        ann_incorrect = ann_pred != y_true
        
        # Tabla de contingencia
        table = np.zeros((2, 2))
        table[0, 0] = np.sum((~rf_incorrect) & (~ann_incorrect))  # Ambos correctos
        table[0, 1] = np.sum((~rf_incorrect) & ann_incorrect)     # RF correcto, ANN incorrecto
        table[1, 0] = np.sum(rf_incorrect & (~ann_incorrect))     # RF incorrecto, ANN correcto
        table[1, 1] = np.sum(rf_incorrect & ann_incorrect)        # Ambos incorrectos
        
        # Realizar prueba de McNemar
        result = mcnemar(table, exact=True)
        return result.statistic, result.pvalue

    def confidence_intervals(self, y_true, y_pred, confidence=0.95):
        """
        Calcula intervalos de confianza para métricas de evaluación
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            confidence: Nivel de confianza (default: 0.95)
            
        Returns:
            dict: Intervalos de confianza para cada métrica
        """
        # Calcular métricas base
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Tamaño de muestra
        n = len(y_true)
        
        # Calcular error estándar e intervalos de confianza
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt((accuracy * (1 - accuracy)) / n)
        
        intervals = {
            'accuracy': (accuracy - z * se, accuracy + z * se),
            'precision': (precision - z * se, precision + z * se),
            'recall': (recall - z * se, recall + z * se),
            'f1': (f1 - z * se, f1 + z * se)
        }
        
        return intervals

    def plot_confidence_intervals(self, rf_intervals, ann_intervals):
        """
        Visualiza los intervalos de confianza para ambos modelos
        
        Args:
            rf_intervals: Intervalos de confianza para Random Forest
            ann_intervals: Intervalos de confianza para ANN
        """
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.35
        
        # Graficar intervalos para RF
        plt.barh(x - width/2, 
                [rf_intervals[m][0] for m in metrics],
                width,
                label='Random Forest',
                alpha=0.8)
        
        # Graficar intervalos para ANN
        plt.barh(x + width/2,
                [ann_intervals[m][0] for m in metrics],
                width,
                label='ANN',
                alpha=0.8)
        
        # Personalizar gráfico
        plt.yticks(x, metrics)
        plt.xlabel('Valor de la métrica')
        plt.title('Intervalos de Confianza por Modelo y Métrica')
        plt.legend()
        
        # Guardar gráfico
        save_path = os.path.join(self.results_dir, 'confidence_intervals.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Gráfico guardado en: {save_path}")

    def save_results(self, results_dict):
        """
        Guarda los resultados del análisis en un archivo de texto
        
        Args:
            results_dict (dict): Diccionario con los resultados a guardar
        """
        save_path = os.path.join(self.results_dir, 'resultados_analisis.txt')
        
        with open(save_path, 'w') as f:
            f.write("Resultados del Análisis Comparativo\n")
            f.write("==================================\n\n")
            
            # Escribir resultados de McNemar
            f.write("Prueba de McNemar:\n")
            f.write(f"Estadístico: {results_dict['mcnemar_stat']:.4f}\n")
            f.write(f"Valor p: {results_dict['mcnemar_p']:.4f}\n\n")
            
            # Escribir intervalos de confianza
            f.write("Intervalos de Confianza (95%):\n")
            f.write("\nRandom Forest:\n")
            for metric, interval in results_dict['rf_intervals'].items():
                f.write(f"{metric}: ({interval[0]:.4f}, {interval[1]:.4f})\n")
            
            f.write("\nRed Neuronal:\n")
            for metric, interval in results_dict['ann_intervals'].items():
                f.write(f"{metric}: ({interval[0]:.4f}, {interval[1]:.4f})\n")
        
        print(f"✓ Resultados guardados en: {save_path}")

    def run_comparison(self):
        """
        Ejecuta la comparación completa entre modelos
        """
        print("\nIniciando comparación de modelos...")
        start_time = time.time()
        
        try:
            # Obtener predicciones
            rf_pred = self.rf_model.predict(self.X_test)
            ann_pred = np.argmax(self.ann_model.predict(self.X_test), axis=1)
            
            # Realizar prueba de McNemar
            mcnemar_stat, mcnemar_p = self.mcnemar_test(self.y_test, rf_pred, ann_pred)
            
            # Calcular intervalos de confianza
            rf_intervals = self.confidence_intervals(self.y_test, rf_pred)
            ann_intervals = self.confidence_intervals(self.y_test, ann_pred)
            
            # Visualizar resultados
            self.plot_confidence_intervals(rf_intervals, ann_intervals)
            
            # Guardar resultados
            results = {
                'mcnemar_stat': mcnemar_stat,
                'mcnemar_p': mcnemar_p,
                'rf_intervals': rf_intervals,
                'ann_intervals': ann_intervals
            }
            self.save_results(results)
            
            # Imprimir resultados
            print("\nResultados de la Prueba de McNemar:")
            print(f"Estadístico: {mcnemar_stat:.4f}")
            print(f"Valor p: {mcnemar_p:.4f}")
            
            print("\nIntervalos de Confianza (95%):")
            print("\nRandom Forest:")
            for metric, interval in rf_intervals.items():
                print(f"{metric}: ({interval[0]:.4f}, {interval[1]:.4f})")
            
            print("\nRed Neuronal:")
            for metric, interval in ann_intervals.items():
                print(f"{metric}: ({interval[0]:.4f}, {interval[1]:.4f})")
            
            print(f"\n✓ Comparación completada en {time.time() - start_time:.2f} segundos")
            
        except Exception as e:
            print(f"\nError durante la comparación: {e}")
            raise

def main():
    """
    Función principal que ejecuta el análisis comparativo
    """
    print("Iniciando análisis comparativo de modelos...")
    start_time = time.time()
    
    try:
        # Crear instancia del comparador
        comparator = ModelComparison()
        
        # Preparar datos
        comparator.prepare_data()
        
        # Cargar modelos
        comparator.load_models()
        
        # Ejecutar comparación
        comparator.run_comparison()
        
        print(f"\nAnálisis comparativo completado en {time.time() - start_time:.2f} segundos")
        
    except Exception as e:
        print(f"\nError durante el análisis comparativo: {e}")
        raise

if __name__ == "__main__":
    main()