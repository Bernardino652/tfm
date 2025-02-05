#Implementacion modelo redes neuronales
#Importar librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
import os

class TourismANN:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.base_path = os.path.abspath('modelos')
        self.figures_path = os.path.join(self.base_path, 'figures')
        self.dashboard_path = os.path.join(self.base_path, 'dashboard')  # Nueva línea
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.dashboard_path, exist_ok=True)  # Nueva línea


    
    def prepare_data(self, data_path='data/processed/turismo_procesado_2019_2023.csv'):
        print("Cargando y preparando datos...")
        df = pd.read_csv(data_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year
        
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                    'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                    'Destino_Principal']
        target = 'Medio_Acceso'
        
        X = df[features]
        y = df[target]
        
        categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            X[feature] = self.label_encoders[feature].fit_transform(X[feature])
        
        X = self.scaler.fit_transform(X)
        
        self.label_encoders['target'] = LabelEncoder()
        y = self.label_encoders['target'].fit_transform(y)
        y = to_categorical(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        print("Construyendo el modelo ANN...")
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        return self.model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        print("Entrenando el modelo ANN...")
        history = self.model.fit(X_train, y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 validation_split=0.2,
                                 verbose=1)
        return history

    def plot_confusion_matrix_detailed(self, y_true_classes, y_pred_classes):
        """Genera matriz de confusión detallada con métricas"""
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        class_names = self.label_encoders['target'].classes_
        
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
        
        # Matriz normalizada
        ax0 = plt.subplot(gs[0])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax0,
                    xticklabels=class_names, yticklabels=class_names)
        ax0.set_title('Matriz de Confusión Normalizada')
        ax0.set_xlabel('Predicción')
        ax0.set_ylabel('Real')
        
        # Métricas por clase
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'ann_confusion_matrix_detailed.png'))
        plt.close()

    def plot_confusion_matrix_simple(self, y_true_classes, y_pred_classes):
        """Genera matriz de confusión simple"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        class_names = self.label_encoders['target'].classes_
        
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
        """Genera curvas ROC para cada clase"""
        y_pred_proba = self.model.predict(X_test)
        n_classes = y_test.shape[1]
        
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                    label=f'{self.label_encoders["target"].classes_[i]} (AUC = {roc_auc:.2f})')
        
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
        """Genera curvas precision-recall para cada clase"""
        y_pred_proba = self.model.predict(X_test)
        n_classes = y_test.shape[1]
        
        plt.figure(figsize=(10, 8))
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
        """Visualiza la curva de aprendizaje del modelo"""
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
        """Genera DataFrames para el dashboard de PowerBI"""
        
        # DataFrame de métricas de precisión por clase
        report = classification_report(y_true_classes, y_pred_classes,
                                    target_names=self.label_encoders['target'].classes_,
                                    output_dict=True)
        
        metricas_precision = pd.DataFrame({
            'Medio_Acceso': list(report.keys())[:-3],
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
        print("\nEvaluando modelo...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Precisión del modelo en datos de prueba: {accuracy*100:.2f}%')
        
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
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


def main():
    print("Iniciando proceso de Red Neuronal Artificial...")
    ann_model = TourismANN()
    
    X_train, X_test, y_train, y_test = ann_model.prepare_data()
    
    model = ann_model.build_model(input_shape=X_train.shape[1], 
                                  num_classes=y_train.shape[1])
    
    history = ann_model.train_model(X_train, y_train)
    
    accuracy = ann_model.evaluate_model(X_test, y_test, history)
    
    print("\nProceso de ANN completado exitosamente")
    print(f"Precisión final: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
