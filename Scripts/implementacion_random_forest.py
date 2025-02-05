#Implementacion modelo random forest
#Importacion de librerias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
plt.switch_backend('Agg')
import os

class TourismRandomForest:
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        self.feature_importance = None
        self.base_path = os.path.abspath('modelos')
        self.figures_path = os.path.join(self.base_path, 'figures')
        self.models_path = os.path.join(self.base_path, 'trained_models')
        self.dashboard_path = os.path.join(self.base_path, 'dashboard')  # Nueva línea
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.dashboard_path, exist_ok=True)  # Nueva línea

    

    def prepare_data(self, data_path='data/processed/turismo_procesado_2019_2023.csv'):
        print("Cargando datos desde:", data_path)
        df = pd.read_csv(data_path)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        df['Mes'] = df['Fecha'].dt.month
        df['Año'] = df['Fecha'].dt.year
        
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                   'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                   'Destino_Principal']
        target = 'Medio_Acceso'
        
        print("\nCodificando variables categóricas...")
        categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        X = df[features]
        y = df[target]
        
        self.label_encoders['target'] = LabelEncoder()
        y = self.label_encoders['target'].fit_transform(y)
        
        print("Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, features

    def train_model(self, X_train, y_train):
        print("\nIniciando entrenamiento del modelo Random Forest...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        print("Realizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        
        self.model = grid_search.best_estimator_
        
        model_path = os.path.join(self.models_path, 'random_forest_model.pkl')
        import joblib
        joblib.dump(self.model, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        return grid_search.best_estimator_

    def plot_confusion_matrix_simple(self, y_test, y_pred):
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
        cm = confusion_matrix(y_test, y_pred)
        class_names = self.label_encoders['target'].classes_
        
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
        
        ax0 = plt.subplot(gs[0])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax0,
                    xticklabels=class_names, yticklabels=class_names)
        ax0.set_title('Matriz de Confusión Normalizada')
        ax0.set_xlabel('Predicción')
        ax0.set_ylabel('Real')
        
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
        plt.savefig(os.path.join(self.figures_path, 'rf_confusion_matrix_detailed.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_roc_curves(self, X_test, y_test):
        plt.figure(figsize=(10, 8))
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.label_encoders['target'].classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.label_encoders["target"].classes_[i]} (AUC = {roc_auc:.2f})')
        
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
        plt.figure(figsize=(10, 8))
        y_score = self.model.predict_proba(X_test)
        n_classes = len(self.label_encoders['target'].classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
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
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
               
        for i, v in enumerate(importance_df['Importance']):
            ax.text(v, i, f'{v:.3f}', va='center')
        
        plt.title('Análisis Detallado de Importancia de Características')
        plt.xlabel('Puntuación de Importancia')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'rf_feature_importance_detailed.png'), bbox_inches='tight')
        plt.close('all')
        

    def generate_dashboard_data(self, X_test, y_test, y_pred, features):
        """Genera DataFrames para el dashboard de PowerBI"""
    
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
    
        # Generar datos para dashboard
        dashboard_data = self.generate_dashboard_data(X_test, y_test, y_pred, features)  # Asegurándonos de pasar features
        print("\nDatos para dashboard generados en:", self.dashboard_path)
        print("Archivos generados:")
        for filename in os.listdir(self.dashboard_path):
              if filename.startswith('rf_'):
               print(f"- {filename}")
    
        # Guardar métricas
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
    print("Iniciando proceso de Random Forest...")
    rf_model = TourismRandomForest()
    X_train, X_test, y_train, y_test, features = rf_model.prepare_data()  # Asegurarnos de capturar features
    rf_model.train_model(X_train, y_train)
    feature_importance = rf_model.evaluate_model(X_test, y_test, features)  # Pasar features
    print("\nProceso completado exitosamente")

if __name__ == "__main__":
    main()
    


