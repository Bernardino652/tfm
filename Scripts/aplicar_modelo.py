#Aplicacion de modelo para generacion de prediccion de medios de acceso
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class TourismPredictor:
    def __init__(self, model_path='random_forest_model.pkl', 
                data_path='dataset_completo_2019_2028.csv'):
       self.model = joblib.load(model_path)
       self.data = pd.read_csv(data_path)
       self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
       self.label_encoders = self._prepare_encoders()

    def _prepare_encoders(self):
       encoders = {}
       categorical_features = ['Pais', 'Estacionalidad', 'Destino_Principal', 'Medio_Acceso']
       for feature in categorical_features:
           encoders[feature] = LabelEncoder()
           encoders[feature].fit(self.data[feature].unique())
       return encoders
    def predict_access(self, data):
        X = data.copy()
    
        # Convertir Anio a Año
        X['Año'] = X['Anio']
        
        features = ['Mes', 'Año', 'Pais', 'Num_Turistas', 'Estacionalidad',
                'Temperatura_Media', 'Precipitacion_Media', 'Satisfaccion',
                'Destino_Principal']
        
        # Codificar variables categóricas
        for col in ['Pais', 'Estacionalidad', 'Destino_Principal']:
            X[col] = self.label_encoders[col].transform(X[col])
        
        # Realizar predicción
        predictions = self.model.predict(X[features])
        probabilities = self.model.predict_proba(X[features])
        
        return (self.label_encoders['Medio_Acceso'].inverse_transform(predictions), 
                np.max(probabilities, axis=1))

    def generate_predictions(self):
       # Separar datos históricos y futuros
       historical = self.data[self.data['Tipo_Dato'] == 'Histórico'].copy()
       future = self.data[self.data['Tipo_Dato'] == 'Datos Generados'].copy()
       
       # Predecir medios de acceso para datos futuros
       predicted_access, confidence = self.predict_access(future)
       
       # Agregar predicciones
       future['Medio_Acceso_Predicho'] = predicted_access
       future['Confianza_Prediccion'] = confidence
       
       # Mantener medio de acceso histórico como predicho para datos históricos
       historical['Medio_Acceso_Predicho'] = historical['Medio_Acceso']
       historical['Confianza_Prediccion'] = 1.0
       
       # Combinar datasets
       result = pd.concat([historical, future], ignore_index=True)
       result.sort_values(['Fecha', 'Pais'], inplace=True)
       
       # Guardar resultados
       result.to_csv('predicciones_finales_2019_2028.csv', index=False)
       
       # Mostrar resumen
       print("\nResumen de predicciones:")
       print("\nDatos históricos (2019-2023):")
       print(historical['Medio_Acceso'].value_counts())
       print("\nPredicciones (2024-2028):")
       print(future['Medio_Acceso_Predicho'].value_counts())
       
       return result

def main():
   predictor = TourismPredictor()
   predictions = predictor.generate_predictions()

if __name__ == "__main__":
   main()