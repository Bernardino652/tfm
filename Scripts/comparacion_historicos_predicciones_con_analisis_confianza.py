#Verificación de datos predichos  generados por el  modelo
#Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PredictionAnalyzer:
   def __init__(self, data_path='predicciones_finales_2019_2028.csv'):
       self.data = pd.read_csv(data_path)
       self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
       
       # Separar datos históricos y predicciones
       self.historical = self.data[self.data['Anio'] <= 2023]
       self.predictions = self.data[self.data['Anio'] > 2023]

   def plot_distribution_comparison(self):
       """Compara distribución de medios de acceso entre histórico y predicciones"""
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
       # Distribución histórica
       hist_dist = self.historical['Medio_Acceso'].value_counts(normalize=True)
       hist_dist.plot(kind='bar', ax=ax1, title='Distribución Histórica (2019-2023)')
       
       # Distribución predicha
       pred_dist = self.predictions['Medio_Acceso_Predicho'].value_counts(normalize=True)
       pred_dist.plot(kind='bar', ax=ax2, title='Distribución Predicha (2024-2028)')
       
       plt.tight_layout()
       plt.savefig('distribucion_comparacion.png')
       plt.close()

   def plot_temporal_evolution(self):
       """Muestra evolución temporal de medios de acceso"""
       plt.figure(figsize=(15, 8))
       
       # Datos históricos
       historical_grouped = self.historical.groupby(['Anio', 'Medio_Acceso'])['Num_Turistas'].sum().reset_index()
       predictions_grouped = self.predictions.groupby(['Anio', 'Medio_Acceso_Predicho'])['Num_Turistas'].sum().reset_index()
       
       # Graficar cada medio de acceso
       for medio in self.historical['Medio_Acceso'].unique():
           # Datos históricos
           hist_data = historical_grouped[historical_grouped['Medio_Acceso'] == medio]
           plt.plot(hist_data['Anio'], hist_data['Num_Turistas'], 
                   marker='o', label=f'Histórico - {medio}')
           
           # Predicciones
           pred_data = predictions_grouped[predictions_grouped['Medio_Acceso_Predicho'] == medio]
           plt.plot(pred_data['Anio'], pred_data['Num_Turistas'], 
                   linestyle='--', marker='x', label=f'Predicción - {medio}')
       
       plt.axvline(x=2023, color='gray', linestyle=':', label='Límite histórico/predicción')
       plt.title('Evolución de Medios de Acceso (2019-2028)')
       plt.xlabel('Año')
       plt.ylabel('Número de Turistas')
       plt.legend()
       plt.grid(True)
       plt.savefig('evolucion_temporal.png')
       plt.close()

   def plot_confidence_analysis(self):
       """Visualiza el análisis de confianza de las predicciones"""
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
       # Boxplot de confianza por medio de acceso
       sns.boxplot(data=self.predictions, x='Medio_Acceso_Predicho', 
                  y='Confianza_Prediccion', ax=ax1)
       ax1.set_title('Distribución de Confianza por Medio de Acceso')
       ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
       
       # Distribución de niveles de confianza
       conf_levels = pd.cut(self.predictions['Confianza_Prediccion'], 
                          bins=[0, 0.25, 0.5, 0.75, 1.0], 
                          labels=['Baja', 'Media', 'Alta', 'Muy Alta'])
       conf_levels.value_counts().plot(kind='bar', ax=ax2)
       ax2.set_title('Distribución de Niveles de Confianza')
       
       plt.tight_layout()
       plt.savefig('analisis_confianza.png')
       plt.close()

   def analyze_patterns(self):
       """Analiza patrones y genera métricas comparativas"""
       print("\nAnálisis de Patrones:")
       
       # Distribución de medios de acceso
       print("\nDistribución de Medios de Acceso:")
       print("\nHistórico (2019-2023):")
       print(self.historical['Medio_Acceso'].value_counts(normalize=True))
       print("\nPredicciones (2024-2028):")
       print(self.predictions['Medio_Acceso_Predicho'].value_counts(normalize=True))
       
       # Crecimiento anual promedio
       print("\nCrecimiento Anual Promedio por Medio de Acceso:")
       for medio in self.historical['Medio_Acceso'].unique():
           hist_growth = self.calculate_growth_rate(self.historical, medio, 'Medio_Acceso')
           pred_growth = self.calculate_growth_rate(self.predictions, medio, 'Medio_Acceso_Predicho')
           print(f"\n{medio}:")
           print(f"Histórico: {hist_growth:.2%}")
           print(f"Predicho: {pred_growth:.2%}")

   def analyze_confidence(self):
       """Analiza la confianza de las predicciones"""
       print("\nAnálisis de Confianza en Predicciones (2024-2028):")
       
       # Confianza promedio por medio de acceso
       print("\nConfianza promedio por medio de acceso:")
       confidence_by_means = self.predictions.groupby('Medio_Acceso_Predicho')['Confianza_Prediccion'].agg(['mean', 'min', 'max'])
       print(confidence_by_means)
       
       # Distribución de niveles de confianza
       confidence_levels = pd.cut(self.predictions['Confianza_Prediccion'], 
                                bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                labels=['Baja', 'Media', 'Alta', 'Muy Alta'])
       print("\nDistribución de niveles de confianza:")
       print(confidence_levels.value_counts())
       
       # Confianza por año
       print("\nConfianza promedio por año:")
       print(self.predictions.groupby('Anio')['Confianza_Prediccion'].mean())

   def calculate_growth_rate(self, data, medio, column_name):
       """Calcula tasa de crecimiento anual"""
       yearly_data = data[data[column_name] == medio].groupby('Anio')['Num_Turistas'].sum()
       if len(yearly_data) >= 2:
           return (yearly_data.iloc[-1] / yearly_data.iloc[0]) ** (1/(len(yearly_data)-1)) - 1
       return 0
   def generate_detailed_tables(self):
    """Genera tablas detalladas de análisis"""
   
    # Tabla de confianza por medio de acceso y año
    confidence_table = self.predictions.pivot_table(
        values='Confianza_Prediccion',
        index='Medio_Acceso_Predicho',
        columns='Anio',
        aggfunc='mean'
    )
    confidence_table.to_csv('confianza_por_medio_anio.csv')
    print("\nConfianza promedio por Medio de Acceso y Año:")
    print(confidence_table)

    # Dataset agrupado por año, país y medio de acceso
    grouping1 = self.data.groupby(['Anio','Mes', 'Pais', 'Medio_Acceso_Predicho'])['Num_Turistas'].sum().reset_index()
    grouping1.to_csv('turistas_por_anio_pais_medio_2019_2028.csv', index=False)
    print("\nPrimeras filas del agrupamiento por año, país y medio de acceso:")
    print(grouping1.head())

    # Dataset agrupado incluyendo destino principal
    grouping2 = self.data.groupby(['Anio','Mes', 'Pais', 'Medio_Acceso_Predicho', 'Destino_Principal'])['Num_Turistas'].sum().reset_index()
    grouping2.to_csv('turistas_por_anio_pais_medio_destino_2019_2028.csv', index=False)
    print("\nPrimeras filas del agrupamiento incluyendo destino principal:")
    print(grouping2.head())



def main():
   analyzer = PredictionAnalyzer()
   analyzer.plot_distribution_comparison()
   analyzer.plot_temporal_evolution()
   analyzer.plot_confidence_analysis()
   analyzer.analyze_patterns()
   analyzer.analyze_confidence()
   analyzer.generate_detailed_tables()

if __name__ == "__main__":
   main()