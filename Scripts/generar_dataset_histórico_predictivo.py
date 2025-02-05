#Generacion de dataset para probar modelo generado
#Importacion de librerias
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class TourismDataGenerator:
   def __init__(self, historical_data_path='turismo_procesado_2019_2023.csv'):
       self.historical_data = pd.read_csv(historical_data_path)
       self.historical_data['Fecha'] = pd.to_datetime(self.historical_data['Fecha'])
       
       # Calcular tasas de crecimiento y patrones por país
       self.growth_patterns = self._calculate_growth_patterns()
       self.seasonal_patterns = self._calculate_seasonal_patterns()
       self.climate_trends = self._calculate_climate_trends()

   def _calculate_growth_patterns(self):
       patterns = {}
       for country in self.historical_data['Pais'].unique():
           country_data = self.historical_data[self.historical_data['Pais'] == country]
           yearly_tourists = country_data.groupby('Anio')['Num_Turistas'].sum()
           
           if len(yearly_tourists) > 1:
               growth_rate = (yearly_tourists.iloc[-1] / yearly_tourists.iloc[0]) ** (1/(len(yearly_tourists)-1))
               # Ajustar crecimiento basado en distancia y conectividad
               if country in ['Francia', 'Portugal', 'Reino Unido']:
                   growth_rate *= 1.1  # Mayor crecimiento para países cercanos
               elif country in ['China', 'Japón', 'Corea del Sur']:
                   growth_rate *= 0.95  # Menor crecimiento para países lejanos
           else:
               growth_rate = 1.05
               
           patterns[country] = growth_rate
       return patterns

   def _calculate_seasonal_patterns(self):
       patterns = {}
       for country in self.historical_data['Pais'].unique():
           country_data = self.historical_data[self.historical_data['Pais'] == country]
           monthly_avg = country_data.groupby('Mes')['Num_Turistas'].mean()
           yearly_avg = monthly_avg.mean()
           patterns[country] = monthly_avg / yearly_avg
       return patterns

   def _calculate_climate_trends(self):
       # Tendencias climáticas por temporada
       seasons = {
           'verano': [6,7,8],
           'invierno': [12,1,2],
           'primavera': [3,4,5],
           'otoño': [9,10,11]
       }
       
       trends = {}
       for season, months in seasons.items():
           season_data = self.historical_data[self.historical_data['Mes'].isin(months)]
           trends[season] = {
               'temp_trend': 0.02,  # Incremento anual de temperatura
               'precip_adjustment': {
                   'verano': 0.95,  # Menos precipitación en verano
                   'invierno': 1.05,  # Más precipitación en invierno
                   'primavera': 1.0,
                   'otoño': 1.0
               }[season]
           }
       return trends

   def generate_future_data(self, start_year=2024, end_year=2028):
       future_records = []
       
       for country in self.historical_data['Pais'].unique():
           growth_rate = self.growth_patterns[country]
           seasonal_pattern = self.seasonal_patterns[country]
           
           for year in range(start_year, end_year + 1):
               for month in range(1, 13):
                   # Base: último registro histórico para este país/mes
                   base_data = self.historical_data[
                       (self.historical_data['Pais'] == country) &
                       (self.historical_data['Mes'] == month)
                   ].copy()
                   
                   if len(base_data) == 0:
                       continue
                       
                   for medio_acceso in ['Aeropuerto', 'Carretera', 'Puerto', 'Tren']:
                       record = base_data[base_data['Medio_Acceso'] == medio_acceso].iloc[-1].copy()
                       
                       # Ajustar año y fecha
                       record['Anio'] = year
                       record['Fecha'] = pd.to_datetime(f'{year}-{month:02d}-01')
                       
                       # Ajustar número de turistas
                       years_diff = year - 2023
                       base_tourists = record['Num_Turistas']
                       seasonal_factor = seasonal_pattern[month]
                       record['Num_Turistas'] = int(base_tourists * (growth_rate ** years_diff) * seasonal_factor)
                       
                       # Ajustar variables climáticas
                       season = self._get_season(month)
                       record['Temperatura_Media'] += self.climate_trends[season]['temp_trend'] * years_diff
                       record['Precipitacion_Media'] *= self.climate_trends[season]['precip_adjustment'] ** years_diff
                       
                       future_records.append(record)
       
       future_df = pd.DataFrame(future_records)
       future_df['Tipo_Dato'] = 'Datos Generados'
       return future_df

   def _get_season(self, month):
       if month in [12,1,2]: return 'invierno'
       elif month in [3,4,5]: return 'primavera'
       elif month in [6,7,8]: return 'verano'
       else: return 'otoño'

   def generate_complete_dataset(self):
       historical_df = self.historical_data.copy()
       historical_df['Tipo_Dato'] = 'Histórico'
       
       future_df = self.generate_future_data()
       combined_df = pd.concat([historical_df, future_df], ignore_index=True)
       combined_df.sort_values(['Fecha', 'Pais'], inplace=True)
       
       print("\nResumen del dataset generado:")
       print(f"Total registros: {len(combined_df)}")
       print(f"Período: {combined_df['Anio'].min()} - {combined_df['Anio'].max()}")
       print("\nDistribución por tipo de dato:")
       print(combined_df['Tipo_Dato'].value_counts())
       
       return combined_df

def main():
   generator = TourismDataGenerator()
   dataset_completo = generator.generate_complete_dataset()
   dataset_completo.to_csv('dataset_completo_2019_2028.csv', index=False)
   print("\nDataset guardado exitosamente")

if __name__ == "__main__":
   main()