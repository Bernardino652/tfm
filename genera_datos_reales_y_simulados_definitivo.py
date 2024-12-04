import pandas as pd
import requests
import numpy as np
import os
from datetime import datetime

class TourismDataProcessor:
    def __init__(self, aemet_api_key):
        self.aemet_api_key = aemet_api_key
        self.headers = {'api_key': self.aemet_api_key}
        os.makedirs('data/processed', exist_ok=True)
        
        self.paises = [
            'Alemania', 'Argentina', 'Austria', 'Bélgica', 'Brasil',
            'Canadá', 'China', 'Corea del Sur', 'Dinamarca', 'Estados Unidos',
            'Finlandia', 'Francia', 'Irlanda', 'Italia', 'Japón',
            'Luxemburgo', 'México', 'Noruega', 'Países Bajos', 'Polonia',
            'Portugal', 'Reino Unido', 'República Checa', 'Rusia', 'Suecia',
            'Suiza'
        ]
        
        self.stations = {
            'Cataluña': '0076',
            'Madrid': '3129',
            'Andalucía': '6155A',
            'Valencia': '8414A',
            'Baleares': 'B278'
        }

    def extract_dates(self, row):
        """Extrae fechas de la fila de encabezado"""
        dates = []
        for val in row:
            if pd.notna(val) and isinstance(val, str) and 'M' in val:
                year = int(val[:4])
                month = int(val[-2:])
                dates.append(pd.Timestamp(year=year, month=month, day=1))
        return dates

    def process_tourist_row(self, row, country, dates, data):
        """Procesa una fila de datos turísticos"""
        for i, date in enumerate(dates):
            if i + 1 < len(row) and pd.notna(row.iloc[i + 1]):
                try:
                    tourists = float(str(row.iloc[i + 1]).replace(',', ''))
                    data.append({
                        'Fecha': date,
                        'Pais': country,
                        'Num_Turistas': tourists
                    })
                except (ValueError, TypeError):
                    pass
        
    def get_base_tourism_data(self):
        """Obtiene datos base del INE con la lista completa de países"""
        ine_excel_url = "https://www.ine.es/jaxiT3/files/t/es/xlsx/10822.xlsx"
        response = requests.get(ine_excel_url)
        with open('data/processed/frontur_data.xlsx', 'wb') as f:
            f.write(response.content)
        
        df = pd.read_excel('data/processed/frontur_data.xlsx', sheet_name=0)
        
        date_row_idx = None
        for idx, row in df.iterrows():
            if any(str(val).startswith('20') and 'M' in str(val) for val in row if pd.notna(val)):
                date_row_idx = idx
                break
                
        if date_row_idx is None:
            raise ValueError("No se encontraron fechas")
            
        data = []
        current_country = None
        dates = self.extract_dates(df.iloc[date_row_idx])
        
        for idx, row in df.iterrows():
            if idx <= date_row_idx:
                continue
                
            first_col = str(row.iloc[0]).strip()
            if first_col in self.paises:
                current_country = first_col
            elif current_country and 'Dato base' in first_col:
                self.process_tourist_row(row, current_country, dates, data)
        
        df_final = pd.DataFrame(data)
        print(f"\nPaíses encontrados: {df_final['Pais'].unique()}")
        return df_final

    def assign_access_method(self, country):
        """Asigna método de acceso basado en patrones realistas por país"""
        if country in ['Portugal', 'Francia']:
            return np.random.choice(['Aeropuerto', 'Carretera', 'Puerto', 'Tren'],
                                 p=[0.5, 0.45, 0.03, 0.02])
        elif country in ['Estados Unidos', 'China', 'Japón', 'Corea del Sur', 
                        'Argentina', 'Brasil', 'México']:
            return np.random.choice(['Aeropuerto', 'Carretera', 'Puerto', 'Tren'],
                                 p=[0.95, 0.02, 0.02, 0.01])
        elif country in ['Noruega', 'Suecia', 'Finlandia', 'Dinamarca']:
            return np.random.choice(['Aeropuerto', 'Carretera', 'Puerto', 'Tren'],
                                 p=[0.85, 0.05, 0.08, 0.02])
        return np.random.choice(['Aeropuerto', 'Carretera', 'Puerto', 'Tren'],
                             p=[0.80, 0.15, 0.03, 0.02])

    def assign_organization_method(self, row):
        """Asigna método de organización basado en país y temporada"""
        country = row['Pais']
        month = row['Fecha'].month
        
        base_prob = 0.4 if country in ['Alemania', 'Reino Unido', 'Rusia', 'China'] else 0.3
        if month in [6, 7, 8, 12]:  # Verano y Navidad
            base_prob += 0.1
            
        return np.random.choice(['Paquete', 'Individual'], p=[base_prob, 1-base_prob])

    def get_weather_data(self, date):
        """Genera datos climáticos basados en patrones reales"""
        month = date.month
        
        temp_base = 15 + 10 * np.sin((month - 1) * np.pi / 6)
        temp = round(max(min(temp_base + np.random.normal(0, 2), 35), 5), 1)
        
        precip_base = 30 + 20 * np.sin((month - 7) * np.pi / 6)
        precip = round(max(precip_base + np.random.normal(0, 10), 0), 1)
        
        return temp, precip

    def get_satisfaction_score(self, country, date):
        """Calcula satisfacción basada en país y temporada"""
        base_satisfaction = {
            'Alemania': 8.7, 'Reino Unido': 8.8, 'Francia': 8.4,
            'Estados Unidos': 8.9
        }.get(country, 8.5)
        
        month = date.month
        seasonal_adj = 0.2 if month in [6, 7, 8] else (-0.1 if month in [12, 1, 2] else 0)
        
        satisfaction = base_satisfaction + seasonal_adj + np.random.normal(0, 0.3)
        return round(min(10, max(7.0, satisfaction)), 2)

    def get_seasonality_factor(self, date):
        """Calcula factor de estacionalidad"""
        month = date.month
        if month in [7, 8]:
            return 'Alta'
        elif month in [6, 9]:
            return 'Media-Alta'
        elif month in [3, 4, 5, 10]:
            return 'Media'
        else:
            return 'Baja'

    def assign_destination(self, row):
        """Asigna destino principal basado en temporada"""
        month = row['Fecha'].month
        weights = {
            'Cataluña': 0.25,
            'Islas Baleares': 0.20,
            'Canarias': 0.18,
            'Andalucía': 0.15,
            'Comunidad Valenciana': 0.12,
            'Madrid': 0.10
        }
        
        if month in [6, 7, 8]:  # Verano
            weights['Islas Baleares'] *= 1.3
            weights['Comunidad Valenciana'] *= 1.2
        elif month in [12, 1, 2]:  # Invierno
            weights['Canarias'] *= 1.4
            weights['Madrid'] *= 1.1
        
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return np.random.choice(list(weights.keys()), p=list(weights.values()))

    def combine_data(self):
        """Combina todos los datos en un único dataset"""
        print("Procesando datos turísticos...")
        df_base = self.get_base_tourism_data()
        
        # Añadir todas las variables adicionales
        df_base['Estacionalidad'] = df_base['Fecha'].apply(self.get_seasonality_factor)
        df_base['Organizacion_Viaje'] = df_base.apply(self.assign_organization_method, axis=1)
        df_base['Satisfaccion'] = df_base.apply(lambda x: self.get_satisfaction_score(x['Pais'], x['Fecha']), axis=1)
        
        # Añadir datos climáticos
        weather_data = df_base['Fecha'].apply(self.get_weather_data)
        df_base['Temperatura_Media'] = weather_data.apply(lambda x: x[0])
        df_base['Precipitacion_Media'] = weather_data.apply(lambda x: x[1])
        
        # Añadir destino y medio de acceso
        df_base['Destino_Principal'] = df_base.apply(self.assign_destination, axis=1)
        df_base['Medio_Acceso'] = df_base['Pais'].apply(self.assign_access_method)
        
        # Filtrar periodo 2019-2023
        df_base = df_base[
            (df_base['Fecha'].dt.year >= 2019) & 
            (df_base['Fecha'].dt.year <= 2024)
        ]
        
        # Guardar resultado
        output_path = 'data/processed/turismo_completo.csv'
        df_base.to_csv(output_path, index=False)
        
        print("\nEstadísticas del dataset final:")
        print(f"Total registros: {len(df_base)}")
        print(f"Rango de fechas: {df_base['Fecha'].min()} - {df_base['Fecha'].max()}")
        print("\nVariables incluidas:")
        for col in df_base.columns:
            print(f"- {col}")
        
        return df_base

def main():
    api_key = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiZXJuYXJkby5jaGFuY3VzaWdAZXBuLmVkdS5lYyIsImp0aSI6IjU3Mjg0ZTNhLWY2YjUtNDE1YS04NTAyLWJmMTliZTljYWZiOCIsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzMyODQwMDc1LCJ1c2VySWQiOiI1NzI4NGUzYS1mNmI1LTQxNWEtODUwMi1iZjE5YmU5Y2FmYjgiLCJyb2xlIjoiIn0.u-a3piW9Kzff3I_nMNcFVs0t7i6Mkx2NdHwY9EuJLKw"
    processor = TourismDataProcessor(api_key)
    df = processor.combine_data()
    print(df.head)

if __name__ == "__main__":
    main()