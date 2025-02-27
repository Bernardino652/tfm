#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: dinamarca.py
Generador de Datos Turísticos de Dinmarca (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes de Dinamarca a España durante el período 2019-2023. Simula patrones 
estacionales, impacto de la pandemia COVID-19, y la subsecuente recuperación.

El script crea un archivo CSV con información detallada sobre:
- Distribución mensual de turistas
- Medios de acceso (avión, carretera, etc.)
- Destinos principales visitados
- Organización del viaje (paquete turístico vs. individual)
- Datos climáticos asociados
- Niveles de satisfacción
- Información sobre vuelos (cuando está disponible)

Los datos generados se basan en estadísticas reales pero incluyen
elementos simulados para crear un conjunto de datos completo y coherente.

Autor: Bernardino Chancusig Espin
Fecha: 25/02/2025
Versión: 1.0

Dependencias:
-----------
- pandas (1.5.0+): Manipulación y análisis de datos estructurados.
  Se utiliza para crear, manipular y exportar el DataFrame final a CSV.
  
- numpy (1.22.0+): Operaciones numéricas y selección aleatoria ponderada.
  Se utiliza específicamente para la selección aleatoria de destinos
  basada en sus probabilidades (np.random.choice con pesos).
  
- random: Generación de números aleatorios para datos sintéticos.
  Se usan las funciones uniform() para generar valores continuos en rangos
  específicos y choice() para selecciones aleatorias simples.
"""
import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 1202527, 'porc_paquete': 37.7},
   2020: {'total_turistas': 276863, 'porc_paquete': 37.3},
   2021: {'total_turistas': 623406, 'porc_paquete': 26.5},
   2022: {'total_turistas': 1144735, 'porc_paquete': 31.0},
   2023: {'total_turistas': 1177809, 'porc_paquete': 32.2}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 5.2, 2: 5.8, 3: 7.2, 4: 8.5, 5: 8.8, 6: 8.2,
       7: 8.5, 8: 13.5, 9: 10.2, 10: 11.5, 11: 7.2, 12: 5.4
   },
   2020: {
       1: 25.2, 2: 35.5, 3: 12.5, 4: 0.5, 5: 0.3, 6: 1.2,
       7: 8.5, 8: 12.5, 9: 6.2, 10: 4.2, 11: 2.5, 12: 2.8
   },
   2021: {
       1: 0.5, 2: 0.8, 3: 1.2, 4: 1.5, 5: 2.8, 6: 5.2,
       7: 21.5, 8: 12.5, 9: 14.2, 10: 18.5, 11: 11.2, 12: 10.1
   },
   2022: {
       1: 5.2, 2: 5.5, 3: 6.8, 4: 7.5, 5: 8.2, 6: 7.5,
       7: 15.5, 8: 7.8, 9: 10.2, 10: 10.8, 11: 8.5, 12: 6.5
   },
   2023: {
       1: 7.2, 2: 6.8, 3: 8.5, 4: 8.2, 5: 7.5, 6: 7.2,
       7: 13.5, 8: 8.2, 9: 8.5, 10: 11.2, 11: 6.8, 12: 6.4
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 97.7, 'Carretera': 1.8, 'Puerto': 0.4, 'Tren': 0.1},
   2020: {'Aeropuerto': 97.3, 'Carretera': 2.0, 'Puerto': 0.6, 'Tren': 0.0},
   2021: {'Aeropuerto': 97.6, 'Carretera': 1.9, 'Puerto': 0.4, 'Tren': 0.1},
   2022: {'Aeropuerto': 97.8, 'Carretera': 1.7, 'Puerto': 0.5, 'Tren': 0.1},
   2023: {'Aeropuerto': 97.5, 'Carretera': 1.9, 'Puerto': 0.6, 'Tren': 0.1}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Canarias': 25.9, 'Andalucía': 25.0, 'Baleares': 16.7,
       'Cataluña': 14.3, 'Comunidad Valenciana': 7.5
   },
   2020: {
       'Canarias': 41.5, 'Andalucía': 28.1, 'Cataluña': 8.2,
       'Baleares': 8.1, 'Comunidad Valenciana': 7.8
   },
   2021: {
       'Andalucía': 25.8, 'Baleares': 24.1, 'Canarias': 24.0,
       'Cataluña': 12.6, 'Comunidad Valenciana': 8.3
   },
   2022: {
       'Canarias': 25.4, 'Andalucía': 23.2, 'Baleares': 20.1,
       'Cataluña': 15.0, 'Comunidad Valenciana': 8.7
   },
   2023: {
       'Canarias': 25.6, 'Andalucía': 25.3, 'Baleares': 16.1,
       'Cataluña': 14.8, 'Comunidad Valenciana': 9.1
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 2731, 'asientos': 500739, 'pasajeros': 344961},
   2021: {'vuelos': 5422, 'asientos': 986385, 'pasajeros': 761943},
   2022: {'vuelos': 9552, 'asientos': 1755350, 'pasajeros': 1510248},
   2023: {'vuelos': 10050, 'asientos': 1863490, 'pasajeros': 1656800}
}

# Aeropuertos principales
aeropuertos_principales = {
   2020: [{'nombre': 'Copenhague/Kastrup', 'pasajeros': 265052, 'porcentaje': 76.84}],
   2021: [
       {'nombre': 'Copenhague/Kastrup', 'pasajeros': 578325, 'porcentaje': 75.90},
       {'nombre': 'Billund', 'pasajeros': 146451, 'porcentaje': 19.48}
   ],
   2022: [
       {'nombre': 'Copenhague/Kastrup', 'pasajeros': 1106607, 'porcentaje': 73.27},
       {'nombre': 'Billund', 'pasajeros': 310898, 'porcentaje': 20.59}
   ],
   2023: [
       {'nombre': 'Copenhague/Kastrup', 'pasajeros': 1193889, 'porcentaje': 72.06},
       {'nombre': 'Billund', 'pasajeros': 330078, 'porcentaje': 19.92},
       {'nombre': 'Aalborg', 'pasajeros': 89916, 'porcentaje': 5.43}
   ]
}

def normalizar_probabilidades(destinos):
   total = sum(destinos.values())
   return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
   base = 8.2 if tipo_organizacion == 'Paquete' else 8.0
   if mes in [6, 7, 8, 9]:
       base += uniform(0.3, 0.8)
   elif mes in [4, 5, 10]:
       base += uniform(0.1, 0.4)
   else:
       base += uniform(-0.2, 0.3)
   return round(base, 2)

def get_temperatura_precipitacion(mes):
   temp_precip = {
       1: (uniform(-1.0, 2.0), uniform(50.0, 70.0)),
       2: (uniform(-1.0, 3.0), uniform(40.0, 60.0)),
       3: (uniform(2.0, 6.0), uniform(40.0, 60.0)),
       4: (uniform(5.0, 10.0), uniform(35.0, 55.0)),
       5: (uniform(10.0, 15.0), uniform(40.0, 60.0)),
       6: (uniform(14.0, 18.0), uniform(50.0, 70.0)),
       7: (uniform(16.0, 20.0), uniform(60.0, 80.0)),
       8: (uniform(16.0, 20.0), uniform(65.0, 85.0)),
       9: (uniform(13.0, 17.0), uniform(60.0, 80.0)),
       10: (uniform(9.0, 13.0), uniform(60.0, 80.0)),
       11: (uniform(4.0, 8.0), uniform(60.0, 80.0)),
       12: (uniform(0.0, 4.0), uniform(55.0, 75.0))
   }
   return temp_precip[mes]

def get_estacionalidad(porcentaje):
   if porcentaje >= 12:
       return 'Alta'
   elif porcentaje >= 9:
       return 'Media-Alta'
   elif porcentaje >= 6:
       return 'Media'
   else:
       return 'Baja'

data_rows = []

for year in datos_anuales.keys():
   total_turistas = datos_anuales[year]['total_turistas']
   porc_paquete = datos_anuales[year]['porc_paquete']
   
   for mes in range(1, 13):
       porcentaje_mes = distribucion_mensual[year][mes]
       turistas_mes = round(total_turistas * porcentaje_mes / 100)
       
       for medio, porcentaje in medios_acceso_por_año[year].items():
           turistas_medio = round(turistas_mes * porcentaje / 100)
           tipo_org = 'Paquete' if uniform(0, 100) < porc_paquete else 'Individual'
           temp, precip = get_temperatura_precipitacion(mes)
           
           destinos = destinos_principales_por_año[year]
           destinos_norm = normalizar_probabilidades(destinos)
           
           row = {
               'Año': year,
               'mes': mes,
               'Pais': 'Dinamarca',
               'Num_Turistas': turistas_medio,
               'Estacionalidad': get_estacionalidad(porcentaje_mes),
               'Organizacion_Satisfaccion': tipo_org,
               'Temperatura': round(temp, 1),
               'Precipitacion': round(precip, 1),
               'Destino_Principal': np.random.choice(
                   list(destinos_norm.keys()),
                   p=list(destinos_norm.values())
               ),
               'Medio_Acceso': medio,
               'Num_Vuelos': info_vuelos[year]['vuelos'],
               'Num_Asientos': info_vuelos[year]['asientos'],
               'Num_Pasajeros': info_vuelos[year]['pasajeros']
           }
           data_rows.append(row)

df = pd.DataFrame(data_rows)
df['Satisfaccion'] = df.apply(lambda x: generar_satisfaccion(x['mes'], x['Organizacion_Satisfaccion']), axis=1)
df['Num_Turistas'] = df['Num_Turistas'].apply(lambda x: '{:,}'.format(x).replace(',', '.'))
df = df.sort_values(['Año', 'mes', 'Medio_Acceso'])
df.to_csv('turistas_dinamarca_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))