#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: eeuu.py
Generador de Datos Turísticos de Estados Unidos (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes de Estados Unidos a España durante el período 2019-2023. Simula patrones 
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
   2019: {'total_turistas': 3324870, 'porc_paquete': 20.6},
   2020: {'total_turistas': 405810, 'porc_paquete': 9.8},
   2021: {'total_turistas': 797844, 'porc_paquete': 12.2},
   2022: {'total_turistas': 2801476, 'porc_paquete': 16.8},
   2023: {'total_turistas': 3835884, 'porc_paquete': 19.6}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 4.2, 2: 4.5, 3: 7.2, 4: 7.8, 5: 10.2, 6: 12.5,
       7: 12.8, 8: 8.5, 9: 11.2, 10: 9.2, 11: 7.2, 12: 4.7
   },
   2020: {
       1: 35.2, 2: 28.5, 3: 18.5, 4: 0.5, 5: 0.3, 6: 1.2,
       7: 3.5, 8: 4.2, 9: 3.8, 10: 2.8, 11: 2.5, 12: 2.8
   },
   2021: {
       1: 1.2, 2: 1.5, 3: 1.8, 4: 2.2, 5: 2.5, 6: 5.2,
       7: 15.5, 8: 14.2, 9: 17.2, 10: 16.5, 11: 13.2, 12: 9.0
   },
   2022: {
       1: 2.5, 2: 2.8, 3: 5.2, 4: 7.5, 5: 12.5, 6: 13.2,
       7: 12.8, 8: 8.5, 9: 12.2, 10: 11.5, 11: 7.2, 12: 4.1
   },
   2023: {
       1: 3.8, 2: 3.5, 3: 5.8, 4: 6.8, 5: 8.5, 6: 12.5,
       7: 13.8, 8: 10.2, 9: 8.2, 10: 11.2, 11: 9.8, 12: 5.9
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 89.8, 'Carretera': 3.4, 'Puerto': 5.7, 'Tren': 1.1},
   2020: {'Aeropuerto': 90.6, 'Carretera': 6.6, 'Puerto': 1.1, 'Tren': 1.7},
   2021: {'Aeropuerto': 86.5, 'Carretera': 5.7, 'Puerto': 6.5, 'Tren': 1.3},
   2022: {'Aeropuerto': 89.1, 'Carretera': 2.3, 'Puerto': 7.6, 'Tren': 1.0},
   2023: {'Aeropuerto': 86.7, 'Carretera': 2.0, 'Puerto': 9.9, 'Tren': 1.4}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Cataluña': 45.3,
       'Comunidad de Madrid': 25.9,
       'Andalucía': 10.3
   },
   2020: {
       'Cataluña': 36.5,
       'Comunidad de Madrid': 29.9,
       'Andalucía': 13.2
   },
   2021: {
       'Cataluña': 30.8,
       'Comunidad de Madrid': 30.5,
       'Andalucía': 11.7,
       'Baleares': 7.7,
       'Galicia': 5.3
   },
   2022: {
       'Cataluña': 41.5,
       'Comunidad de Madrid': 26.7,
       'Andalucía': 11.1,
       'Baleares': 6.0
   },
   2023: {
       'Cataluña': 41.6,
       'Comunidad de Madrid': 24.7,
       'Andalucía': 15.1,
       'Baleares': 7.2
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 2039, 'asientos': 567210, 'pasajeros': 349252},
   2021: {'vuelos': 3126, 'asientos': 775418, 'pasajeros': 528871},
   2022: {'vuelos': 8722, 'asientos': 2267243, 'pasajeros': 1946177},
   2023: {'vuelos': 10027, 'asientos': 2631390, 'pasajeros': 2308384}
}

# Aeropuertos principales
aeropuertos_principales = {
   2020: [
       {'nombre': 'J.Fitzgerald Kennedy N.York', 'pasajeros': 122692, 'porcentaje': 35.13},
       {'nombre': 'Miami International (Km)', 'pasajeros': 102300, 'porcentaje': 29.29}
   ],
   2021: [
       {'nombre': 'Miami International (Km)', 'pasajeros': 184801, 'porcentaje': 34.94},
       {'nombre': 'J.Fitzgerald Kennedy N.York', 'pasajeros': 169671, 'porcentaje': 32.08}
   ],
   2022: [
       {'nombre': 'J.Fitzgerald Kennedy N.York', 'pasajeros': 593962, 'porcentaje': 30.52},
       {'nombre': 'Miami International (Km)', 'pasajeros': 428178, 'porcentaje': 22.00}
   ],
   2023: [
       {'nombre': 'J.Fitzgerald Kennedy N.York', 'pasajeros': 655397, 'porcentaje': 28.39},
       {'nombre': 'Miami International (Km)', 'pasajeros': 455133, 'porcentaje': 19.72},
       {'nombre': 'Chicago / O\'hare (Ko)', 'pasajeros': 152678, 'porcentaje': 6.61}
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
       1: (uniform(0.0, 5.0), uniform(90.0, 110.0)),    
       2: (uniform(2.0, 7.0), uniform(80.0, 100.0)),   
       3: (uniform(5.0, 10.0), uniform(100.0, 120.0)),  
       4: (uniform(10.0, 15.0), uniform(90.0, 110.0)),  
       5: (uniform(15.0, 20.0), uniform(80.0, 100.0)),  
       6: (uniform(20.0, 25.0), uniform(70.0, 90.0)),  
       7: (uniform(23.0, 28.0), uniform(60.0, 80.0)),  
       8: (uniform(23.0, 28.0), uniform(60.0, 80.0)),  
       9: (uniform(20.0, 25.0), uniform(70.0, 90.0)),  
       10: (uniform(15.0, 20.0), uniform(80.0, 100.0)), 
       11: (uniform(8.0, 13.0), uniform(90.0, 110.0)), 
       12: (uniform(2.0, 7.0), uniform(90.0, 110.0))    
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
               'Pais': 'EE.UU.',
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
df.to_csv('turistas_eeuu_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))