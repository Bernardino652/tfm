#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: canada.py
Generador de Datos Turísticos de Alemania (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes de Canada a España durante el período 2019-2023. Simula patrones 
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
   2019: {'total_turistas': 450663, 'porc_paquete': 21.0},
   2020: {'total_turistas': 115130, 'porc_paquete': 13.1},
   2021: {'total_turistas': 139449, 'porc_paquete': 10.2},
   2022: {'total_turistas': 380100, 'porc_paquete': 15.4},
   2023: {'total_turistas': 653628, 'porc_paquete': 18.9}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 4.8, 2: 7.2, 3: 9.8, 4: 10.2, 5: 10.5, 6: 9.8,
       7: 8.2, 8: 7.5, 9: 10.2, 10: 9.8, 11: 6.5, 12: 5.5
   },
   2020: {
       1: 18.5, 2: 16.2, 3: 10.5, 4: 0.5, 5: 0.3, 6: 10.2,
       7: 5.8, 8: 6.2, 9: 14.2, 10: 8.5, 11: 4.2, 12: 4.9
   },
   2021: {
       1: 2.5, 2: 3.8, 3: 4.8, 4: 5.2, 5: 5.5, 6: 3.2,
       7: 12.5, 8: 16.8, 9: 13.2, 10: 13.5, 11: 11.2, 12: 7.8
   },
   2022: {
       1: 3.2, 2: 4.2, 3: 4.8, 4: 5.2, 5: 7.2, 6: 11.5,
       7: 11.8, 8: 12.5, 9: 11.2, 10: 15.8, 11: 6.8, 12: 5.8
   },
   2023: {
       1: 3.5, 2: 3.2, 3: 4.8, 4: 8.2, 5: 8.5, 6: 15.2,
       7: 8.5, 8: 10.8, 9: 11.2, 10: 11.0, 11: 9.2, 12: 5.9
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 86.5, 'Carretera': 6.6, 'Puerto': 5.3, 'Tren': 1.6},
   2020: {'Aeropuerto': 81.8, 'Carretera': 14.8, 'Puerto': 0.8, 'Tren': 2.6},
   2021: {'Aeropuerto': 82.4, 'Carretera': 15.2, 'Puerto': 1.1, 'Tren': 1.3},
   2022: {'Aeropuerto': 84.7, 'Carretera': 8.5, 'Puerto': 5.8, 'Tren': 1.0},
   2023: {'Aeropuerto': 81.0, 'Carretera': 4.4, 'Puerto': 13.3, 'Tren': 1.4}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Cataluña': 49.9, 'Andalucía': 15.5,
       'Comunidad de Madrid': 13.8, 'Baleares': 6.0
   },
   2020: {
       'Cataluña': 38.6, 'Comunidad de Madrid': 16.4, 'Andalucía': 15.5,
       'Canarias': 6.8, 'Comunidad Valenciana': 6.8
   },
   2021: {
       'Cataluña': 25.9, 'Comunidad de Madrid': 19.8, 'Andalucía': 15.0,
       'Extremadura': 7.7, 'Canarias': 7.4
   },
   2022: {
       'Cataluña': 39.2, 'Andalucía': 18.4, 'Comunidad de Madrid': 13.5,
       'Baleares': 7.2, 'Comunidad Valenciana': 6.8
   },
   2023: {
       'Cataluña': 47.3, 'Andalucía': 18.9, 'Comunidad de Madrid': 11.6,
       'Baleares': 7.8, 'Comunidad Valenciana': 5.6
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2021: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2022: {'vuelos': 762, 'asientos': 203908, 'pasajeros': 167695},
   2023: {'vuelos': 1051, 'asientos': 307276, 'pasajeros': 273273}
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
       1: (uniform(-5.0, 0.0), uniform(40.0, 60.0)),
       2: (uniform(-3.0, 2.0), uniform(35.0, 55.0)),
       3: (uniform(2.0, 7.0), uniform(40.0, 60.0)),
       4: (uniform(7.0, 12.0), uniform(50.0, 70.0)),
       5: (uniform(12.0, 17.0), uniform(60.0, 80.0)),
       6: (uniform(17.0, 22.0), uniform(70.0, 90.0)),
       7: (uniform(20.0, 25.0), uniform(65.0, 85.0)),
       8: (uniform(19.0, 24.0), uniform(60.0, 80.0)),
       9: (uniform(15.0, 20.0), uniform(55.0, 75.0)),
       10: (uniform(9.0, 14.0), uniform(50.0, 70.0)),
       11: (uniform(3.0, 8.0), uniform(45.0, 65.0)),
       12: (uniform(-2.0, 3.0), uniform(40.0, 60.0))
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
               'Pais': 'Canadá',
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
df.to_csv('turistas_canada_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))