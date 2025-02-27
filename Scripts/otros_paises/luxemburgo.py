#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: luxemburgo.py
Generador de Datos Turísticos de Luxemburgo (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes de Luxemburgo a España durante el período 2019-2023. Simula patrones 
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

datos_anuales = {
    2019: {'total_turistas': 192173, 'porc_paquete': 26.3},
    2020: {'total_turistas': 83062, 'porc_paquete': 21.8},
    2021: {'total_turistas': 201851, 'porc_paquete': 34.4},
    2022: {'total_turistas': 224270, 'porc_paquete': 22.7},
    2023: {'total_turistas': 284959, 'porc_paquete': 27.2}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 88.9, 'Carretera': 10.7, 'Puerto': 0.1, 'Tren': 0.3},
    2020: {'Aeropuerto': 79.0, 'Carretera': 20.8, 'Puerto': 0.0, 'Tren': 0.1},
    2021: {'Aeropuerto': 91.3, 'Carretera': 8.6, 'Puerto': 0.0, 'Tren': 0.0},
    2022: {'Aeropuerto': 85.4, 'Carretera': 14.1, 'Puerto': 0.5, 'Tren': 0.0},
    2023: {'Aeropuerto': 84.2, 'Carretera': 15.7, 'Puerto': 0.1, 'Tren': 0.0}
}

destinos_principales_por_año = {
    2019: {
        'Cataluña': 28.0, 'Baleares': 23.1, 'Canarias': 16.0,
        'Comunidad de Madrid': 11.1, 'Andalucía': 9.1
    },
    2020: {
        'Baleares': 24.0, 'Cataluña': 22.0, 'Comunidad de Madrid': 11.5,
        'Andalucía': 11.5, 'Comunidad Valenciana': 9.5
    },
    2021: {
        'Baleares': 35.5, 'Cataluña': 17.9, 'Canarias': 15.4,
        'Comunidad de Madrid': 14.4, 'Andalucía': 6.0
    },
    2022: {
        'Baleares': 31.1, 'Cataluña': 20.2, 'Comunidad de Madrid': 12.3,
        'Canarias': 10.1, 'Andalucía': 9.6
    },
    2023: {
        'Baleares': 27.5, 'Comunidad de Madrid': 17.9, 'Andalucía': 14.0,
        'Canarias': 12.5, 'Cataluña': 9.8
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2021: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2022: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2023: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'}
}

def normalizar_probabilidades(destinos):
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
    base = 8.5 if tipo_organizacion == 'Paquete' else 8.2
    if mes in [6, 7, 8]:
        base += uniform(0.2, 0.6)
    elif mes in [3, 10, 11]:
        base += uniform(0.1, 0.4)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

def get_temperatura_precipitacion(mes):
    temp_precip = {
        1: (uniform(0.0, 4.0), uniform(70.0, 80.0)),
        2: (uniform(1.0, 5.0), uniform(65.0, 75.0)),
        3: (uniform(4.0, 8.0), uniform(60.0, 70.0)),
        4: (uniform(7.0, 12.0), uniform(55.0, 65.0)),
        5: (uniform(11.0, 16.0), uniform(50.0, 60.0)),
        6: (uniform(14.0, 19.0), uniform(45.0, 55.0)),
        7: (uniform(16.0, 21.0), uniform(40.0, 50.0)),
        8: (uniform(16.0, 21.0), uniform(40.0, 50.0)),
        9: (uniform(13.0, 18.0), uniform(45.0, 55.0)),
        10: (uniform(9.0, 14.0), uniform(55.0, 65.0)),
        11: (uniform(4.0, 9.0), uniform(65.0, 75.0)),
        12: (uniform(1.0, 5.0), uniform(70.0, 80.0))
    }
    return temp_precip[mes]

def get_estacionalidad(porcentaje):
    if porcentaje >= 11:
        return 'Alta'
    elif porcentaje >= 8:
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
        porcentaje_mes = 8.33  # Distribución uniforme si no hay datos específicos
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
                'Pais': 'Luxemburgo',
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
df.to_csv('turistas_luxemburgo_2019_2023.csv', index=False, encoding='utf-8', decimal=',')