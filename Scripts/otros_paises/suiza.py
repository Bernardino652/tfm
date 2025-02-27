#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: suiza.py
Generador de Datos Turísticos de Suiza (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes de Suiza a España durante el período 2019-2023. Simula patrones 
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
    2019: {'total_turistas': 1811865, 'porc_paquete': 19.1},
    2020: {'total_turistas': 397074, 'porc_paquete': 12.8},
    2021: {'total_turistas': 945710, 'porc_paquete': 13.1},
    2022: {'total_turistas': 1654733, 'porc_paquete': 14.2},
    2023: {'total_turistas': 2002197, 'porc_paquete': 13.7}
}

# Distribución mensual por año
distribucion_mensual = {
    2019: {
        1: 5.2, 2: 5.5, 3: 6.8, 4: 8.2, 5: 9.5, 6: 9.8, 
        7: 8.5, 8: 12.5, 9: 10.8, 10: 10.5, 11: 7.2, 12: 5.5
    },
    2020: {
        1: 20.0, 2: 19.5, 3: 10.0, 4: 1.0, 5: 0.5, 6: 1.0, 
        7: 2.0, 8: 20.0, 9: 5.0, 10: 3.5, 11: 3.0, 12: 4.5
    },
    2021: {
        1: 2.0, 2: 1.5, 3: 2.0, 4: 3.5, 5: 5.0, 6: 7.5, 
        7: 16.5, 8: 15.0, 9: 12.0, 10: 20.0, 11: 8.0, 12: 7.0
    },
    2022: {
        1: 3.5, 2: 4.0, 3: 5.0, 4: 5.5, 5: 10.5, 6: 9.5, 
        7: 13.5, 8: 10.5, 9: 10.5, 10: 12.5, 11: 4.5, 12: 4.5
    },
    2023: {
        1: 5.0, 2: 5.0, 3: 5.0, 4: 7.0, 5: 10.0, 6: 9.0,
        7: 12.0, 8: 11.0, 9: 10.5, 10: 11.0, 11: 5.5, 12: 5.0
    }
}

# Medios de acceso por año
medios_acceso_por_año = {
    2019: {'Aeropuerto': 89.8, 'Carretera': 9.5, 'Tren': 0.3, 'Puerto': 0.3},
    2020: {'Aeropuerto': 78.6, 'Carretera': 20.9, 'Tren': 0.3, 'Puerto': 0.2},
    2021: {'Aeropuerto': 82.8, 'Carretera': 16.9, 'Tren': 0.2, 'Puerto': 0.1},
    2022: {'Aeropuerto': 87.1, 'Carretera': 12.0, 'Tren': 0.4, 'Puerto': 0.6},
    2023: {'Aeropuerto': 86.6, 'Carretera': 12.5, 'Tren': 0.3, 'Puerto': 0.5}
}

# Destinos principales por año
destinos_principales_por_año = {
    2019: {
        'Baleares': 26.0, 'Cataluña': 17.1, 'Comunidad Valenciana': 14.6,
        'Andalucía': 14.3, 'Canarias': 10.5
    },
    2020: {
        'Comunidad Valenciana': 19.2, 'Canarias': 16.2, 'Baleares': 15.7,
        'Andalucía': 11.5, 'Cataluña': 11.2
    },
    2021: {
        'Baleares': 32.3, 'Cataluña': 16.4, 'Comunidad Valenciana': 15.9,
        'Andalucía': 11.6, 'Canarias': 9.0
    },
    2022: {
        'Baleares': 24.9, 'Cataluña': 18.5, 'Comunidad Valenciana': 18.3,
        'Andalucía': 13.2, 'Canarias': 11.1
    },
    2023: {
        'Baleares': 24.5, 'Comunidad Valenciana': 18.7, 'Cataluña': 16.3,
        'Canarias': 11.8, 'Andalucía': 10.8
    }
}

# Información de vuelos por año
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 7561, 'asientos': 1169412, 'pasajeros': 734325},
    2021: {'vuelos': 13934, 'asientos': 2173433, 'pasajeros': 1563450},
    2022: {'vuelos': 21703, 'asientos': 3544804, 'pasajeros': 2814560},
    2023: {'vuelos': 23816, 'asientos': 3903471, 'pasajeros': 3262456}
}

def normalizar_probabilidades(destinos):
    """Normaliza las probabilidades para que sumen 1"""
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
    base = 8.7 if tipo_organizacion == 'Paquete' else 8.4
    if mes in [7, 8]:  # Alta temporada de verano
        base += uniform(0.3, 0.7)
    elif mes in [6, 9]:  # Temporadas intermedias
        base += uniform(0.2, 0.5)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

def get_temperatura_precipitacion(mes):
    # Temperaturas y precipitaciones adaptadas para destinos españoles populares entre turistas suizos
    temp_precip = {
        1: (uniform(10.0, 14.0), uniform(30.0, 40.0)),    
        2: (uniform(11.0, 15.0), uniform(30.0, 40.0)),   
        3: (uniform(13.0, 17.0), uniform(30.0, 40.0)),  
        4: (uniform(15.0, 19.0), uniform(40.0, 50.0)),  
        5: (uniform(18.0, 22.0), uniform(40.0, 50.0)),  
        6: (uniform(22.0, 26.0), uniform(20.0, 30.0)),  
        7: (uniform(25.0, 29.0), uniform(10.0, 20.0)),  
        8: (uniform(25.0, 29.0), uniform(10.0, 20.0)),  
        9: (uniform(22.0, 26.0), uniform(30.0, 40.0)),  
        10: (uniform(18.0, 22.0), uniform(40.0, 50.0)), 
        11: (uniform(14.0, 18.0), uniform(40.0, 50.0)), 
        12: (uniform(11.0, 15.0), uniform(30.0, 40.0))    
    }
    return temp_precip[mes]

def get_estacionalidad(porcentaje):
    if porcentaje >= 11:
        return 'Alta'
    elif porcentaje >= 9:
        return 'Media-Alta'
    elif porcentaje >= 7:
        return 'Media'
    else:
        return 'Baja'

# Crear lista para almacenar los datos
data_rows = []

# Generar datos para todos los años
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
                'Pais': 'Suiza',
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

# Crear DataFrame
df = pd.DataFrame(data_rows)

# Agregar puntuación de satisfacción
df['Satisfaccion'] = df.apply(lambda x: generar_satisfaccion(x['mes'], x['Organizacion_Satisfaccion']), axis=1)

# Formatear números
df['Num_Turistas'] = df['Num_Turistas'].apply(lambda x: '{:,}'.format(x).replace(',', '.'))

# Ordenar el DataFrame
df = df.sort_values(['Año', 'mes', 'Medio_Acceso'])

# Guardar el DataFrame
df.to_csv('turistas_suiza_2019_2023.csv', index=False, encoding='utf-8', decimal=',')