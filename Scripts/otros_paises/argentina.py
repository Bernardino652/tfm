#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: argentina.py
Generador de Datos Turísticos de Alemania (2019-2023).

Este script genera un conjunto de datos reales y simulados que representa el flujo turístico
de visitantes argentina a España durante el período 2019-2023. Simula patrones 
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
    2019: {'total_turistas': 721697, 'porc_paquete': 6.3},
    2020: {'total_turistas': 162311, 'porc_paquete': 3.5},
    2021: {'total_turistas': 64287, 'porc_paquete': 7.0},
    2022: {'total_turistas': 495351, 'porc_paquete': 5.1},
    2023: {'total_turistas': 572276, 'porc_paquete': 4.0}
}

# Distribución mensual por año
distribucion_mensual = {
    2019: {
        1: 6.5, 2: 5.8, 3: 6.2, 4: 7.5, 5: 8.2, 6: 12.5, 7: 8.5, 
        8: 11.8, 9: 12.2, 10: 9.5, 11: 6.8, 12: 4.5
    },
    2020: {
        1: 30.0, 2: 28.5, 3: 15.0, 4: 2.5, 5: 1.0, 6: 1.2, 7: 2.8,
        8: 3.5, 9: 5.2, 10: 4.2, 11: 3.5, 12: 2.6
    },
    2021: {
        1: 5.0, 2: 3.2, 3: 2.5, 4: 5.8, 5: 7.2, 6: 8.5, 7: 10.5,
        8: 8.2, 9: 10.8, 10: 8.5, 11: 15.2, 12: 14.6
    },
    2022: {
        1: 4.5, 2: 3.8, 3: 6.5, 4: 7.8, 5: 8.2, 6: 10.2, 7: 7.5,
        8: 12.5, 9: 13.2, 10: 11.5, 11: 9.2, 12: 5.1
    },
    2023: {
        1: 7.8, 2: 5.5, 3: 6.2, 4: 8.5, 5: 9.2, 6: 9.8, 7: 10.2,
        8: 10.5, 9: 10.2, 10: 9.8, 11: 7.2, 12: 5.1
    }
}

# Medios de acceso por año
medios_acceso_por_año = {
    2019: {'Aeropuerto': 92.3, 'Carretera': 4.7, 'Puerto': 2.0, 'Tren': 1.0},
    2020: {'Aeropuerto': 75.7, 'Carretera': 20.3, 'Puerto': 0.9, 'Tren': 3.0},
    2021: {'Aeropuerto': 86.5, 'Carretera': 4.2, 'Puerto': 0.4, 'Tren': 8.9},
    2022: {'Aeropuerto': 94.2, 'Carretera': 1.1, 'Puerto': 2.9, 'Tren': 1.8},
    2023: {'Aeropuerto': 90.7, 'Carretera': 2.9, 'Puerto': 4.1, 'Tren': 2.4}
}

# Destinos principales por año
destinos_principales_por_año = {
    2019: {
        'Comunidad de Madrid': 42.8, 'Cataluña': 29.1, 'Andalucía': 10.2, 'Baleares': 6.4
    },
    2020: {
        'Comunidad de Madrid': 32.3, 'Andalucía': 27.1, 'Cataluña': 23.7
    },
    2021: {
        'Comunidad de Madrid': 42.9, 'Cataluña': 25.0, 'Comunidad Valenciana': 12.3,
        'Andalucía': 7.5, 'Baleares': 5.3
    },
    2022: {
        'Comunidad de Madrid': 47.4, 'Cataluña': 25.4, 'Andalucía': 9.7,
        'Comunidad Valenciana': 5.5
    },
    2023: {
        'Comunidad de Madrid': 45.6, 'Cataluña': 24.7, 'Andalucía': 8.9,
        'Baleares': 7.3, 'Comunidad Valenciana': 5.0
    }
}

# Información de vuelos por año
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 622, 'asientos': 193697, 'pasajeros': 161701},
    2021: {'vuelos': 656, 'asientos': 208346, 'pasajeros': 160046},
    2022: {'vuelos': 1865, 'asientos': 575992, 'pasajeros': 543937},
    2023: {'vuelos': 2152, 'asientos': 671176, 'pasajeros': 628527}
}

def normalizar_probabilidades(destinos):
    """Normaliza las probabilidades para que sumen 1"""
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
        1: (uniform(6.0, 8.0), uniform(50.0, 60.0)),    
        2: (uniform(8.0, 10.0), uniform(45.0, 55.0)),   
        3: (uniform(11.0, 13.0), uniform(35.0, 45.0)),  
        4: (uniform(13.0, 15.0), uniform(40.0, 50.0)),  
        5: (uniform(16.0, 19.0), uniform(45.0, 55.0)),  
        6: (uniform(21.0, 25.0), uniform(20.0, 30.0)),  
        7: (uniform(24.0, 28.0), uniform(10.0, 20.0)),  
        8: (uniform(24.0, 28.0), uniform(10.0, 20.0)),  
        9: (uniform(21.0, 24.0), uniform(25.0, 35.0)),  
        10: (uniform(15.0, 18.0), uniform(45.0, 55.0)), 
        11: (uniform(10.0, 13.0), uniform(55.0, 65.0)), 
        12: (uniform(7.0, 9.0), uniform(50.0, 60.0))    
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
            
            # Normalizar probabilidades de destinos
            destinos = destinos_principales_por_año[year]
            destinos_norm = normalizar_probabilidades(destinos)
            
            row = {
                'Año': year,
                'mes': mes,
                'Pais': 'Argentina',
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
df.to_csv('turistas_argentina_2019_2023.csv', index=False, encoding='utf-8', decimal=',')

# Mostrar las primeras filas
print(df.head(20))