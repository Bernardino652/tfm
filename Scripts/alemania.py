#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
script: alemania.py
Generador de Datos Turísticos de Alemania (2019-2023).

Este script genera un conjunto de datos sintéticos que representa el flujo turístico
de visitantes alemanes a España durante el período 2019-2023. Simula patrones 
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
elementos sintéticos para crear un conjunto de datos completo y coherente.

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

import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas y muestreo aleatorio ponderado
from random import uniform, choice  # Para generación de valores aleatorios

def normalizar_probabilidades(destinos):
    """
    Normaliza un diccionario de probabilidades para que sumen 1.
    
    Parameters
    ----------
    destinos : dict
        Diccionario con destinos como claves y porcentajes como valores.
        
    Returns
    -------
    dict
        Diccionario normalizado donde los valores suman exactamente 1.
        
    Examples
    --------
    >>> normalizar_probabilidades({'Baleares': 40.6, 'Canarias': 22.6})
    {'Baleares': 0.6423501423501424, 'Canarias': 0.3576498576498577}
    """
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def get_temperatura_precipitacion(mes):
    """
    Genera valores aleatorios de temperatura y precipitación para cada mes.
    
    Los rangos de temperatura y precipitación varían según la estacionalidad
    típica del clima mediterráneo español.
    
    Parameters
    ----------
    mes : int
        Número del mes (1-12).
        
    Returns
    -------
    tuple
        Tupla con (temperatura_media, precipitacion_media).
        
    Notes
    -----
    Los valores generados son sintéticos pero siguen patrones estacionales
    realistas para destinos turísticos españoles.
    """
    temp_precip = {
        1: (uniform(10.0, 14.0), uniform(30.0, 40.0)),    # Enero
        2: (uniform(11.0, 15.0), uniform(30.0, 40.0)),    # Febrero
        3: (uniform(13.0, 17.0), uniform(30.0, 40.0)),    # Marzo
        4: (uniform(15.0, 19.0), uniform(40.0, 50.0)),    # Abril
        5: (uniform(18.0, 22.0), uniform(40.0, 50.0)),    # Mayo
        6: (uniform(22.0, 26.0), uniform(20.0, 30.0)),    # Junio
        7: (uniform(25.0, 29.0), uniform(10.0, 20.0)),    # Julio
        8: (uniform(25.0, 29.0), uniform(10.0, 20.0)),    # Agosto
        9: (uniform(22.0, 26.0), uniform(30.0, 40.0)),    # Septiembre
        10: (uniform(18.0, 22.0), uniform(40.0, 50.0)),   # Octubre
        11: (uniform(14.0, 18.0), uniform(40.0, 50.0)),   # Noviembre
        12: (uniform(11.0, 15.0), uniform(30.0, 40.0))    # Diciembre
    }
    return temp_precip[mes]

def get_estacionalidad(porcentaje):
    """
    Determina el nivel de estacionalidad basado en el porcentaje mensual de turistas.
    
    Parameters
    ----------
    porcentaje : float
        Porcentaje de turistas del año que visitan en un mes específico.
        
    Returns
    -------
    str
        Categoría de estacionalidad: 'Alta', 'Media-Alta', 'Media' o 'Baja'.
        
    Notes
    -----
    Los umbrales son:
    - Alta: >= 11% del total anual
    - Media-Alta: >= 9% y < 11%
    - Media: >= 7% y < 9%
    - Baja: < 7%
    """
    if porcentaje >= 11:
        return 'Alta'
    elif porcentaje >= 9:
        return 'Media-Alta'
    elif porcentaje >= 7:
        return 'Media'
    else:
        return 'Baja'

def generar_satisfaccion(mes, tipo_organizacion, base_paquete=8.5, base_individual=8.2):
    """
    Genera el nivel de satisfacción basado en el mes y tipo de organización del viaje.
    
    Parameters
    ----------
    mes : int
        Número del mes (1-12).
    tipo_organizacion : str
        Tipo de organización del viaje ('Paquete' o 'Individual').
    base_paquete : float, optional
        Nivel base de satisfacción para viajes en paquete, por defecto 8.5.
    base_individual : float, optional
        Nivel base de satisfacción para viajes individuales, por defecto 8.2.
        
    Returns
    -------
    float
        Puntuación de satisfacción redondeada a 2 decimales (escala 0-10).
        
    Notes
    -----
    - Los viajes en paquete tienden a tener mayor satisfacción base.
    - La temporada afecta la satisfacción:
      - Alta temporada (jul-ago): mayor incremento
      - Temporada media (jun-sep): incremento moderado
      - Temporada baja: variación más aleatoria
    """
    base = base_paquete if tipo_organizacion == 'Paquete' else base_individual
    if mes in [7, 8]:  # Alta temporada de verano
        base += uniform(0.3, 0.7)
    elif mes in [6, 9]:  # Temporadas intermedias
        base += uniform(0.2, 0.5)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

# Datos históricos de turistas alemanes (2019-2023)
# Fuente: Basado en estadísticas turísticas oficiales
datos_anuales = {
    2019: {'total_turistas': 11158022, 'porc_paquete': 41.9},  # Año pre-pandemia
    2020: {'total_turistas': 2391437, 'porc_paquete': 36.0},   # Año COVID, gran caída
    2021: {'total_turistas': 5208894, 'porc_paquete': 33.6},   # Inicio recuperación
    2022: {'total_turistas': 9768600, 'porc_paquete': 39.1},   # Recuperación sustancial
    2023: {'total_turistas': 10989659, 'porc_paquete': 38.5}   # Casi recuperación total
}

# Distribución mensual de turistas (% del total anual)
# Refleja estacionalidad y efectos de la pandemia
distribucion_mensual = {
    # Año normal pre-pandemia
    2019: {1: 4.8, 2: 5.0, 3: 6.2, 4: 8.5, 5: 9.8, 6: 10.5, 7: 11.0, 8: 11.2, 9: 10.8, 10: 9.4, 11: 6.8, 12: 6.0},
    # Año COVID - Concentración en primeros meses y agosto (antes de restricciones y breve apertura)
    2020: {1: 20.0, 2: 18.5, 3: 10.5, 4: 1.5, 5: 1.0, 6: 2.0, 7: 3.5, 8: 17.0, 9: 4.5, 10: 4.0, 11: 3.5, 12: 4.0},
    # Recuperación inicial - Mayor concentración en segundo semestre
    2021: {1: 2.0, 2: 2.0, 3: 3.0, 4: 3.5, 5: 6.0, 6: 10.0, 7: 13.5, 8: 12.5, 9: 16.0, 10: 15.5, 11: 9.0, 12: 7.0},
    # Normalización parcial
    2022: {1: 3.0, 2: 4.0, 3: 5.0, 4: 6.5, 5: 10.0, 6: 11.5, 7: 12.0, 8: 12.0, 9: 12.0, 10: 11.0, 11: 7.5, 12: 5.5},
    # Patrón cercano a normalidad pre-pandemia
    2023: {1: 4.5, 2: 5.0, 3: 6.5, 4: 8.0, 5: 9.5, 6: 11.0, 7: 11.5, 8: 12.0, 9: 11.0, 10: 10.0, 11: 6.0, 12: 5.0}
}

# Distribución por medio de acceso (%)
# Muestra el predominio del avión para turistas alemanes
medios_acceso = {
    2019: {'Aeropuerto': 95.0, 'Carretera': 4.6, 'Tren': 0.1, 'Puerto': 0.3},
    2020: {'Aeropuerto': 91.9, 'Carretera': 7.7, 'Tren': 0.1, 'Puerto': 0.3},  # Mayor proporción por carretera durante COVID
    2021: {'Aeropuerto': 94.0, 'Carretera': 5.6, 'Tren': 0.1, 'Puerto': 0.2},
    2022: {'Aeropuerto': 95.1, 'Carretera': 4.4, 'Tren': 0.2, 'Puerto': 0.4},
    2023: {'Aeropuerto': 93.8, 'Carretera': 5.4, 'Tren': 0.1, 'Puerto': 0.7}
}

# Distribución por destinos principales (%)
# Refleja la preferencia por islas y la evolución durante/post pandemia
destinos_principales = {
    2019: {'Baleares': 40.6, 'Canarias': 22.6, 'Cataluña': 12.8, 'Andalucía': 10.0, 'Comunidad Valenciana': 5.4},
    2020: {'Canarias': 33.8, 'Baleares': 29.8, 'Andalucía': 10.1, 'Cataluña': 9.5, 'Comunidad Valenciana': 7.2},
    2021: {'Baleares': 45.1, 'Canarias': 23.8, 'Cataluña': 9.6, 'Andalucía': 8.6, 'Comunidad Valenciana': 6.9},
    2022: {'Baleares': 44.0, 'Canarias': 21.6, 'Cataluña': 12.4, 'Andalucía': 8.8, 'Comunidad Valenciana': 6.0},
    2023: {'Baleares': 41.9, 'Canarias': 21.7, 'Cataluña': 12.6, 'Andalucía': 9.4, 'Comunidad Valenciana': 6.7}
}

# Información sobre vuelos (cuando disponible)
# Muestra la evolución de la conectividad aérea
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 29242, 'asientos': 5066063, 'pasajeros': 3215497},  # Baja ocupación por COVID
    2021: {'vuelos': 49048, 'asientos': 8444068, 'pasajeros': 6313977},  # Recuperación parcial
    2022: {'vuelos': 78398, 'asientos': 14011945, 'pasajeros': 11927295},
    2023: {'vuelos': 82730, 'asientos': 14954324, 'pasajeros': 13294464}
}

def main():
    """
    Función principal que genera el conjunto de datos de turistas alemanes.
    
    Procesa los datos históricos definidos y genera un DataFrame completo
    con información detallada de turistas alemanes por año, mes, destino,
    medio de acceso y otras variables relevantes.
    
    El resultado se guarda en un archivo CSV.
    """
    print("Iniciando generación de datos turísticos de Alemania (2019-2023)...")
    
    # Crear lista para almacenar los datos
    data_rows = []

    # Generar datos para cada año
    for year in datos_anuales.keys():
        print(f"Procesando año {year}...")
        total_turistas = datos_anuales[year]['total_turistas']
        porc_paquete = datos_anuales[year]['porc_paquete']
        
        for mes in range(1, 13):
            porcentaje_mes = distribucion_mensual[year][mes]
            turistas_mes = round(total_turistas * porcentaje_mes / 100)
            
            # Para cada medio de acceso
            for medio, porcentaje in medios_acceso[year].items():
                turistas_medio = round(turistas_mes * porcentaje / 100)
                tipo_org = 'Paquete' if uniform(0, 100) < porc_paquete else 'Individual'
                temp, precip = get_temperatura_precipitacion(mes)
                
                # Normalizar probabilidades de destinos
                destinos = destinos_principales[year]
                destinos_norm = normalizar_probabilidades(destinos)
                
                # Crear fila de datos
                row = {
                    'Año': year,
                    'mes': mes,
                    'Pais': 'Alemania',
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
    print("Generando puntuaciones de satisfacción...")
    df['Satisfaccion'] = df.apply(lambda x: generar_satisfaccion(x['mes'], x['Organizacion_Satisfaccion']), axis=1)

    # Formatear números para formato español
    df['Num_Turistas'] = df['Num_Turistas'].apply(lambda x: '{:,}'.format(x).replace(',', '.'))

    # Ordenar el DataFrame
    df = df.sort_values(['Año', 'mes', 'Medio_Acceso'])

    # Guardar el DataFrame
    output_file = 'turistas_alemania_2019_2023.csv'
    df.to_csv(output_file, index=False, encoding='utf-8', decimal=',')
    print(f"Datos guardados exitosamente en {output_file}")
    print(f"Total de registros generados: {len(df)}")

if __name__ == "__main__":
    main()