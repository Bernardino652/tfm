import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
    2019: {'total_turistas': 1314078, 'porc_paquete': 26.4},
    2020: {'total_turistas': 155961, 'porc_paquete': 13.7},
    2021: {'total_turistas': 134242, 'porc_paquete': 7.2},
    2022: {'total_turistas': 267760, 'porc_paquete': 3.6},
    2023: {'total_turistas': 328625, 'porc_paquete': 1.0}
}

# Distribución mensual por año
distribucion_mensual = {
    2019: {
        1: 4.2, 2: 4.8, 3: 5.8, 4: 6.2, 5: 9.2, 6: 12.8, 
        7: 13.2, 8: 12.5, 9: 11.8, 10: 8.5, 11: 6.2, 12: 4.8
    },
    2020: {
        1: 40.0, 2: 30.0, 3: 15.0, 4: 2.0, 5: 1.0, 6: 1.5, 
        7: 2.0, 8: 2.5, 9: 2.0, 10: 2.0, 11: 1.5, 12: 0.5
    },
    2021: {
        1: 2.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 4.0, 6: 8.5,
        7: 15.0, 8: 14.5, 9: 15.5, 10: 14.0, 11: 10.5, 12: 8.0
    },
    2022: {
        1: 6.5, 2: 4.5, 3: 6.0, 4: 5.5, 5: 7.0, 6: 11.5,
        7: 12.0, 8: 12.5, 9: 11.0, 10: 9.0, 11: 8.0, 12: 6.5
    },
    2023: {
        1: 9.5, 2: 6.5, 3: 6.0, 4: 7.0, 5: 7.5, 6: 8.0,
        7: 13.5, 8: 11.0, 9: 10.0, 10: 8.0, 11: 7.0, 12: 6.0
    }
}

# Medios de acceso por año
medios_acceso_por_año = {
    2019: {'Aeropuerto': 96.2, 'Carretera': 0.7, 'Tren': 0.2, 'Puerto': 2.9},
    2020: {'Aeropuerto': 94.0, 'Carretera': 2.1, 'Tren': 1.3, 'Puerto': 2.6},
    2021: {'Aeropuerto': 95.8, 'Carretera': 3.0, 'Tren': 0.8, 'Puerto': 0.4},
    2022: {'Aeropuerto': 97.8, 'Carretera': 0.0, 'Tren': 0.2, 'Puerto': 2.0},
    2023: {'Aeropuerto': 97.3, 'Carretera': 0.5, 'Tren': 0.5, 'Puerto': 1.8}
}

# Destinos principales por año
destinos_principales_por_año = {
    2019: {
        'Cataluña': 20.5, 'Comunidad Valenciana': 14.1, 
        'Baleares': 9.2, 'Comunidad de Madrid': 8.4
    },
    2020: {
        'Cataluña': 25.9, 'Comunidad de Madrid': 13.5,
        'Canarias': 10.2, 'Andalucía': 7.0
    },
    2021: {
        'Cataluña': 41.4, 'Comunidad de Madrid': 17.6,
        'Comunidad Valenciana': 16.3
    },
    2022: {
        'Cataluña': 47.3, 'Comunidad de Madrid': 14.9, 'Andalucía': 12.7,
        'Comunidad Valenciana': 11.1, 'Canarias': 5.0
    },
    2023: {
        'Cataluña': 44.4, 'Comunidad de Madrid': 18.7, 'Comunidad Valenciana': 11.3,
        'País Vasco': 6.9, 'Andalucía': 6.8
    }
}

# Información de vuelos por año
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2021: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2022: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2023: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'}
}

def normalizar_probabilidades(destinos):
    """Normaliza las probabilidades para que sumen 1"""
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
    base = 8.5 if tipo_organizacion == 'Paquete' else 8.2
    if mes in [7, 8]:  # Alta temporada de verano
        base += uniform(0.3, 0.7)
    elif mes in [5, 6, 9]:  # Temporada media
        base += uniform(0.2, 0.5)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

def get_temperatura_precipitacion(mes):
    # Temperaturas y precipitaciones adaptadas para destinos españoles populares entre turistas rusos
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
    if porcentaje >= 12:
        return 'Alta'
    elif porcentaje >= 9:
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
            
            destinos = destinos_principales_por_año[year]
            destinos_norm = normalizar_probabilidades(destinos)
            
            row = {
                'Año': year,
                'mes': mes,
                'Pais': 'Rusia',
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
df.to_csv('turistas_rusia_2019_2023.csv', index=False, encoding='utf-8', decimal=',')