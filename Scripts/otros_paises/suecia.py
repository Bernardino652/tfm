import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
    2019: {'total_turistas': 2086452, 'porc_paquete': 37.9},
    2020: {'total_turistas': 458752, 'porc_paquete': 41.5},
    2021: {'total_turistas': 747351, 'porc_paquete': 20.9},
    2022: {'total_turistas': 1426410, 'porc_paquete': 24.4},
    2023: {'total_turistas': 1674706, 'porc_paquete': 29.7}
}

# Distribución mensual por año
distribucion_mensual = {
    2019: {
        1: 6.8, 2: 5.5, 3: 7.2, 4: 7.5, 5: 8.2, 6: 9.8, 
        7: 10.5, 8: 11.2, 9: 9.2, 10: 8.5, 11: 8.2, 12: 7.4
    },
    2020: {
        1: 21.0, 2: 19.5, 3: 12.0, 4: 1.5, 5: 0.5, 6: 0.5, 
        7: 1.0, 8: 8.0, 9: 7.0, 10: 8.5, 11: 4.5, 12: 3.5
    },
    2021: {
        1: 2.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 4.5, 6: 8.5, 
        7: 15.0, 8: 10.5, 9: 16.5, 10: 15.0, 11: 12.5, 12: 7.5
    },
    2022: {
        1: 4.5, 2: 5.0, 3: 6.0, 4: 7.5, 5: 8.5, 6: 9.0,
        7: 11.5, 8: 7.0, 9: 10.0, 10: 8.5, 11: 7.5, 12: 5.0
    },
    2023: {
        1: 6.0, 2: 5.5, 3: 6.5, 4: 7.0, 5: 8.0, 6: 8.5,
        7: 11.0, 8: 6.5, 9: 8.5, 10: 8.5, 11: 7.0, 12: 5.0
    }
}

# Medios de acceso por año
medios_acceso_por_año = {
    2019: {'Aeropuerto': 97.1, 'Carretera': 2.2, 'Tren': 0.2, 'Puerto': 0.5},
    2020: {'Aeropuerto': 97.3, 'Carretera': 2.7, 'Tren': 0.1, 'Puerto': 0.0},
    2021: {'Aeropuerto': 96.5, 'Carretera': 3.1, 'Tren': 0.1, 'Puerto': 0.3},
    2022: {'Aeropuerto': 97.7, 'Carretera': 1.8, 'Tren': 0.1, 'Puerto': 0.4},
    2023: {'Aeropuerto': 97.8, 'Carretera': 1.4, 'Tren': 0.3, 'Puerto': 0.5}
}

# Destinos principales por año
destinos_principales_por_año = {
    2019: {
        'Canarias': 27.9, 'Andalucía': 21.1, 'Baleares': 17.5,
        'Comunidad Valenciana': 17.1, 'Cataluña': 10.0
    },
    2020: {
        'Canarias': 43.0, 'Andalucía': 21.8, 'Comunidad Valenciana': 17.0,
        'Baleares': 7.8, 'Cataluña': 0.2
    },
    2021: {
        'Andalucía': 27.1, 'Canarias': 20.8, 'Comunidad Valenciana': 19.6,
        'Baleares': 19.1, 'Cataluña': 8.9
    },
    2022: {
        'Andalucía': 23.5, 'Comunidad Valenciana': 21.5, 'Canarias': 20.4,
        'Baleares': 16.5, 'Cataluña': 11.8
    },
    2023: {
        'Andalucía': 26.5, 'Canarias': 23.1, 'Comunidad Valenciana': 20.2,
        'Baleares': 14.0, 'Cataluña': 10.8
    }
}

# Información de vuelos por año
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 3240, 'asientos': 607949, 'pasajeros': 441820},
    2021: {'vuelos': 5002, 'asientos': 891902, 'pasajeros': 693325},
    2022: {'vuelos': 8446, 'asientos': 1532426, 'pasajeros': 1330082},
    2023: {'vuelos': 9352, 'asientos': 1716810, 'pasajeros': 1562253}
}

def normalizar_probabilidades(destinos):
    """Normaliza las probabilidades para que sumen 1"""
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
    base = 8.6 if tipo_organizacion == 'Paquete' else 8.3
    if mes in [7, 8]:  # Alta temporada de verano
        base += uniform(0.3, 0.7)
    elif mes in [6, 9]:  # Temporadas intermedias
        base += uniform(0.2, 0.5)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

def get_temperatura_precipitacion(mes):
    # Temperaturas y precipitaciones adaptadas para destinos españoles populares entre turistas suecos
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
    if porcentaje >= 10:
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
            
            destinos = destinos_principales_por_año[year]
            destinos_norm = normalizar_probabilidades(destinos)
            
            row = {
                'Año': year,
                'mes': mes,
                'Pais': 'Suecia',
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
df.to_csv('turistas_suecia_2019_2023.csv', index=False, encoding='utf-8', decimal=',')