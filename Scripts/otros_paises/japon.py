import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 677659, 'porc_paquete': 42.2},
    2020: {'total_turistas': 112916, 'porc_paquete': 49.1},
    2021: {'total_turistas': 29368, 'porc_paquete': 26.8},
    2022: {'total_turistas': 124290, 'porc_paquete': 9.5},
    2023: {'total_turistas': 310532, 'porc_paquete': 20.6}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 84.5, 'Carretera': 13.1, 'Puerto': 2.0, 'Tren': 0.5},
    2020: {'Aeropuerto': 89.0, 'Carretera': 8.1, 'Puerto': 1.4, 'Tren': 1.6},
    2021: {'Aeropuerto': 87.2, 'Carretera': 8.8, 'Puerto': 0.0, 'Tren': 3.9},
    2022: {'Aeropuerto': 83.6, 'Carretera': 15.3, 'Puerto': 0.9, 'Tren': 0.2},
    2023: {'Aeropuerto': 97.5, 'Carretera': 0.0, 'Puerto': 2.1, 'Tren': 0.5}
}

destinos_principales_por_año = {
    2019: {
        'Cataluña': 54.1, 'Comunidad de Madrid': 20.2, 'Andalucía': 11.2
    },
    2020: {
        'Cataluña': 55.2, 'Comunidad de Madrid': 26.2, 'Andalucía': 7.6
    },
    2021: {
        'Cataluña': 41.8, 'Comunidad de Madrid': 32.8, 'Comunidad Valenciana': 9.6,
        'Andalucía': 5.2
    },
    2022: {
        'Cataluña': 47.3, 'Comunidad Valenciana': 19.2, 'Comunidad de Madrid': 15.8,
        'Andalucía': 7.7
    },
    2023: {
        'Cataluña': 58.4, 'Comunidad de Madrid': 22.4, 'Andalucía': 6.8
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
        1: (uniform(5.0, 9.0), uniform(50.0, 60.0)),
        2: (uniform(6.0, 10.0), uniform(45.0, 55.0)),
        3: (uniform(8.0, 13.0), uniform(40.0, 50.0)),
        4: (uniform(12.0, 17.0), uniform(35.0, 45.0)),
        5: (uniform(17.0, 22.0), uniform(30.0, 40.0)),
        6: (uniform(20.0, 25.0), uniform(25.0, 35.0)),
        7: (uniform(24.0, 29.0), uniform(20.0, 30.0)),
        8: (uniform(25.0, 30.0), uniform(20.0, 30.0)),
        9: (uniform(21.0, 26.0), uniform(30.0, 40.0)),
        10: (uniform(16.0, 21.0), uniform(40.0, 50.0)),
        11: (uniform(11.0, 16.0), uniform(45.0, 55.0)),
        12: (uniform(6.0, 11.0), uniform(50.0, 60.0))
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
                'Pais': 'Japón',
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
df.to_csv('turistas_japon_2019_2023.csv', index=False, encoding='utf-8', decimal=',')