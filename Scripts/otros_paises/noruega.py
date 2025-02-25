import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 1480843, 'porc_paquete': 30.1},
    2020: {'total_turistas': 261345, 'porc_paquete': 39.3},
    2021: {'total_turistas': 293731, 'porc_paquete': 19.4},
    2022: {'total_turistas': 1253752, 'porc_paquete': 24.6},
    2023: {'total_turistas': 1354610, 'porc_paquete': 28.2}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 97.5, 'Carretera': 1.8, 'Puerto': 0.6, 'Tren': 0.1},
    2020: {'Aeropuerto': 99.0, 'Carretera': 0.9, 'Puerto': 0.0, 'Tren': 0.0},
    2021: {'Aeropuerto': 97.1, 'Carretera': 2.1, 'Puerto': 0.7, 'Tren': 0.0},
    2022: {'Aeropuerto': 96.2, 'Carretera': 2.2, 'Puerto': 1.5, 'Tren': 0.1},
    2023: {'Aeropuerto': 95.9, 'Carretera': 3.1, 'Puerto': 1.0, 'Tren': 0.0}
}

destinos_principales_por_año = {
    2019: {
        'Canarias': 33.5, 'Comunidad Valenciana': 22.8, 'Andalucía': 16.4,
        'Baleares': 10.3, 'Cataluña': 8.7
    },
    2020: {
        'Canarias': 52.4, 'Comunidad Valenciana': 21.1, 'Andalucía': 13.4
    },
    2021: {
        'Canarias': 32.7, 'Comunidad Valenciana': 29.5, 'Andalucía': 19.9,
        'Cataluña': 7.5, 'Baleares': 5.0
    },
    2022: {
        'Comunidad Valenciana': 29.2, 'Canarias': 22.2, 'Andalucía': 16.4,
        'Cataluña': 13.4, 'Baleares': 9.1
    },
    2023: {
        'Canarias': 30.4, 'Comunidad Valenciana': 25.6, 'Andalucía': 17.3,
        'Cataluña': 10.5, 'Baleares': 9.3
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 2044, 'asientos': 392173, 'pasajeros': 258026},
    2021: {'vuelos': 2478, 'asientos': 449679, 'pasajeros': 309673},
    2022: {'vuelos': 8418, 'asientos': 1570549, 'pasajeros': 1302942},
    2023: {'vuelos': 8332, 'asientos': 1557822, 'pasajeros': 1375929}
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
        1: (uniform(-5.0, 0.0), uniform(70.0, 80.0)),
        2: (uniform(-4.0, 1.0), uniform(65.0, 75.0)),
        3: (uniform(-2.0, 3.0), uniform(60.0, 70.0)),
        4: (uniform(2.0, 7.0), uniform(50.0, 60.0)),
        5: (uniform(7.0, 12.0), uniform(45.0, 55.0)),
        6: (uniform(12.0, 17.0), uniform(40.0, 50.0)),
        7: (uniform(14.0, 19.0), uniform(45.0, 55.0)),
        8: (uniform(14.0, 19.0), uniform(50.0, 60.0)),
        9: (uniform(10.0, 15.0), uniform(55.0, 65.0)),
        10: (uniform(5.0, 10.0), uniform(65.0, 75.0)),
        11: (uniform(0.0, 5.0), uniform(70.0, 80.0)),
        12: (uniform(-4.0, 1.0), uniform(75.0, 85.0))
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
                'Pais': 'Noruega',
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
df.to_csv('turistas_noruega_2019_2023.csv', index=False, encoding='utf-8', decimal=',')