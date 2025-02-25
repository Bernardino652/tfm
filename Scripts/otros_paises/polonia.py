import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 1369795, 'porc_paquete': 38.8},
    2020: {'total_turistas': 462425, 'porc_paquete': 46.6},
    2021: {'total_turistas': 823234, 'porc_paquete': 41.8},
    2022: {'total_turistas': 1551252, 'porc_paquete': 35.5},
    2023: {'total_turistas': 1868754, 'porc_paquete': 33.9}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 93.2, 'Carretera': 6.7, 'Puerto': 0.1, 'Tren': 0.0},
    2020: {'Aeropuerto': 94.3, 'Carretera': 5.6, 'Puerto': 0.1, 'Tren': 0.0},
    2021: {'Aeropuerto': 97.8, 'Carretera': 2.1, 'Puerto': 0.1, 'Tren': 0.1},
    2022: {'Aeropuerto': 95.8, 'Carretera': 3.9, 'Puerto': 0.2, 'Tren': 0.0},
    2023: {'Aeropuerto': 97.8, 'Carretera': 1.5, 'Puerto': 0.7, 'Tren': 0.0}
}

destinos_principales_por_año = {
    2019: {
        'Canarias': 27.4, 'Cataluña': 22.7, 'Andalucía': 18.5,
        'Baleares': 11.3, 'Comunidad Valenciana': 8.4
    },
    2020: {
        'Canarias': 37.8, 'Baleares': 16.9, 'Andalucía': 16.7,
        'Cataluña': 11.2, 'Comunidad Valenciana': 9.8
    },
    2021: {
        'Canarias': 32.9, 'Andalucía': 16.9, 'Baleares': 16.1,
        'Cataluña': 15.3, 'Comunidad Valenciana': 10.3
    },
    2022: {
        'Canarias': 27.7, 'Cataluña': 19.6, 'Baleares': 16.6,
        'Andalucía': 12.4, 'Comunidad Valenciana': 11.1
    },
    2023: {
        'Canarias': 25.3, 'Cataluña': 21.8, 'Andalucía': 17.2,
        'Comunidad Valenciana': 15.2, 'Baleares': 12.4
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 2504, 'asientos': 449084, 'pasajeros': 314625},
    2021: {'vuelos': 4318, 'asientos': 785885, 'pasajeros': 657695},
    2022: {'vuelos': 6984, 'asientos': 1330964, 'pasajeros': 1220950},
    2023: {'vuelos': 9782, 'asientos': 1936637, 'pasajeros': 1801263}
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
        1: (uniform(-4.0, 0.0), uniform(30.0, 40.0)),
        2: (uniform(-3.0, 1.0), uniform(25.0, 35.0)),
        3: (uniform(1.0, 5.0), uniform(30.0, 40.0)),
        4: (uniform(7.0, 11.0), uniform(35.0, 45.0)),
        5: (uniform(12.0, 16.0), uniform(45.0, 55.0)),
        6: (uniform(15.0, 19.0), uniform(65.0, 75.0)),
        7: (uniform(17.0, 21.0), uniform(75.0, 85.0)),
        8: (uniform(17.0, 21.0), uniform(65.0, 75.0)),
        9: (uniform(13.0, 17.0), uniform(50.0, 60.0)),
        10: (uniform(8.0, 12.0), uniform(35.0, 45.0)),
        11: (uniform(3.0, 7.0), uniform(35.0, 45.0)),
        12: (uniform(-2.0, 2.0), uniform(30.0, 40.0))
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
                'Pais': 'Polonia',
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
df.to_csv('turistas_polonia_2019_2023.csv', index=False, encoding='utf-8', decimal=',')