import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 3684260, 'porc_paquete': 27.7},
    2020: {'total_turistas': 918361, 'porc_paquete': 24.5},
    2021: {'total_turistas': 2048853, 'porc_paquete': 21.8},
    2022: {'total_turistas': 3923089, 'porc_paquete': 26.7},
    2023: {'total_turistas': 4320630, 'porc_paquete': 26.6}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 88.9, 'Carretera': 10.7, 'Puerto': 0.3, 'Tren': 0.1},
    2020: {'Aeropuerto': 85.9, 'Carretera': 13.9, 'Puerto': 0.1, 'Tren': 0.1},
    2021: {'Aeropuerto': 86.5, 'Carretera': 13.3, 'Puerto': 0.1, 'Tren': 0.1},
    2022: {'Aeropuerto': 88.4, 'Carretera': 10.7, 'Puerto': 0.7, 'Tren': 0.2},
    2023: {'Aeropuerto': 88.2, 'Carretera': 10.6, 'Puerto': 1.0, 'Tren': 0.2}
}

destinos_principales_por_año = {
    2019: {
        'Cataluña': 20.3, 'Baleares': 18.0, 'Andalucía': 16.7,
        'Canarias': 16.5, 'Comunidad Valenciana': 16.0
    },
    2020: {
        'Comunidad Valenciana': 21.2, 'Andalucía': 21.2, 'Canarias': 20.4,
        'Cataluña': 15.5, 'Baleares': 10.1
    },
    2021: {
        'Baleares': 20.9, 'Comunidad Valenciana': 18.5, 'Cataluña': 17.5,
        'Andalucía': 17.1, 'Canarias': 16.5
    },
    2022: {
        'Canarias': 19.3, 'Comunidad Valenciana': 18.3, 'Andalucía': 17.7,
        'Cataluña': 17.0, 'Baleares': 16.2
    },
    2023: {
        'Andalucía': 18.7, 'Comunidad Valenciana': 18.0, 'Cataluña': 17.4,
        'Baleares': 16.6, 'Canarias': 16.5
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 11526, 'asientos': 2008487, 'pasajeros': 1199889},
    2021: {'vuelos': 18586, 'asientos': 3157199, 'pasajeros': 2343982},
    2022: {'vuelos': 27897, 'asientos': 4904288, 'pasajeros': 4168769},
    2023: {'vuelos': 29574, 'asientos': 5296986, 'pasajeros': 4651941}
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
        1: (uniform(2.0, 6.0), uniform(65.0, 75.0)),
        2: (uniform(3.0, 7.0), uniform(60.0, 70.0)),
        3: (uniform(5.0, 9.0), uniform(55.0, 65.0)),
        4: (uniform(8.0, 12.0), uniform(50.0, 60.0)),
        5: (uniform(11.0, 15.0), uniform(45.0, 55.0)),
        6: (uniform(14.0, 18.0), uniform(40.0, 50.0)),
        7: (uniform(16.0, 20.0), uniform(45.0, 55.0)),
        8: (uniform(16.0, 20.0), uniform(50.0, 60.0)),
        9: (uniform(13.0, 17.0), uniform(55.0, 65.0)),
        10: (uniform(10.0, 14.0), uniform(60.0, 70.0)),
        11: (uniform(6.0, 10.0), uniform(65.0, 75.0)),
        12: (uniform(3.0, 7.0), uniform(70.0, 80.0))
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
                'Pais': 'Países Bajos',
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
df.to_csv('turistas_paisesBajos_2019_2023.csv', index=False, encoding='utf-8', decimal=',')