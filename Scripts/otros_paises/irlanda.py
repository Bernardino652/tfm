import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 2177592, 'porc_paquete': 24.2},
    2020: {'total_turistas': 329043, 'porc_paquete': 22.1},
    2021: {'total_turistas': 631314, 'porc_paquete': 14.3},
    2022: {'total_turistas': 2089464, 'porc_paquete': 22.0},
    2023: {'total_turistas': 2476105, 'porc_paquete': 29.2}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 98.2, 'Carretera': 1.0, 'Puerto': 0.8, 'Tren': 0.0},
    2020: {'Aeropuerto': 98.4, 'Carretera': 0.7, 'Puerto': 0.9, 'Tren': 0.0},
    2021: {'Aeropuerto': 96.1, 'Carretera': 1.8, 'Puerto': 2.1, 'Tren': 0.1},
    2022: {'Aeropuerto': 96.7, 'Carretera': 1.2, 'Puerto': 2.0, 'Tren': 0.1},
    2023: {'Aeropuerto': 96.9, 'Carretera': 1.3, 'Puerto': 1.8, 'Tren': 0.0}
}

destinos_principales_por_año = {
    2019: {
        'Canarias': 29.1, 'Andalucía': 24.8, 'Cataluña': 16.8,
        'Comunidad Valenciana': 10.2, 'Baleares': 7.9
    },
    2020: {
        'Canarias': 43.1, 'Andalucía': 22.8, 'Comunidad Valenciana': 9.4,
        'Cataluña': 8.5, 'Comunidad de Madrid': 6.2
    },
    2021: {
        'Canarias': 31.8, 'Andalucía': 25.2, 'Cataluña': 14.2,
        'Comunidad Valenciana': 11.7, 'Baleares': 6.7
    },
    2022: {
        'Canarias': 29.9, 'Andalucía': 21.9, 'Cataluña': 16.7,
        'Baleares': 12.5, 'Comunidad Valenciana': 10.4
    },
    2023: {
        'Canarias': 29.3, 'Andalucía': 24.4, 'Cataluña': 14.9,
        'Comunidad Valenciana': 11.9, 'Baleares': 10.1
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 4184, 'asientos': 772832, 'pasajeros': 389059},
    2021: {'vuelos': 5240, 'asientos': 965780, 'pasajeros': 675117},
    2022: {'vuelos': 13889, 'asientos': 2594988, 'pasajeros': 2278614},
    2023: {'vuelos': 16177, 'asientos': 3032958, 'pasajeros': 2737315}
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
        1: (uniform(4.0, 8.0), uniform(70.0, 80.0)),
        2: (uniform(4.0, 8.0), uniform(60.0, 70.0)),
        3: (uniform(6.0, 10.0), uniform(50.0, 60.0)),
        4: (uniform(7.0, 12.0), uniform(45.0, 55.0)),
        5: (uniform(10.0, 15.0), uniform(40.0, 50.0)),
        6: (uniform(13.0, 18.0), uniform(35.0, 45.0)),
        7: (uniform(15.0, 20.0), uniform(40.0, 50.0)),
        8: (uniform(15.0, 20.0), uniform(45.0, 55.0)),
        9: (uniform(13.0, 18.0), uniform(50.0, 60.0)),
        10: (uniform(10.0, 15.0), uniform(60.0, 70.0)),
        11: (uniform(6.0, 11.0), uniform(65.0, 75.0)),
        12: (uniform(4.0, 9.0), uniform(70.0, 80.0))
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
                'Pais': 'Irlanda',
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
df.to_csv('turistas_irlanda_2019_2023.csv', index=False, encoding='utf-8', decimal=',')