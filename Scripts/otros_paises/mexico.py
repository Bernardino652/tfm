import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 597777, 'porc_paquete': 15.3},
    2020: {'total_turistas': 129785, 'porc_paquete': 12.0},
    2021: {'total_turistas': 249732, 'porc_paquete': 5.4},
    2022: {'total_turistas': 725093, 'porc_paquete': 14.5},
    2023: {'total_turistas': 984259, 'porc_paquete': 17.2}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 93.6, 'Carretera': 1.4, 'Puerto': 4.2, 'Tren': 0.8},
    2020: {'Aeropuerto': 95.3, 'Carretera': 1.9, 'Puerto': 0.8, 'Tren': 2.1},
    2021: {'Aeropuerto': 96.6, 'Carretera': 0.9, 'Puerto': 0.9, 'Tren': 1.6},
    2022: {'Aeropuerto': 94.8, 'Carretera': 0.7, 'Puerto': 3.8, 'Tren': 0.8},
    2023: {'Aeropuerto': 92.8, 'Carretera': 1.1, 'Puerto': 5.4, 'Tren': 0.7}
}

destinos_principales_por_año = {
    2019: {
        'Comunidad de Madrid': 54.5, 'Cataluña': 22.5, 'Andalucía': 5.9
    },
    2020: {
        'Comunidad de Madrid': 56.7, 'Cataluña': 22.2, 'Andalucía': 6.0
    },
    2021: {
        'Comunidad de Madrid': 57.6, 'Cataluña': 19.9, 'Andalucía': 7.2
    },
    2022: {
        'Comunidad de Madrid': 60.0, 'Cataluña': 19.4, 'Andalucía': 6.9
    },
    2023: {
        'Comunidad de Madrid': 67.3, 'Cataluña': 17.4
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 916, 'asientos': 278931, 'pasajeros': 169658},
    2021: {'vuelos': 1526, 'asientos': 479716, 'pasajeros': 347514},
    2022: {'vuelos': 2874, 'asientos': 914525, 'pasajeros': 747756},
    2023: {'vuelos': 3350, 'asientos': 1059984, 'pasajeros': 878661}
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
        1: (uniform(18.0, 22.0), uniform(5.0, 15.0)),
        2: (uniform(19.0, 23.0), uniform(5.0, 15.0)),
        3: (uniform(21.0, 25.0), uniform(10.0, 20.0)),
        4: (uniform(23.0, 27.0), uniform(15.0, 25.0)),
        5: (uniform(24.0, 28.0), uniform(30.0, 40.0)),
        6: (uniform(23.0, 27.0), uniform(70.0, 80.0)),
        7: (uniform(22.0, 26.0), uniform(90.0, 100.0)),
        8: (uniform(22.0, 26.0), uniform(90.0, 100.0)),
        9: (uniform(21.0, 25.0), uniform(80.0, 90.0)),
        10: (uniform(20.0, 24.0), uniform(40.0, 50.0)),
        11: (uniform(19.0, 23.0), uniform(15.0, 25.0)),
        12: (uniform(18.0, 22.0), uniform(5.0, 15.0))
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
                'Pais': 'México',
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
df.to_csv('turistas_mexico_2019_2023.csv', index=False, encoding='utf-8', decimal=',')