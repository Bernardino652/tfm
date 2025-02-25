import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 2428790, 'porc_paquete': 14.8},
    2020: {'total_turistas': 762384, 'porc_paquete': 6.4},
    2021: {'total_turistas': 1193649, 'porc_paquete': 6.1},
    2022: {'total_turistas': 2415936, 'porc_paquete': 10.8},
    2023: {'total_turistas': 2802774, 'porc_paquete': 11.0}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 17.2, 'Carretera': 82.0, 'Puerto': 0.3, 'Tren': 0.5},
    2020: {'Aeropuerto': 13.3, 'Carretera': 86.3, 'Puerto': 0.2, 'Tren': 0.2},
    2021: {'Aeropuerto': 20.6, 'Carretera': 78.8, 'Puerto': 0.4, 'Tren': 0.1},
    2022: {'Aeropuerto': 23.3, 'Carretera': 75.8, 'Puerto': 0.7, 'Tren': 0.2},
    2023: {'Aeropuerto': 26.2, 'Carretera': 72.4, 'Puerto': 1.1, 'Tren': 0.3}
}

destinos_principales_por_año = {
    2019: {
        'Galicia': 23.5, 'Andalucía': 18.4, 'Comunidad de Madrid': 12.3,
        'Extremadura': 11.7, 'Castilla y León': 8.9
    },
    2020: {
        'Galicia': 28.7, 'Andalucía': 20.8, 'Extremadura': 11.4,
        'Comunidad de Madrid': 11.1, 'Castilla y León': 9.3
    },
    2021: {
        'Galicia': 31.5, 'Andalucía': 18.6, 'Extremadura': 12.2,
        'Comunidad de Madrid': 9.2, 'Cataluña': 6.1
    },
    2022: {
        'Galicia': 26.9, 'Andalucía': 20.6, 'Comunidad de Madrid': 8.4,
        'Extremadura': 7.5, 'Castilla y León': 7.3
    },
    2023: {
        'Galicia': 25.6, 'Andalucía': 22.8, 'Cataluña': 10.5,
        'Comunidad de Madrid': 8.9, 'Baleares': 5.7
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 7992, 'asientos': 1098710, 'pasajeros': 715188},
    2021: {'vuelos': 10540, 'asientos': 1461389, 'pasajeros': 945150},
    2022: {'vuelos': 22431, 'asientos': 3407070, 'pasajeros': 2662110},
    2023: {'vuelos': 26207, 'asientos': 4279174, 'pasajeros': 3438909}
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
        1: (uniform(8.0, 12.0), uniform(100.0, 120.0)),
        2: (uniform(9.0, 13.0), uniform(90.0, 110.0)),
        3: (uniform(11.0, 15.0), uniform(70.0, 90.0)),
        4: (uniform(12.0, 16.0), uniform(60.0, 80.0)),
        5: (uniform(14.0, 18.0), uniform(50.0, 70.0)),
        6: (uniform(17.0, 21.0), uniform(30.0, 50.0)),
        7: (uniform(19.0, 23.0), uniform(10.0, 30.0)),
        8: (uniform(19.0, 23.0), uniform(10.0, 30.0)),
        9: (uniform(18.0, 22.0), uniform(30.0, 50.0)),
        10: (uniform(14.0, 18.0), uniform(60.0, 80.0)),
        11: (uniform(11.0, 15.0), uniform(90.0, 110.0)),
        12: (uniform(9.0, 13.0), uniform(100.0, 120.0))
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
                'Pais': 'Portugal',
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
df.to_csv('turistas_portugal_2019_2023.csv', index=False, encoding='utf-8', decimal=',')