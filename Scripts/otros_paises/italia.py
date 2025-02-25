import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 4534515, 'porc_paquete': 16.2},
    2020: {'total_turistas': 947406, 'porc_paquete': 11.9},
    2021: {'total_turistas': 1703423, 'porc_paquete': 8.4},
    2022: {'total_turistas': 4011139, 'porc_paquete': 10.4},
    2023: {'total_turistas': 4849748, 'porc_paquete': 11.6}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 92.5, 'Carretera': 4.7, 'Puerto': 2.7, 'Tren': 0.1},
    2020: {'Aeropuerto': 90.0, 'Carretera': 7.5, 'Puerto': 2.5, 'Tren': 0.0},
    2021: {'Aeropuerto': 93.1, 'Carretera': 5.5, 'Puerto': 1.3, 'Tren': 0.0},
    2022: {'Aeropuerto': 93.8, 'Carretera': 4.4, 'Puerto': 1.8, 'Tren': 0.0},
    2023: {'Aeropuerto': 94.4, 'Carretera': 3.1, 'Puerto': 2.4, 'Tren': 0.0}
}

destinos_principales_por_año = {
    2019: {
        'Cataluña': 28.5, 'Baleares': 16.1, 'Comunidad de Madrid': 14.5,
        'Andalucía': 11.9, 'Comunidad Valenciana': 11.4
    },
    2020: {
        'Cataluña': 20.8, 'Comunidad Valenciana': 15.7, 'Comunidad de Madrid': 14.6,
        'Canarias': 13.8, 'Andalucía': 13.0
    },
    2021: {
        'Cataluña': 24.6, 'Baleares': 17.3, 'Canarias': 15.9,
        'Comunidad Valenciana': 13.6, 'Andalucía': 11.9
    },
    2022: {
        'Cataluña': 25.8, 'Baleares': 18.4, 'Canarias': 15.1,
        'Andalucía': 12.4, 'Comunidad Valenciana': 11.5
    },
    2023: {
        'Cataluña': 26.0, 'Baleares': 16.2, 'Andalucía': 13.9,
        'Comunidad de Madrid': 12.6, 'Canarias': 11.9
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 16058, 'asientos': 2766697, 'pasajeros': 1681839},
    2021: {'vuelos': 25235, 'asientos': 4308146, 'pasajeros': 2991684},
    2022: {'vuelos': 47777, 'asientos': 8537472, 'pasajeros': 7104551},
    2023: {'vuelos': 55826, 'asientos': 10097579, 'pasajeros': 8981807}
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
        1: (uniform(8.0, 12.0), uniform(65.0, 75.0)),
        2: (uniform(9.0, 13.0), uniform(60.0, 70.0)),
        3: (uniform(11.0, 15.0), uniform(55.0, 65.0)),
        4: (uniform(13.0, 17.0), uniform(50.0, 60.0)),
        5: (uniform(17.0, 21.0), uniform(45.0, 55.0)),
        6: (uniform(21.0, 25.0), uniform(35.0, 45.0)),
        7: (uniform(24.0, 28.0), uniform(25.0, 35.0)),
        8: (uniform(24.0, 28.0), uniform(30.0, 40.0)),
        9: (uniform(21.0, 25.0), uniform(45.0, 55.0)),
        10: (uniform(16.0, 20.0), uniform(55.0, 65.0)),
        11: (uniform(12.0, 16.0), uniform(65.0, 75.0)),
        12: (uniform(9.0, 13.0), uniform(70.0, 80.0))
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
                'Pais': 'Italia',
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
df.to_csv('turistas_italia_2019_2023.csv', index=False, encoding='utf-8', decimal=',')