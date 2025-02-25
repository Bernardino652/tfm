import pandas as pd
import numpy as np
from random import uniform, choice

datos_anuales = {
    2019: {'total_turistas': 18012484, 'porc_paquete': 41.7},
    2020: {'total_turistas': 3150204, 'porc_paquete': 29.8},
    2021: {'total_turistas': 4302634, 'porc_paquete': 31.2},
    2022: {'total_turistas': 15121910, 'porc_paquete': 40.5},
    2023: {'total_turistas': 17262287, 'porc_paquete': 42.9}
}

medios_acceso_por_año = {
    2019: {'Aeropuerto': 96.5, 'Carretera': 2.5, 'Puerto': 0.8, 'Tren': 0.1},
    2020: {'Aeropuerto': 93.8, 'Carretera': 4.7, 'Puerto': 1.4, 'Tren': 0.1},
    2021: {'Aeropuerto': 96.1, 'Carretera': 2.2, 'Puerto': 1.6, 'Tren': 0.1},
    2022: {'Aeropuerto': 97.8, 'Carretera': 0.9, 'Puerto': 1.2, 'Tren': 0.1},
    2023: {'Aeropuerto': 97.6, 'Carretera': 1.2, 'Puerto': 1.1, 'Tren': 0.1}
}

destinos_principales_por_año = {
    2019: {
        'Canarias': 27.1, 'Baleares': 20.5, 'Andalucía': 16.7,
        'Comunidad Valenciana': 15.8, 'Cataluña': 11.2
    },
    2020: {
        'Canarias': 36.3, 'Andalucía': 18.5, 'Comunidad Valenciana': 18.5,
        'Cataluña': 8.9, 'Baleares': 7.1
    },
    2021: {
        'Canarias': 29.0, 'Baleares': 24.0, 'Andalucía': 16.3,
        'Comunidad Valenciana': 15.5, 'Cataluña': 6.6
    },
    2022: {
        'Canarias': 30.9, 'Baleares': 22.2, 'Andalucía': 15.8,
        'Comunidad Valenciana': 14.6, 'Cataluña': 9.6
    },
    2023: {
        'Canarias': 31.1, 'Baleares': 21.4, 'Comunidad Valenciana': 15.4,
        'Andalucía': 15.1, 'Cataluña': 10.4
    }
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 38408, 'asientos': 6934337, 'pasajeros': 3990718},
    2021: {'vuelos': 47983, 'asientos': 8567550, 'pasajeros': 4982455},
    2022: {'vuelos': 120582, 'asientos': 21949980, 'pasajeros': 18554582},
    2023: {'vuelos': 131048, 'asientos': 23968193, 'pasajeros': 21407608}
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
        4: (uniform(8.0, 12.0), uniform(45.0, 55.0)),
        5: (uniform(11.0, 15.0), uniform(45.0, 55.0)),
        6: (uniform(14.0, 18.0), uniform(40.0, 50.0)),
        7: (uniform(16.0, 20.0), uniform(45.0, 55.0)),
        8: (uniform(16.0, 20.0), uniform(45.0, 55.0)),
        9: (uniform(14.0, 18.0), uniform(50.0, 60.0)),
        10: (uniform(11.0, 15.0), uniform(60.0, 70.0)),
        11: (uniform(7.0, 11.0), uniform(70.0, 80.0)),
        12: (uniform(5.0, 9.0), uniform(70.0, 80.0))
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
                'Pais': 'Reino Unido',
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
df.to_csv('turistas_reinoUnido_2019_2023.csv', index=False, encoding='utf-8', decimal=',')