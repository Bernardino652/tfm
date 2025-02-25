import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
    2019: {'total_turistas': 760290, 'porc_paquete': 36.7},
    2020: {'total_turistas': 178370, 'porc_paquete': 46.9},
    2021: {'total_turistas': 175167, 'porc_paquete': 33.0},
    2022: {'total_turistas': 466327, 'porc_paquete': 35.1},
    2023: {'total_turistas': 588550, 'porc_paquete': 37.9}
}

# Medios de acceso por año
medios_acceso_por_año = {
    2019: {'Aeropuerto': 98.5, 'Carretera': 1.2, 'Puerto': 0.1, 'Tren': 0.1},
    2020: {'Aeropuerto': 97.2, 'Carretera': 2.2, 'Puerto': 0.6, 'Tren': 0.1},
    2021: {'Aeropuerto': 91.8, 'Carretera': 6.8, 'Puerto': 1.3, 'Tren': 0.1},
    2022: {'Aeropuerto': 98.7, 'Carretera': 0.5, 'Puerto': 0.7, 'Tren': 0.1},
    2023: {'Aeropuerto': 97.9, 'Carretera': 1.1, 'Puerto': 0.9, 'Tren': 0.1}
}

# Destinos principales por año
destinos_principales_por_año = {
    2019: {
        'Canarias': 26.4, 'Andalucía': 31.5, 'Comunidad Valenciana': 10.5,
        'Cataluña': 8.6, 'Baleares': 7.8
    },
    2020: {
        'Canarias': 58.0, 'Andalucía': 23.2, 'Cataluña': 7.6,
        'Comunidad Valenciana': 7.1
    },
    2021: {
        'Canarias': 33.3, 'Andalucía': 31.0, 'Cataluña': 14.9,
        'Comunidad Valenciana': 10.6
    },
    2022: {
        'Canarias': 34.7, 'Andalucía': 29.5, 'Comunidad Valenciana': 13.6,
        'Cataluña': 12.5, 'Baleares': 5.8
    },
    2023: {
        'Canarias': 35.3, 'Andalucía': 32.6, 'Comunidad Valenciana': 12.6,
        'Cataluña': 11.3
    }
}

# Información de vuelos por año
info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 1206, 'asientos': 234368, 'pasajeros': 190953},
    2021: {'vuelos': 1311, 'asientos': 243081, 'pasajeros': 192620},
    2022: {'vuelos': 3602, 'asientos': 696238, 'pasajeros': 591453},
    2023: {'vuelos': 3997, 'asientos': 784582, 'pasajeros': 678169}
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
        1: (uniform(-5.0, -1.0), uniform(40.0, 50.0)),
        2: (uniform(-4.0, 0.0), uniform(35.0, 45.0)),
        3: (uniform(0.0, 4.0), uniform(30.0, 40.0)),
        4: (uniform(5.0, 10.0), uniform(35.0, 45.0)),
        5: (uniform(10.0, 15.0), uniform(40.0, 50.0)),
        6: (uniform(15.0, 20.0), uniform(50.0, 60.0)),
        7: (uniform(17.0, 22.0), uniform(60.0, 70.0)),
        8: (uniform(16.0, 21.0), uniform(70.0, 80.0)),
        9: (uniform(11.0, 16.0), uniform(60.0, 70.0)),
        10: (uniform(6.0, 11.0), uniform(70.0, 80.0)),
        11: (uniform(1.0, 5.0), uniform(60.0, 70.0)),
        12: (uniform(-3.0, 1.0), uniform(50.0, 60.0))
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

# Lista para almacenar datos
data_rows = []

# Generar datos para todos los años
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
                'Pais': 'Finlandia',
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
df.to_csv('turistas_finlandia_2019_2023.csv', index=False, encoding='utf-8', decimal=',')