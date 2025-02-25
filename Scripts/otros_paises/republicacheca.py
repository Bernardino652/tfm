import pandas as pd
import numpy as np
from random import uniform, choice

def normalizar_probabilidades(destinos):
    """Normaliza las probabilidades para que sumen 1"""
    total = sum(destinos.values())
    return {k: v/total for k, v in destinos.items()}

def get_temperatura_precipitacion(mes):
    """Genera temperaturas y precipitaciones para cada mes"""
    temp_precip = {
        1: (uniform(10.0, 14.0), uniform(30.0, 40.0)),    
        2: (uniform(11.0, 15.0), uniform(30.0, 40.0)),   
        3: (uniform(13.0, 17.0), uniform(30.0, 40.0)),  
        4: (uniform(15.0, 19.0), uniform(40.0, 50.0)),  
        5: (uniform(18.0, 22.0), uniform(40.0, 50.0)),  
        6: (uniform(22.0, 26.0), uniform(20.0, 30.0)),  
        7: (uniform(25.0, 29.0), uniform(10.0, 20.0)),  
        8: (uniform(25.0, 29.0), uniform(10.0, 20.0)),  
        9: (uniform(22.0, 26.0), uniform(30.0, 40.0)),  
        10: (uniform(18.0, 22.0), uniform(40.0, 50.0)), 
        11: (uniform(14.0, 18.0), uniform(40.0, 50.0)), 
        12: (uniform(11.0, 15.0), uniform(30.0, 40.0))    
    }
    return temp_precip[mes]

def get_estacionalidad(porcentaje):
    """Determina el nivel de estacionalidad basado en el porcentaje"""
    if porcentaje >= 11:
        return 'Alta'
    elif porcentaje >= 9:
        return 'Media-Alta'
    elif porcentaje >= 7:
        return 'Media'
    else:
        return 'Baja'

def generar_satisfaccion(mes, tipo_organizacion, base_paquete=8.5, base_individual=8.2):
    """Genera el nivel de satisfacción basado en el mes y tipo de organización"""
    base = base_paquete if tipo_organizacion == 'Paquete' else base_individual
    if mes in [7, 8]:  # Alta temporada de verano
        base += uniform(0.3, 0.7)
    elif mes in [6, 9]:  # Temporadas intermedias
        base += uniform(0.2, 0.5)
    else:
        base += uniform(-0.2, 0.3)
    return round(base, 2)

# Datos de República Checa
datos_anuales = {
    2019: {'total_turistas': 459727, 'porc_paquete': 27.6},
    2020: {'total_turistas': 128260, 'porc_paquete': 13.7},
    2021: {'total_turistas': 259896, 'porc_paquete': 30.7},
    2022: {'total_turistas': 509093, 'porc_paquete': 31.6},
    2023: {'total_turistas': 620316, 'porc_paquete': 30.1}
}

distribucion_mensual = {
    2019: {1: 4.2, 2: 4.5, 3: 6.8, 4: 8.5, 5: 10.2, 6: 12.8, 7: 15.5, 8: 14.8, 9: 9.8, 10: 6.5, 11: 3.5, 12: 2.9},
    2020: {1: 15.5, 2: 14.8, 3: 8.2, 4: 2.1, 5: 3.5, 6: 8.2, 7: 12.5, 8: 13.2, 9: 8.5, 10: 6.2, 11: 4.2, 12: 3.1},
    2021: {1: 3.2, 2: 3.5, 3: 4.8, 4: 7.2, 5: 9.5, 6: 12.8, 7: 15.2, 8: 14.5, 9: 11.8, 10: 8.2, 11: 5.2, 12: 4.1},
    2022: {1: 3.8, 2: 4.2, 3: 6.5, 4: 8.8, 5: 10.5, 6: 13.2, 7: 14.8, 8: 14.2, 9: 10.5, 10: 7.2, 11: 3.8, 12: 2.5},
    2023: {1: 4.0, 2: 4.5, 3: 7.2, 4: 9.2, 5: 11.5, 6: 13.5, 7: 14.8, 8: 14.0, 9: 9.8, 10: 6.5, 11: 3.2, 12: 1.8}
}

medios_acceso = {
    2019: {'Aeropuerto': 97.0, 'Carretera': 2.7, 'Tren': 0.1, 'Puerto': 0.2},
    2020: {'Aeropuerto': 98.0, 'Carretera': 2.0, 'Tren': 0.0, 'Puerto': 0.0},
    2021: {'Aeropuerto': 97.2, 'Carretera': 2.7, 'Tren': 0.0, 'Puerto': 0.0},
    2022: {'Aeropuerto': 97.9, 'Carretera': 1.7, 'Tren': 0.1, 'Puerto': 0.3},
    2023: {'Aeropuerto': 97.3, 'Carretera': 2.0, 'Tren': 0.0, 'Puerto': 0.6}
}

destinos_principales = {
    2019: {'Cataluña': 29.4, 'Canarias': 20.7, 'Baleares': 18.2, 'Andalucía': 10.7, 'Comunidad de Madrid': 7.8},
    2020: {'Cataluña': 27.1, 'Canarias': 22.1, 'Comunidad de Madrid': 14.5, 'Baleares': 13.0, 'Andalucía': 7.3},
    2021: {'Canarias': 39.5, 'Baleares': 26.4, 'Cataluña': 13.5, 'Andalucía': 6.0, 'Comunidad Valenciana': 5.5},
    2022: {'Canarias': 27.9, 'Cataluña': 23.0, 'Baleares': 21.7, 'Andalucía': 8.5, 'Comunidad de Madrid': 7.3},
    2023: {'Canarias': 28.0, 'Cataluña': 18.1, 'Baleares': 17.2, 'Andalucía': 11.0, 'Comunidad Valenciana': 9.3}
}

info_vuelos = {
    2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
    2020: {'vuelos': 1188, 'asientos': 200292, 'pasajeros': 127823},
    2021: {'vuelos': 1949, 'asientos': 310647, 'pasajeros': 230645},
    2022: {'vuelos': 3492, 'asientos': 583058, 'pasajeros': 488770},
    2023: {'vuelos': 4258, 'asientos': 735746, 'pasajeros': 623450}
}

# Crear lista para almacenar los datos
data_rows = []

# Generar datos para cada año
for year in datos_anuales.keys():
    total_turistas = datos_anuales[year]['total_turistas']
    porc_paquete = datos_anuales[year]['porc_paquete']
    
    for mes in range(1, 13):
        porcentaje_mes = distribucion_mensual[year][mes]
        turistas_mes = round(total_turistas * porcentaje_mes / 100)
        
        for medio, porcentaje in medios_acceso[year].items():
            turistas_medio = round(turistas_mes * porcentaje / 100)
            tipo_org = 'Paquete' if uniform(0, 100) < porc_paquete else 'Individual'
            temp, precip = get_temperatura_precipitacion(mes)
            
            destinos = destinos_principales[year]
            destinos_norm = normalizar_probabilidades(destinos)
            
            row = {
                'Año': year,
                'mes': mes,
                'Pais': 'República Checa',
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
df.to_csv('turistas_republica_checa_2019_2023.csv', index=False, encoding='utf-8', decimal=',')