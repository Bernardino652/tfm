import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 564854, 'porc_paquete': 18.0},
   2020: {'total_turistas': 167215, 'porc_paquete': 14.9},
   2021: {'total_turistas': 139937, 'porc_paquete': 2.3},
   2022: {'total_turistas': 347798, 'porc_paquete': 11.3},
   2023: {'total_turistas': 565823, 'porc_paquete': 20.1}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 8.5, 2: 6.2, 3: 8.8, 4: 6.5, 5: 8.2, 6: 11.5,
       7: 9.8, 8: 12.5, 9: 7.2, 10: 5.2, 11: 8.5, 12: 7.1
   },
   2020: {
       1: 30.2, 2: 25.5, 3: 8.5, 4: 0.5, 5: 0.3, 6: 2.5,
       7: 5.8, 8: 9.2, 9: 8.5, 10: 4.8, 11: 2.2, 12: 2.0
   },
   2021: {
       1: 4.8, 2: 2.5, 3: 4.2, 4: 2.8, 5: 3.2, 6: 20.5,
       7: 9.5, 8: 7.2, 9: 5.8, 10: 8.5, 11: 12.5, 12: 15.8
   },
   2022: {
       1: 4.5, 2: 3.8, 3: 5.2, 4: 3.2, 5: 9.8, 6: 8.2,
       7: 10.5, 8: 9.8, 9: 9.2, 10: 15.5, 11: 8.2, 12: 8.8
   },
   2023: {
       1: 8.2, 2: 7.2, 3: 7.8, 4: 8.5, 5: 10.8, 6: 9.8,
       7: 15.2, 8: 8.2, 9: 8.5, 10: 8.2, 11: 4.8, 12: 2.8
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 74.3, 'Carretera': 20.4, 'Puerto': 3.5, 'Tren': 1.8},
   2020: {'Aeropuerto': 58.9, 'Carretera': 32.0, 'Puerto': 6.2, 'Tren': 2.9},
   2021: {'Aeropuerto': 50.6, 'Carretera': 46.8, 'Puerto': 0.2, 'Tren': 2.4},
   2022: {'Aeropuerto': 82.7, 'Carretera': 12.1, 'Puerto': 3.6, 'Tren': 1.7},
   2023: {'Aeropuerto': 77.1, 'Carretera': 14.9, 'Puerto': 6.9, 'Tren': 1.1}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Cataluña': 43.4, 'Comunidad de Madrid': 27.2,
       'Andalucía': 9.3, 'Galicia': 8.3
   },
   2020: {
       'Cataluña': 33.8, 'Comunidad de Madrid': 27.3,
       'Galicia': 23.2, 'Andalucía': 8.9
   },
   2021: {
       'Galicia': 40.0, 'Comunidad de Madrid': 22.8,
       'Cataluña': 16.9, 'Andalucía': 9.5
   },
   2022: {
       'Cataluña': 38.4, 'Comunidad de Madrid': 29.5,
       'Andalucía': 9.5, 'Galicia': 7.1, 'Baleares': 6.1
   },
   2023: {
       'Comunidad de Madrid': 32.7, 'Cataluña': 31.2,
       'Andalucía': 13.9, 'Galicia': 6.9
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 638, 'asientos': 204957, 'pasajeros': 140512},
   2021: {'vuelos': 601, 'asientos': 191626, 'pasajeros': 119507},
   2022: {'vuelos': 1331, 'asientos': 402450, 'pasajeros': 381569},
   2023: {'vuelos': 1544, 'asientos': 500843, 'pasajeros': 484236}
}

# Aeropuertos principales
aeropuertos_principales = {
   2020: [{'nombre': 'Sao Paulo/Guarulhos Int.', 'pasajeros': 102515, 'porcentaje': 72.96}],
   2021: [{'nombre': 'Sao Paulo/Guarulhos Int.', 'pasajeros': 115637, 'porcentaje': 96.76}],
   2022: [{'nombre': 'Sao Paulo/Guarulhos Int.', 'pasajeros': 371874, 'porcentaje': 97.46}],
   2023: [{'nombre': 'Sao Paulo/Guarulhos Int.', 'pasajeros': 402947, 'porcentaje': 83.21}]
}

def normalizar_probabilidades(destinos):
   total = sum(destinos.values())
   return {k: v/total for k, v in destinos.items()}

def generar_satisfaccion(mes, tipo_organizacion):
   base = 8.2 if tipo_organizacion == 'Paquete' else 8.0
   if mes in [6, 7, 8, 9]:
       base += uniform(0.3, 0.8)
   elif mes in [4, 5, 10]:
       base += uniform(0.1, 0.4)
   else:
       base += uniform(-0.2, 0.3)
   return round(base, 2)

def get_temperatura_precipitacion(mes):
   temp_precip = {
       1: (uniform(25.0, 30.0), uniform(200.0, 250.0)),
       2: (uniform(25.0, 30.0), uniform(180.0, 230.0)),
       3: (uniform(24.0, 29.0), uniform(160.0, 210.0)),
       4: (uniform(22.0, 27.0), uniform(120.0, 170.0)),
       5: (uniform(20.0, 25.0), uniform(80.0, 130.0)),
       6: (uniform(18.0, 23.0), uniform(40.0, 90.0)),
       7: (uniform(18.0, 23.0), uniform(30.0, 80.0)),
       8: (uniform(19.0, 24.0), uniform(40.0, 90.0)),
       9: (uniform(20.0, 25.0), uniform(70.0, 120.0)),
       10: (uniform(22.0, 27.0), uniform(100.0, 150.0)),
       11: (uniform(23.0, 28.0), uniform(130.0, 180.0)),
       12: (uniform(24.0, 29.0), uniform(170.0, 220.0))
   }
   return temp_precip[mes]

def get_estacionalidad(porcentaje):
   if porcentaje >= 12:
       return 'Alta'
   elif porcentaje >= 9:
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
       porcentaje_mes = distribucion_mensual[year][mes]
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
               'Pais': 'Brasil',
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
df.to_csv('turistas_brasil_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))