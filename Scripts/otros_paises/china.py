import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 700748, 'porc_paquete': 43.9},
   2020: {'total_turistas': 134612, 'porc_paquete': 46.4},
   2021: {'total_turistas': 29131, 'porc_paquete': 3.1},
   2022: {'total_turistas': 56646, 'porc_paquete': 11.1},
   2023: {'total_turistas': 388515, 'porc_paquete': 15.0}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 5.2, 2: 7.8, 3: 6.5, 4: 7.2, 5: 8.5, 6: 11.8,
       7: 8.2, 8: 11.5, 9: 11.2, 10: 9.8, 11: 7.5, 12: 4.8
   },
   2020: {
       1: 35.2, 2: 18.5, 3: 5.2, 4: 0.5, 5: 0.3, 6: 2.5,
       7: 12.5, 8: 8.2, 9: 20.5, 10: 4.8, 11: 3.5, 12: 3.2
   },
   2021: {
       1: 2.5, 2: 3.2, 3: 5.2, 4: 6.5, 5: 5.8, 6: 6.2,
       7: 8.5, 8: 30.2, 9: 18.5, 10: 15.2, 11: 6.8, 12: 3.5
   },
   2022: {
       1: 4.2, 2: 4.5, 3: 3.8, 4: 5.2, 5: 7.2, 6: 10.8,
       7: 15.5, 8: 9.5, 9: 9.8, 10: 9.5, 11: 3.2, 12: 11.5
   },
   2023: {
       1: 1.5, 2: 6.2, 3: 6.5, 4: 4.8, 5: 9.2, 6: 11.2,
       7: 10.5, 8: 8.5, 9: 16.8, 10: 8.8, 11: 8.5, 12: 7.5
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 86.9, 'Carretera': 9.3, 'Puerto': 3.1, 'Tren': 0.7},
   2020: {'Aeropuerto': 68.8, 'Carretera': 27.6, 'Puerto': 2.1, 'Tren': 1.5},
   2021: {'Aeropuerto': 89.9, 'Carretera': 1.8, 'Puerto': 0.0, 'Tren': 8.4},
   2022: {'Aeropuerto': 90.4, 'Carretera': 5.7, 'Puerto': 3.5, 'Tren': 0.3},
   2023: {'Aeropuerto': 92.7, 'Carretera': 0.2, 'Puerto': 6.5, 'Tren': 0.7}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Cataluña': 50.2,
       'Comunidad de Madrid': 29.9,
       'Andalucía': 10.3
   },
   2020: {
       'Cataluña': 43.8,
       'Comunidad de Madrid': 22.5,
       'Andalucía': 19.2,
       'Comunidad Valenciana': 10.4
   },
   2021: {
       'Comunidad de Madrid': 34.0,
       'Cataluña': 24.1,
       'Baleares': 17.7,
       'Andalucía': 9.1
   },
   2022: {
       'Cataluña': 43.4,
       'Comunidad de Madrid': 30.1,
       'Baleares': 8.0,
       'Andalucía': 6.7
   },
   2023: {
       'Cataluña': 49.1,
       'Comunidad de Madrid': 29.4,
       'Andalucía': 6.9,
       'Comunidad Valenciana': 6.0
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2021: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2022: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2023: {'vuelos': 761, 'asientos': 224048, 'pasajeros': 172684}
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
       1: (uniform(2.0, 6.0), uniform(20.0, 40.0)),
       2: (uniform(4.0, 8.0), uniform(25.0, 45.0)),
       3: (uniform(8.0, 12.0), uniform(30.0, 50.0)),
       4: (uniform(14.0, 18.0), uniform(40.0, 60.0)),
       5: (uniform(18.0, 22.0), uniform(60.0, 80.0)),
       6: (uniform(22.0, 26.0), uniform(80.0, 100.0)),
       7: (uniform(24.0, 28.0), uniform(100.0, 120.0)),
       8: (uniform(24.0, 28.0), uniform(90.0, 110.0)),
       9: (uniform(20.0, 24.0), uniform(70.0, 90.0)),
       10: (uniform(16.0, 20.0), uniform(50.0, 70.0)),
       11: (uniform(10.0, 14.0), uniform(30.0, 50.0)),
       12: (uniform(4.0, 8.0), uniform(20.0, 40.0))
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
               'Pais': 'China',
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
df.to_csv('turistas_china_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))