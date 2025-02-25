import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 630797, 'porc_paquete': 47.0},
   2020: {'total_turistas': 121931, 'porc_paquete': 40.6},
   2021: {'total_turistas': 37692, 'porc_paquete': 33.3},
   2022: {'total_turistas': 180290, 'porc_paquete': 22.7},
   2023: {'total_turistas': 434583, 'porc_paquete': 30.6}
}

# Distribución mensual por año 
distribucion_mensual = {
   2019: {
       1: 8.5, 2: 7.8, 3: 8.2, 4: 8.8, 5: 9.5, 6: 10.2,
       7: 6.8, 8: 7.2, 9: 7.5, 10: 7.2, 11: 8.5, 12: 9.8
   },
   2020: {
       1: 45.2, 2: 35.5, 3: 8.2, 4: 0.5, 5: 0.3, 6: 1.2,
       7: 2.5, 8: 2.8, 9: 1.5, 10: 1.2, 11: 0.8, 12: 0.3
   },
   2021: {
       1: 3.5, 2: 4.8, 3: 5.2, 4: 4.5, 5: 6.8, 6: 7.5,
       7: 16.8, 8: 11.2, 9: 6.5, 10: 7.2, 11: 8.5, 12: 31.2
   },
   2022: {
       1: 1.2, 2: 2.5, 3: 3.2, 4: 4.5, 5: 6.8, 6: 7.5,
       7: 11.2, 8: 5.8, 9: 10.2, 10: 11.5, 11: 12.8, 12: 17.5
   },
   2023: {
       1: 2.5, 2: 4.8, 3: 5.2, 4: 15.5, 5: 10.2, 6: 7.5,
       7: 8.2, 8: 4.2, 9: 6.5, 10: 16.8, 11: 8.5, 12: 10.1
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 83.4, 'Carretera': 14.9, 'Puerto': 1.1, 'Tren': 0.7},
   2020: {'Aeropuerto': 91.9, 'Carretera': 5.3, 'Puerto': 0.0, 'Tren': 2.8},
   2021: {'Aeropuerto': 81.2, 'Carretera': 0.0, 'Puerto': 0.2, 'Tren': 18.6},
   2022: {'Aeropuerto': 99.2, 'Carretera': 0.1, 'Puerto': 0.0, 'Tren': 0.7},
   2023: {'Aeropuerto': 99.3, 'Carretera': 0.1, 'Puerto': 0.1, 'Tren': 0.4}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Cataluña': 46.8,
       'Comunidad de Madrid': 31.3,
       'Andalucía': 18.7
   },
   2020: {
       'Cataluña': 71.1,
       'Comunidad de Madrid': 18.4,
       'Andalucía': 9.2
   },
   2021: {
       'Cataluña': 48.7,
       'Andalucía': 13.7,
       'Comunidad de Madrid': 12.6
   },
   2022: {
       'Cataluña': 58.9,
       'Comunidad de Madrid': 24.8,
       'Andalucía': 6.7
   },
   2023: {
       'Cataluña': 74.7,
       'Andalucía': 10.7,
       'Comunidad de Madrid': 9.1
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2021: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2022: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2023: {'vuelos': 484, 'asientos': 137987, 'pasajeros': 123635}
}

# Aeropuertos principales
aeropuertos_principales = {
   2023: [{'nombre': 'Incheon Intl (Seúl)', 'pasajeros': 123631, 'porcentaje': 100.00}]
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
       1: (uniform(-3.0, 1.0), uniform(20.0, 40.0)),
       2: (uniform(0.0, 4.0), uniform(25.0, 45.0)),
       3: (uniform(5.0, 9.0), uniform(30.0, 50.0)),
       4: (uniform(12.0, 16.0), uniform(50.0, 70.0)),
       5: (uniform(17.0, 21.0), uniform(80.0, 100.0)),
       6: (uniform(21.0, 25.0), uniform(120.0, 140.0)),
       7: (uniform(24.0, 28.0), uniform(150.0, 170.0)),
       8: (uniform(24.0, 28.0), uniform(130.0, 150.0)),
       9: (uniform(19.0, 23.0), uniform(90.0, 110.0)),
       10: (uniform(13.0, 17.0), uniform(50.0, 70.0)),
       11: (uniform(6.0, 10.0), uniform(30.0, 50.0)),
       12: (uniform(-1.0, 3.0), uniform(20.0, 40.0))
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
               'Pais': 'Corea del Sur',
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
df.to_csv('turistas_corea_del_sur_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))