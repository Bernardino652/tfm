import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 2525887, 'porc_paquete': 22.6},
   2020: {'total_turistas': 743411, 'porc_paquete': 17.4},
   2021: {'total_turistas': 1464091, 'porc_paquete': 19.8},
   2022: {'total_turistas': 2513389, 'porc_paquete': 19.8},
   2023: {'total_turistas': 2758684, 'porc_paquete': 19.7}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 5.0, 2: 5.2, 3: 6.8, 4: 7.5, 5: 9.8, 6: 10.2,
       7: 8.5, 8: 15.2, 9: 11.8, 10: 8.5, 11: 6.2, 12: 5.3
   },
   2020: {
       1: 15.8, 2: 18.2, 3: 9.5, 4: 0.5, 5: 0.3, 6: 2.1,
       7: 21.5, 8: 16.2, 9: 5.8, 10: 5.5, 11: 2.5, 12: 2.1
   },
   2021: {
       1: 1.2, 2: 1.0, 3: 1.2, 4: 2.5, 5: 5.2, 6: 8.5,
       7: 16.8, 8: 17.2, 9: 16.5, 10: 14.2, 11: 10.5, 12: 5.2
   },
   2022: {
       1: 4.8, 2: 5.2, 3: 6.5, 4: 9.2, 5: 8.5, 6: 9.8,
       7: 10.5, 8: 16.8, 9: 12.5, 10: 11.2, 11: 5.8, 12: 5.2
   },
   2023: {
       1: 5.2, 2: 5.8, 3: 6.5, 4: 9.2, 5: 8.5, 6: 8.2,
       7: 8.5, 8: 16.2, 9: 12.8, 10: 10.5, 11: 7.2, 12: 6.5
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 86.9, 'Carretera': 11.7, 'Puerto': 1.3, 'Tren': 0.1},
   2020: {'Aeropuerto': 80.4, 'Carretera': 19.4, 'Puerto': 0.2, 'Tren': 0.1},
   2021: {'Aeropuerto': 85.1, 'Carretera': 14.6, 'Puerto': 0.2, 'Tren': 0.1},
   2022: {'Aeropuerto': 86.3, 'Carretera': 12.9, 'Puerto': 0.8, 'Tren': 0.1},
   2023: {'Aeropuerto': 84.6, 'Carretera': 14.1, 'Puerto': 1.2, 'Tren': 0.1}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Comunidad Valenciana': 21.2, 'Andalucía': 20.6, 'Canarias': 16.6,
       'Cataluña': 14.6, 'Baleares': 10.2
   },
   2020: {
       'Comunidad Valenciana': 27.2, 'Canarias': 20.4, 'Andalucía': 16.6,
       'Cataluña': 10.4, 'Comunidad de Madrid': 6.8
   },
   2021: {
       'Comunidad Valenciana': 25.8, 'Andalucía': 19.6, 'Canarias': 18.7,
       'Baleares': 11.8, 'Cataluña': 11.7
   },
   2022: {
       'Comunidad Valenciana': 23.7, 'Andalucía': 19.1, 'Canarias': 18.2,
       'Cataluña': 12.9, 'Baleares': 11.9
   },
   2023: {
       'Comunidad Valenciana': 25.1, 'Andalucía': 18.6, 'Canarias': 15.2,
       'Cataluña': 14.8, 'Baleares': 10.0
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 7786, 'asientos': 1336140, 'pasajeros': 881103},
   2021: {'vuelos': 11973, 'asientos': 2027595, 'pasajeros': 1513520},
   2022: {'vuelos': 19254, 'asientos': 3345019, 'pasajeros': 2785160},
   2023: {'vuelos': 20014, 'asientos': 3549419, 'pasajeros': 3043023}
}

# Aeropuertos principales por año
aeropuertos_principales = {
   2020: [{'nombre': 'Bruselas', 'pasajeros': 596808, 'porcentaje': 67.73}],
   2021: [{'nombre': 'Bruselas', 'pasajeros': 987827, 'porcentaje': 65.27}],
   2022: [{'nombre': 'Bruselas', 'pasajeros': 1786411, 'porcentaje': 64.14}],
   2023: [{'nombre': 'Bruselas', 'pasajeros': 1992845, 'porcentaje': 65.49}]
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
       1: (uniform(3.0, 5.0), uniform(60.0, 70.0)),
       2: (uniform(4.0, 6.0), uniform(55.0, 65.0)),
       3: (uniform(7.0, 10.0), uniform(50.0, 60.0)),
       4: (uniform(9.0, 13.0), uniform(45.0, 55.0)),
       5: (uniform(13.0, 17.0), uniform(55.0, 65.0)),
       6: (uniform(16.0, 20.0), uniform(75.0, 85.0)),
       7: (uniform(18.0, 22.0), uniform(80.0, 90.0)),
       8: (uniform(18.0, 22.0), uniform(75.0, 85.0)),
       9: (uniform(15.0, 19.0), uniform(60.0, 70.0)),
       10: (uniform(11.0, 15.0), uniform(50.0, 60.0)),
       11: (uniform(7.0, 10.0), uniform(60.0, 70.0)),
       12: (uniform(4.0, 6.0), uniform(65.0, 75.0))
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
               'Pais': 'Bélgica',
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
df.to_csv('turistas_belgica_2019_2023.csv', index=False, encoding='utf-8', decimal=',')
print(df.head(20))