import pandas as pd
import numpy as np
from random import uniform, choice

# Datos históricos por año
datos_anuales = {
   2019: {'total_turistas': 1195901, 'porc_paquete': 37.3},
   2020: {'total_turistas': 179097, 'porc_paquete': 24.3},
   2021: {'total_turistas': 457444, 'porc_paquete': 20.4},
   2022: {'total_turistas': 952131, 'porc_paquete': 27.9},
   2023: {'total_turistas': 1247570, 'porc_paquete': 26.1}
}

# Distribución mensual por año
distribucion_mensual = {
   2019: {
       1: 6.5, 2: 5.8, 3: 6.2, 4: 7.5, 5: 8.2, 6: 12.5, 
       7: 8.5, 8: 11.8, 9: 12.2, 10: 9.5, 11: 6.8, 12: 4.5
   },
   2020: {
       1: 15.0, 2: 25.0, 3: 12.0, 4: 0.5, 5: 0.3, 6: 2.5,
       7: 8.5, 8: 12.5, 9: 10.2, 10: 6.5, 11: 4.5, 12: 2.5
   },
   2021: {
       1: 1.2, 2: 1.5, 3: 1.8, 4: 2.0, 5: 5.5, 6: 12.5,
       7: 15.2, 8: 22.5, 9: 15.8, 10: 12.5, 11: 6.5, 12: 3.0
   },
   2022: {
       1: 2.0, 2: 3.5, 3: 5.2, 4: 8.5, 5: 11.2, 6: 16.5,
       7: 10.2, 8: 14.5, 9: 13.2, 10: 8.5, 11: 4.2, 12: 2.5
   },
   2023: {
       1: 3.5, 2: 4.2, 3: 5.5, 4: 12.0, 5: 13.5, 6: 15.2,
       7: 11.5, 8: 14.8, 9: 12.5, 10: 9.8, 11: 4.5, 12: 3.0
   }
}

# Medios de acceso por año
medios_acceso_por_año = {
   2019: {'Aeropuerto': 96.4, 'Carretera': 2.9, 'Puerto': 0.6, 'Tren': 0.1},
   2020: {'Aeropuerto': 89.8, 'Carretera': 9.6, 'Puerto': 0.5, 'Tren': 0.1},
   2021: {'Aeropuerto': 94.3, 'Carretera': 5.4, 'Puerto': 0.2, 'Tren': 0.1},
   2022: {'Aeropuerto': 93.6, 'Carretera': 5.7, 'Puerto': 0.6, 'Tren': 0.1},
   2023: {'Aeropuerto': 92.0, 'Carretera': 7.4, 'Puerto': 0.4, 'Tren': 0.1}
}

# Destinos principales por año
destinos_principales_por_año = {
   2019: {
       'Baleares': 39.2, 'Andalucía': 17.0, 'Canarias': 13.1,
       'Cataluña': 13.0, 'Comunidad Valenciana': 7.0
   },
   2020: {
       'Baleares': 25.0, 'Cataluña': 17.2, 'Canarias': 16.5,
       'Comunidad de Madrid': 15.4, 'Andalucía': 12.7
   },
   2021: {
       'Baleares': 41.3, 'Cataluña': 17.3, 'Canarias': 13.4,
       'Andalucía': 12.1, 'Comunidad Valenciana': 8.7
   },
   2022: {
       'Baleares': 39.7, 'Andalucía': 18.1, 'Cataluña': 15.5,
       'Canarias': 11.4, 'Comunidad Valenciana': 7.4
   },
   2023: {
       'Baleares': 38.3, 'Cataluña': 16.1, 'Canarias': 13.5,
       'Andalucía': 10.4, 'Comunidad Valenciana': 9.2
   }
}

# Información de vuelos por año
info_vuelos = {
   2019: {'vuelos': 'No disponible', 'asientos': 'No disponible', 'pasajeros': 'No disponible'},
   2020: {'vuelos': 2727, 'asientos': 454878, 'pasajeros': 264718},
   2021: {'vuelos': 4302, 'asientos': 704169, 'pasajeros': 508715},
   2022: {'vuelos': 8268, 'asientos': 1437442, 'pasajeros': 1233621},
   2023: {'vuelos': 9219, 'asientos': 1625472, 'pasajeros': 1457245}
}

# Aeropuertos principales
aeropuertos_principales = {
   2020: [{'nombre': 'Viena Int. (Lo)', 'pasajeros': 255349, 'porcentaje': 96.46}],
   2021: [{'nombre': 'Viena Int. (Lo)', 'pasajeros': 474377, 'porcentaje': 93.25}],
   2022: [{'nombre': 'Viena Int. (Lo)', 'pasajeros': 1170329, 'porcentaje': 94.87}],
   2023: [{'nombre': 'Viena Int. (Lo)', 'pasajeros': 1350473, 'porcentaje': 92.67}]
}

def normalizar_probabilidades(destinos):
   """Normaliza las probabilidades para que sumen 1"""
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
       1: (uniform(0.0, 2.0), uniform(70.0, 80.0)),
       2: (uniform(1.0, 3.0), uniform(65.0, 75.0)),
       3: (uniform(5.0, 8.0), uniform(60.0, 70.0)),
       4: (uniform(8.0, 12.0), uniform(55.0, 65.0)),
       5: (uniform(12.0, 16.0), uniform(65.0, 75.0)),
       6: (uniform(15.0, 20.0), uniform(85.0, 95.0)),
       7: (uniform(17.0, 22.0), uniform(90.0, 100.0)),
       8: (uniform(17.0, 22.0), uniform(85.0, 95.0)),
       9: (uniform(14.0, 18.0), uniform(70.0, 80.0)),
       10: (uniform(9.0, 13.0), uniform(60.0, 70.0)),
       11: (uniform(4.0, 7.0), uniform(70.0, 80.0)),
       12: (uniform(1.0, 3.0), uniform(75.0, 85.0))
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

# Crear lista para almacenar los datos
data_rows = []

# Generar datos para todos los años
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
               'Pais': 'Austria',
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
df.to_csv('turistas_austria_2019_2023.csv', index=False, encoding='utf-8', decimal=',')

# Mostrar las primeras filas
print(df.head(20))