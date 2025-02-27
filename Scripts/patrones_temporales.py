""""
script: patrones_temporales.py
Este script genera una visualización avanzada de los patrones e
stacionales de los medios de acceso turístico a España para el período 2019-2023.
author: Bernardino Chancusig Espin
fecha: 25/02/2025
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# Configuración estética simplificada
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Cargar los datos reales con la estructura correcta
df = pd.read_csv('turismo_completo_2019_2023.csv', encoding='utf-8')

# Verificar que las columnas necesarias estén disponibles
required_columns = ['Año', 'mes', 'Pais', 'Num_Turistas', 'Medio_Acceso']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Columna requerida '{col}' no encontrada en el dataset")

# Agrupar por mes y medio de acceso, calculando el promedio de turistas
monthly_avg = df.groupby(['mes', 'Medio_Acceso'])['Num_Turistas'].mean().reset_index()

# Pivotar los datos para tener una columna por medio de acceso
df_pivot = monthly_avg.pivot(index='mes', columns='Medio_Acceso', values='Num_Turistas').reset_index()

# Rellenar valores NaN con 0 si los hay
df_pivot = df_pivot.fillna(0)

# Verificar los medios de acceso disponibles
medios_acceso = [col for col in df_pivot.columns if col != 'mes']
print(f"Medios de acceso disponibles: {medios_acceso}")

# Crear el gráfico
plt.figure(figsize=(14, 8))

# Definir colores específicos para cada medio de acceso
colors = {
    'Aeropuerto': '#1f77b4',  # Azul
    'Carretera': '#ff7f0e',   # Naranja
    'Puerto': '#2ca02c',      # Verde
    'Tren': '#d62728'         # Rojo
}

# Crear gráfico con líneas y áreas sombreadas
for medio in medios_acceso:
    color = colors.get(medio, plt.cm.tab10.colors[medios_acceso.index(medio) % 10])
    plt.plot(df_pivot['mes'], df_pivot[medio], '-', linewidth=3, color=color, label=medio)
    plt.fill_between(df_pivot['mes'], 0, df_pivot[medio], alpha=0.1, color=color)

# Configurar el eje X para mostrar los nombres de los meses
month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
plt.xticks(np.arange(1, 13), month_names)

# Añadir cuadrículas, títulos y leyendas
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Patrones Estacionales por Medio de Acceso (2019-2023)', fontsize=16, pad=20)
plt.xlabel('Mes', fontsize=14)
plt.ylabel('Volumen de Turistas (Promedio)', fontsize=14)

# Formatear los números en miles o millones para mejor legibilidad
def format_thousands(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    else:
        return f'{x:.0f}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))

# Intentar añadir anotaciones relevantes basadas en los datos reales
try:
    # Picos en aeropuerto (temporada alta)
    if 'Aeropuerto' in df_pivot.columns:
        idx_max_aero = df_pivot['Aeropuerto'].idxmax()
        mes_max_aero = df_pivot.iloc[idx_max_aero]['mes']
        valor_max_aero = df_pivot.iloc[idx_max_aero]['Aeropuerto']
        plt.annotate('Picos en\ntemporada alta', 
                    xy=(mes_max_aero, valor_max_aero),
                    xytext=(mes_max_aero, valor_max_aero * 0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=10)
    
    # Estabilidad en carretera
    if 'Carretera' in df_pivot.columns:
        mes_medio = 6  # Mes de junio como punto medio
        valor_carretera = df_pivot.loc[df_pivot['mes'] == mes_medio, 'Carretera'].values[0]
        plt.annotate('Estable durante\ntodo el año', 
                    xy=(mes_medio, valor_carretera),
                    xytext=(mes_medio, valor_carretera * 1.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=10)
    
    # Estacionalidad en puerto
    if 'Puerto' in df_pivot.columns:
        idx_max_puerto = df_pivot['Puerto'].idxmax()
        mes_max_puerto = df_pivot.iloc[idx_max_puerto]['mes']
        valor_max_puerto = df_pivot.iloc[idx_max_puerto]['Puerto']
        plt.annotate('Mayor\nestacionalidad', 
                    xy=(mes_max_puerto, valor_max_puerto),
                    xytext=(mes_max_puerto, valor_max_puerto * 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=10)
    
    # Tendencia en tren
    if 'Tren' in df_pivot.columns:
        mes_final = 10  # Octubre como mes representativo
        valor_tren = df_pivot.loc[df_pivot['mes'] == mes_final, 'Tren'].values[0]
        plt.annotate('Tendencia\ndecreciente', 
                    xy=(mes_final, valor_tren),
                    xytext=(mes_final, valor_tren * 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', fontsize=10)
except Exception as e:
    print(f"No se pudieron añadir todas las anotaciones: {e}")

# Destacar la temporada alta con un área sombreada
plt.axvspan(6.5, 8.5, alpha=0.2, color='yellow', label='Temporada Alta')

# Añadir leyenda con posición personalizada y estilo mejorado
legend = plt.legend(loc='upper right', frameon=True, framealpha=0.95, shadow=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('gray')

# Ajustar los márgenes
plt.tight_layout()

# Guardar la figura en alta resolución
plt.savefig('patrones_estacionales_medios_acceso.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()