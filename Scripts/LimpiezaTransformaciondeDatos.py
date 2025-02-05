import pandas as pd
import numpy as np
from datetime import datetime

def cargar_datos(archivo):
    """
    Carga el dataset y realiza conversiones iniciales
    """
    df = pd.read_csv(archivo, encoding='utf-8')
    columns_to_drop = ['Num_Vuelos', 'Num_Asientos', 'Num_Pasajeros']
    df = df.drop(columns=columns_to_drop)
    return df

def limpiar_datos(df):
    """
    Realiza la limpieza general del dataset
    """
    df_limpio = df.copy()
    
    # Asegurar que Num_Turistas sea numérico
    df_limpio['Num_Turistas'] = pd.to_numeric(df_limpio['Num_Turistas'], errors='coerce')
    
    # Normalizar strings
    columnas_texto = ['Pais', 'Destino_Principal', 'Medio_Acceso', 'Organizacion_Viaje', 'Estacionalidad']
    for columna in columnas_texto:
        df_limpio[columna] = df_limpio[columna].str.strip()
    
    # Asegurar que las columnas numéricas sean float
    df_limpio['Temperatura'] = pd.to_numeric(df_limpio['Temperatura_Media'], errors='coerce')
    df_limpio['Precipitacion'] = pd.to_numeric(df_limpio['Precipitacion_Media'], errors='coerce')
    df_limpio['Satisfaccion'] = pd.to_numeric(df_limpio['Satisfaccion'], errors='coerce')
    
    return df_limpio

def validar_datos(df):
    """
    Realiza validaciones básicas del dataset
    """
    validaciones = {
        'registros_totales': len(df),
        'valores_nulos': df.isnull().sum().to_dict(),
        'paises_unicos': df['Pais'].nunique(),
        'destinos_unicos': df['Destino_Principal'].nunique(),
        'rango_anos': f"{df['Anio'].min()} - {df['Anio'].max()}",
        'rango_satisfaccion': f"{df['Satisfaccion'].min():.2f} - {df['Satisfaccion'].max():.2f}"
    }
    return validaciones

def agregar_caracteristicas(df):
    """
    Agrega nuevas características útiles al dataset
    """
    df_nuevo = df.copy()
    
    # Crear fecha completa
    df_nuevo['Fecha'] = df_nuevo.apply(lambda x: f"{x['Anio']}-{x['mes']:02d}-01", axis=1)
    df_nuevo['Fecha'] = pd.to_datetime(df_nuevo['Fecha'])
    
    # Agregar trimestre
    df_nuevo['Trimestre'] = df_nuevo['Fecha'].dt.quarter
    
    # Categorizar temporada
    condiciones = [
        df_nuevo['mes'].isin([7, 8]),
        df_nuevo['mes'].isin([6, 9]),
        df_nuevo['mes'].isin([4, 5, 10]),
        df_nuevo['mes'].isin([1, 2, 3, 11, 12])
    ]
    valores = ['Alta', 'Media-Alta', 'Media', 'Baja']
    df_nuevo['Temporada'] = np.select(condiciones, valores, default='No Definida')
    
    return df_nuevo

def detectar_anomalias(df):
    """
    Detecta posibles anomalías en los datos
    """
    anomalias = {
        'turistas_cero': df[df['Num_Turistas'] == 0]['Pais'].value_counts().to_dict(),
        'satisfaccion_extrema': df[
            (df['Satisfaccion'] < 5) | (df['Satisfaccion'] > 10)
        ][['Pais', 'Satisfaccion']].values.tolist(),
        'temperatura_extrema': df[
            (df['Temperatura'] < 0) | (df['Temperatura'] > 45)
        ][['Pais', 'Fecha', 'Temperatura']].values.tolist(),
        'precipitacion_extrema': df[
            df['Precipitacion'] > 100
        ][['Pais', 'Fecha', 'Precipitacion']].values.tolist()
    }
    return anomalias

def generar_resumen(df):
    """
    Genera un resumen estadístico del dataset
    """
    resumen = {
        'total_turistas_por_ano': df.groupby('Anio')['Num_Turistas'].sum().to_dict(),
        'distribucion_medios_acceso': df.groupby('Medio_Acceso')['Num_Turistas'].sum().div(df['Num_Turistas'].sum()) * 100,
        'satisfaccion_promedio_por_pais': df.groupby('Pais')['Satisfaccion'].mean().round(2).to_dict(),
        'temperatura_promedio_por_temporada': df.groupby('Temporada')['Temperatura'].mean().round(2).to_dict(),
        'distribucion_organizacion': df.groupby('Organizacion_Viaje')['Num_Turistas'].sum().div(df['Num_Turistas'].sum()) * 100
    }
    return resumen

def main():
    try:
        # Cargar datos
        print("Cargando datos...")
        df_original = cargar_datos('turismo_completo_2019_2023.csv')
        
        # Proceso de limpieza
        print("Limpiando datos...")
        df_limpio = limpiar_datos(df_original)
        
        # Validaciones
        print("\nRealizando validaciones básicas:")
        validaciones = validar_datos(df_limpio)
        for k, v in validaciones.items():
            print(f"{k}: {v}")
        
        # Agregar características
        print("\nAgregando características adicionales...")
        df_final = agregar_caracteristicas(df_limpio)
        
        # Detectar anomalías
        print("\nDetectando anomalías:")
        anomalias = detectar_anomalias(df_final)
        for k, v in anomalias.items():
            if v:  # Solo mostrar si hay anomalías
                print(f"\n{k}:")
                print(v)
        
        # Generar resumen
        print("\nGenerando resumen estadístico:")
        resumen = generar_resumen(df_final)
        for k, v in resumen.items():
            print(f"\n{k}:")
            print(v)
        
        # Guardar dataset procesado
        print("\nGuardando dataset procesado...")
        df_final.to_csv('turismo_procesado_2019_2023.csv', index=False, encoding='utf-8')
        print("Proceso completado con éxito!")
        print(df_final)
        print(df_final.columns)
        
    except Exception as e:
        print(f"Error en el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()