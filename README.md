Título  del proyecto
Predicción del medio de transporte de los turistas hacia España mediante modelos de Machine Learning y Deep Learning
Este repositorio contiene el código, datos y visualizaciones desarrollados para mi Trabajo de Fin de Máster en la Universidad Internacional de La Rioja (UNIR), para el Máster Universitario en Análisis y Visualización de Datos Masivos/Visual Analytics and Big Data.

Descripción del proyecto

El proyecto aborda la predicción de los medios de transporte (aeropuerto, carretera, puerto y tren) que utilizan los turistas para acceder a España. Mediante técnicas avanzadas de Machine Learning (Random Forest) y Deep Learning (Redes Neuronales Artificiales), se analizan patrones en función de múltiples variables como el país de origen, estacionalidad, condiciones climáticas, nivel de satisfacción y volumen de visitantes.
Los modelos desarrollados alcanzan una precisión global del 78% (Random Forest), con particular efectividad en la predicción del acceso aeroportuario (95% de precisión).

Objetivos
Objetivo general
Desarrollar un modelo predictivo basado en Machine Learning y Deep Learning para anticipar los medios de transporte  turístico a España, considerando múltiples variables como país de origen, estacionalidad, condiciones climáticas, nivel de satisfacción de los turistas y volumen de visitantes.

Objetivos específicos

Recopilar y procesar datos turísticos provenientes de FRONTUR, TURESPAÑA y AEMET, integrando información de 26 mercados emisores durante el período 2019-2023.
Analizar la influencia relativa de los factores determinantes en la elección de medios de transporte turístico mediante técnicas de análisis de importancia de variables.
Desarrollar y entrenar modelos predictivos basados en Random Forest y Redes Neuronales Artificiales, aplicando técnicas de validación cruzada y optimización de hiperparámetros.
Evaluar y comparar sistemáticamente el rendimiento de los modelos implementados según métricas de precisión global, recall, F1-score y matrices de confusión.
Implementar los resultados en dashboards interactivos de Business Intelligence que faciliten la visualización de predicciones.

Estructura del repositorio
Datasets
Contiene los conjuntos de datos utilizados y generados durante el proyecto:

turismo_completo_2019_2023.csv: Dataset con los datos recopilados y preprocesados para el período 2019-2023.
turismo_procesado_2019_2023.csv: Dataset limpio y transformado utilizado para el entrenamiento de los modelos.
predicciones_finales_2019_2028.csv: Datos históricos y predicciones generadas por el modelo para el período 2024-2028.
turistas_alemania_2019_2023.csv: Ejemplo de dataset específico para el mercado alemán.
Carpeta otros_paises: Datasets individuales para los 26 países analizados.

Scripts
Contiene todos los scripts Python utilizados en el proyecto:

Procesamiento de datos:

alemania.py: Ejemplo de script para procesar datos específicos de un país.
LimpiezaTransformaciondeDatos.py: Script para la limpieza y transformación del dataset principal.
generar_dataset_histórico_predictivo.py: Generación del dataset para validación de predicciones.


Modelos:

implementacion_random_forest.py: Implementación del modelo Random Forest.
implementacion_redes_neuronales.py: Implementación del modelo de Redes Neuronales Artificiales.
aplicar_modelo.py: Script para aplicar el modelo a datos nuevos.
comparacion_historicos_predicciones_con_analisis_confianza.py: Análisis comparativo de resultados y confianza.
estadisticos_comparacion_modelos.py: Análisis estadístico comparativo entre modelos.


Visualizaciones:

patrones_temporales.py: Generación de gráficos de patrones estacionales.
mapa_calor_accesos_turistico.py: Creación de mapas de calor por medio de acceso.
series_temporales_tendencias.py: Análisis de series temporales.
series_temporales_tendencias_con_dash.py: Dashboard interactivo con Dash.
visualizaciones_turismo_con_treemap.py: Visualizaciones con Treemap.
visualizar_modelo_random_forest.py: Visualización de la estructura del modelo Random Forest.



GraficosInteractivos
Contiene los archivos y recursos para visualizaciones interactivas:

aniopismedioacceso.pbix: Dashboard en Power BI para análisis por país y medio de acceso.
aniopismedioacceso_datos_hist_predichos.pbix: Dashboard con datos históricos y predicciones.
tendenciasypredicciones.twb: Dashboard en Tableau para análisis de tendencias.

Archivos CSV de soporte para las visualizaciones:
accesos_por_pais.csv
turistas_por_anio_pais_medio_2019_2028.csv
turistas_por_anio_pais_medio_destino_2019_2028.csv
metricas_rf_ann.csv



Requisitos e instalación
Prerrequisitos

Python 3.8 o superior
pip (gestor de paquetes de Python)

Instalación de dependencias
El proyecto utiliza diversas bibliotecas de Python para análisis de datos, aprendizaje automático y visualización. Para instalar todas las dependencias necesarias, ejecute el siguiente comando:

pip install -r requirements.txt

Este comando instalará automáticamente:

pandas, numpy para manipulación de datos
scikit-learn para implementación de Random Forest y evaluación de modelos
tensorflow y keras para redes neuronales
matplotlib y seaborn para visualizaciones
Y otras dependencias necesarias para el proyecto

Verificación del entorno
Para verificar que la instalación se ha completado correctamente, puede ejecutar:
python -c "import pandas; import numpy; import tensorflow; import sklearn; print('Entorno configurado correctamente')"

Guía de uso
Procesamiento de datos

Para procesar los datos originales y generar el dataset limpio:
 pre-requisitos: dataset turismo_completo_2019_2023.csv debe copiarse a carpeta Scripts
 Ejecutar el siguiente comando:
  python Scripts/LimpiezaTransformaciondeDatos.py

Entrenamiento de modelos

Para entrenar el modelo Random Forest:
  pre-requisito: dataset turismo_procesado_2019_2023.csv' debe estar copiado en la carpeta Scripts/data/processed/
  Ejecutar el siguiente comando:
  python Scripts/implementacion_random_forest.py

Para entrenar el modelo de Redes Neuronales:
 pre-requisito: dataset turismo_procesado_2019_2023.csv' debe estar copiado en la carpeta Scripts/data/processed/
 Ejecutar el siguiente comando:
  python Scripts/implementacion_redes_neuronales.py

Genera proyecciones futuras (2024-2028) basadas en datos históricos (2019-2023).
 pre-requisito: dataset turismo_procesado_2019_2023.csv' debe estar copiado en la carpeta Scripts/
 Ejecutar el siguiente comando:
 python Scripts/generar_dataset_histórico_predictivo.py

Generación de predicciones

Para aplicar el modelo entrenado a nuevos datos:
 pre-requisito: dataset dataset_completo_2019_2028.csv y modelo entrenado random_forest_model.pkl deben estar copiados en carpeta Scripts/
 Ejecutar el siguiente comando:
 python Scripts/aplicar_modelo.py


Visualizaciones 

Para generar gráficos de patrones estacionales:
pre-requisito: dataset turismo_completo_2019_2023.csv de estar copiado en carpeta Scripts/
Ejecutar el siguiente comando:
python Scripts/patrones_temporales.py

Para crear gráficos mapas de calor accesos turisticos
pre-requisito: dataset turismo_procesado_2019_2023.csv de estar copiado en carpeta Scripts/
Ejecutar el siguiente comando:
python Scripts/mapa_calor_accesos_turistico.py

Resultados principales

El modelo Random Forest alcanzó una precisión global del 78%, superando significativamente al modelo de Redes Neuronales (58%).
Variables más influyentes: número de turistas (43.85%), condiciones climáticas (19.28%) y país de origen (11.77%).
Precisión por medio de acceso (Random Forest):

Aeropuerto: 95%
Carretera: 80%
Puerto: 69%
Tren: 67%



Visualizaciones y dashboards
Los dashboards interactivos se pueden abrir con:

Power BI: Archivos .pbix en la carpeta GraficosInteractivos
Tableau: Archivo tendenciasypredicciones.twb
Para crear  dashboard interactivo series temporales  con Dash:
pre-requisito: dataset turismo_completo_2019_2023.csv de estar copiado en carpeta Scripts/
Ejecutar el siguiente comando:
python Scripts/series_temporales_tendencias_con_dash.py

Para crear  dashboard interactivo con Treemap:
pre-requisito: dataset predicciones_finales_2019_2028.csv de estar copiado en carpeta Scripts/
Ejecutar el siguiente comando:
python Scripts/visualizaciones_turismo_con_treemap.py


Autor
Bernardino Chancusig Espín
Universidad Internacional de La Rioja (UNIR)
Máster Universitario en Análisis y Visualización de Datos Masivos/Visual Analytics and Big Data
Licencia
Este proyecto está bajo licencia CHEB. Ver el archivo LICENSE para más detalles.
