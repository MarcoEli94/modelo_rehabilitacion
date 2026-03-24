# Modelo Asistente de Rehabilitacion Fisica con IA

Proyecto de analisis biomecanico y clasificacion de ejercicios de rehabilitacion fisica usando el dataset UI-PRMD.

Este repositorio integra:
- Analisis exploratorio de datos (EDA) del dataset UI-PRMD.
- Entrenamiento de modelos de clasificacion (Random Forest y TCN).
- Inferencia sobre secuencias de movimiento.
- Generacion de retroalimentacion en lenguaje natural con Ollama.

## 1. Contexto del proyecto

Este trabajo esta orientado al acompanamiento de pacientes durante rehabilitacion domiciliaria, con enfasis en detectar ejecuciones incorrectas de ejercicios y entregar retroalimentacion clara.

Con base en la documentacion de entrega y el EDA, se toma como referencia el dataset UI-PRMD (University of Idaho Physical Rehabilitation Movement Dataset), que contiene secuencias capturadas con Kinect y Vicon para movimientos correctos e incorrectos.

## 2. Objetivo

Construir un asistente capaz de:
1. Analizar secuencias de movimiento humano.
2. Clasificar ejecucion correcta vs incorrecta.
3. Identificar errores biomecanicos relevantes.
4. Traducir los resultados del modelo a feedback entendible para paciente.

## 3. Dataset: UI-PRMD

El proyecto usa la carpeta `dataset/` con estas divisiones principales:
- `Movements/Movements/`
- `Incorrect Movements/Incorrect Movements/`
- `Segmented Movements/Segmented Movements/`
- `Incorrect Segmented Movements/Incorrect Segmented Movements/`

En cada division se encuentran datos de:
- `Kinect/Positions`
- `Kinect/Angles`
- `Vicon/Positions`
- `Vicon/Angles`

En los modelos actuales de hombro se prioriza Kinect (especialmente Positions) para construir ventanas temporales y extraer features biomecanicas.

## 4. Ejercicios objetivo

En esta version se trabaja principalmente con:
- `m07`: Standing Shoulder Internal/External Rotation
- `m09`: Standing Shoulder Abduction

## 5. Estructura del repositorio

- `EDA_UI_PRMD.ipynb`: Analisis exploratorio del dataset (estructura, conteos, carga de archivos y primeras estadisticas).
- `Entrenamiento_Modelo.ipynb`: Pipeline base con Random Forest, foco en recall de clase incorrecta.
- `standing_shoulder_abduction/`: Version especializada del ejercicio m09 (TCN + artefactos + demo).
- `standing_shoulder_internal_external_rotation/`: Version especializada del ejercicio m07 (TCN + artefactos + demo + cliente Ollama).
- `jsonEjemplo.json`: Ejemplo de payload para pruebas de inferencia.

## 6. Pipeline general

1. Carga de secuencias UI-PRMD desde archivos `.txt`.
2. Filtrado por movimiento (m07 o m09).
3. Construccion de ventana temporal fija.
4. Extraccion de features biomecanicas.
5. Entrenamiento del clasificador (Random Forest o TCN segun modulo).
6. Ajuste de umbral con prioridad en recall de ejecuciones incorrectas.
7. Exportacion de artefactos del modelo.
8. Inferencia y generacion de feedback para usuario final.

## 7. Formato de entrada JSON

Los modulos especializados aceptan formatos flexibles de entrada:
- Contenedores: `frames`, `ventana`, `secuencia`, `window`.
- Frame con joints en `puntos_clave` o `puntos`.
- Campo opcional de timestamp `t`.

Esto permite integrar facilmente datos provenientes de app movil o preprocesamiento intermedio.

## 8. Modelos implementados

### 8.1 Random Forest (baseline)
Notebook: `Entrenamiento_Modelo.ipynb`

- Clasificador `RandomForestClassifier` con `class_weight='balanced'`.
- Validacion orientada a sujetos (LOSO / por grupos segun configuracion).
- Ajuste de threshold para mejorar sensibilidad de clase incorrecta.

### 8.2 TCN (modelos por ejercicio)
Notebooks y scripts por ejercicio:
- `standing_shoulder_abduction/Standing_Shoulder_Abduction.py`
- `standing_shoulder_internal_external_rotation/Standing_Shoulder_Internal_External_Rotation.py`

Cada modulo:
- Carga secuencias UI-PRMD del movimiento objetivo.
- Calcula resumen biomecanico.
- Entrena red temporal tipo TCN.
- Exporta `.keras` + `.joblib` en su carpeta de artefactos.

## 9. Retroalimentacion con Ollama

En `standing_shoulder_internal_external_rotation/` se incluye un cliente para generar mensajes de feedback a partir de la salida del modelo:
- `ollama_client.py`
- `main.py`

Flujo:
1. Se toma la prediccion estructurada del modelo.
2. Se normaliza a un `diagnosis` compacto.
3. Se consulta Ollama (`/api/chat`) con esquema JSON de salida.
4. Se obtiene feedback breve y accionable.

## 10. Requisitos

## 10.1 Python
- Recomendado para entrenamiento TCN: Python 3.11 o 3.12 (compatibilidad TensorFlow).
- Python 3.14 puede presentar limitaciones con TensorFlow segun entorno.

## 10.2 Dependencias principales
Instalar en el entorno virtual activo:

```powershell
pip install numpy pandas scikit-learn joblib tensorflow requests matplotlib seaborn pypdf
```

## 10.3 Ollama (opcional para feedback)
- Tener Ollama activo en `http://localhost:11434`.
- Modelo por defecto en cliente: `qwen2.5:1.5b`.

## 11. Como ejecutar

## 11.1 EDA
1. Abrir `EDA_UI_PRMD.ipynb`.
2. Ejecutar celdas en orden para revisar estructura y estadisticas del dataset.

## 11.2 Entrenamiento baseline (RF)
1. Abrir `Entrenamiento_Modelo.ipynb`.
2. Ajustar movimientos objetivo (por defecto m07/m09).
3. Ejecutar entrenamiento y exportacion.

## 11.3 Entrenamiento TCN por ejercicio
1. Abrir notebook del ejercicio deseado:
   - `standing_shoulder_abduction/Standing_Shoulder_Abduction.ipynb`
   - `standing_shoulder_internal_external_rotation/Standing_Shoulder_Internal_External_Rotation.ipynb`
2. Verificar rutas de dataset y dependencias.
3. Ejecutar celdas de `get_dataset()` y `train_and_export()`.

## 11.4 Prueba de feedback con Ollama (m07)
Desde la carpeta `standing_shoulder_internal_external_rotation/`:

```powershell
python main.py
```

## 12. Resultados esperados

- Clasificacion binaria: `correcta` / `incorrecta`.
- Probabilidad de ejecucion incorrecta y threshold aplicado.
- Severidad estimada.
- Lista de errores biomecanicos detectados.
- Resumen biomecanico por secuencia.
- Feedback textual orientado a correccion prioritaria.

## 13. Limitaciones actuales

- Los modelos especializados estan centrados en m07 y m09.
- La calidad depende de la cobertura y limpieza de secuencias del dataset.
- El feedback LLM depende de disponibilidad local de Ollama y del modelo descargado.
- Para despliegue clinico se requiere validacion adicional con expertos y pruebas de seguridad/uso.

## 14. Trabajo futuro

- Extender cobertura a mas ejercicios UI-PRMD (m01-m10).
- Mejorar calibracion de severidad por paciente.
- Integrar API formal para app movil en tiempo real.
- Incorporar evaluacion longitudinal de progreso del paciente.

## 15. Referencias

- UI-PRMD: University of Idaho Physical Rehabilitation Movement Dataset.
- Vakanski et al., trabajo original del dataset UI-PRMD.
- Documentacion interna del proyecto: `Entrega_3 (5).pdf`.
