# ModeloRehabilitacion 🤖🏥

Sistema de Rehabilitación Motriz con IA y API REST basado en análisis biomecánico de movimientos.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

ModeloRehabilitacion es una solución completa de rehabilitación motriz basada en inteligencia artificial desarrollada con Python. El sistema integra modelos de aprendizaje profundo (TCN - Temporal Convolutional Networks) para la clasificación de movimientos correctos/incorrectos en ejercicios de rehabilitación de hombros, una API REST moderna para la exposición de servicios, y herramientas de análisis de datos para el procesamiento de secuencias biomecánicas.

### 🎯 Casos de Uso
- **Clasificación automática** de movimientos correctos vs incorrectos
- **Análisis biomecánico** de secuencias de rehabilitación
- **API REST** para integración con aplicaciones móviles/web
- **Análisis exploratorio** de datos de movimiento
- **Entrenamiento de modelos** con datasets especializados

## 🚀 Características Principales

- ✅ **Modelos TCN** para clasificación temporal de movimientos
- ✅ **API FastAPI** con documentación automática OpenAPI/Swagger
- ✅ **Procesamiento biomecánico** avanzado (ángulos, velocidad, simetría)
- ✅ **Dataset UI-PRMD** integrado para movimientos m07/m09
- ✅ **Análisis exploratorio** con Jupyter Notebooks
- ✅ **Visualización** de datos y resultados
- ✅ **Integración Ollama** para IA generativa complementaria
- ✅ **Despliegue containerizado** con Docker

## 🏗️ Arquitectura

```
modelo_rehabilitacion/
├── api/                          # API REST con FastAPI
│   ├── main.py                   # Servidor principal
│   ├── ollama_client.py          # Cliente Ollama
│   └── requirements.txt          # Dependencias API
├── docs/                         # Documentación y anexos
│   ├── DOCUMENTACION_STACK_TECNOLOGICO_PROYECTO.ipynb
│   └── EDA_UI_PRMD.pdf
├── models/
│   └── exports/                  # Exportaciones legacy de modelos
├── standing_shoulder_abduction/  # Modelo m09 (Abducción)
│   ├── *.py                      # Lógica del modelo
│   ├── *_artifacts/              # Modelos entrenados
│   └── *.ipynb                   # Notebooks de análisis
├── standing_shoulder_internal_external_rotation/  # Modelo m07
│   ├── *.py                      # Lógica del modelo
│   ├── *_artifacts/              # Modelos entrenados
│   └── *.ipynb                   # Notebooks de análisis
├── dataset/                      # Datos UI-PRMD
├── main.py                       # Script de orquestación
├── EDA_UI_PRMD.ipynb             # Análisis exploratorio
├── Entrenamiento_Modelo.ipynb    # Entrenamiento modelos
├── requirements.txt              # Dependencias globales
└── README.md                     # Esta documentación
```

## 🛠️ Stack Tecnológico

### Lenguaje y Plataforma
- **Python 3.11+** - Lenguaje principal
- **FastAPI + Uvicorn** - Framework web y servidor ASGI
- **TensorFlow 2.15** - Framework de aprendizaje profundo
- **Jupyter Notebook** - Entorno de análisis interactivo

### Librerías Principales

| Categoría | Librería | Versión | Propósito |
|-----------|----------|---------|-----------|
| **Framework Web** | FastAPI | 0.104.1 | API REST moderna |
| | Uvicorn | 0.24.0 | Servidor ASGI |
| **Aprendizaje Profundo** | TensorFlow | 2.15.0 | Modelos TCN |
| | Scikit-learn | 1.3.2 | ML y evaluación |
| | Joblib | 1.3.2 | Serialización |
| **Ciencia de Datos** | NumPy | 1.26.2 | Arrays numéricos |
| | Pandas | 2.1.4 | Datos tabulares |
| | SciPy | 1.11.4 | Análisis científico |
| **Visualización** | Matplotlib | 3.8.2 | Gráficos 2D |
| | Seaborn | 0.13.0 | Visualización estadística |

## 📦 Instalación

### Prerrequisitos
- Python 3.11 o superior
- pip (gestor de paquetes)
- Git (opcional, para clonar)

### Instalación Local

1. **Clonar el repositorio** (si aplica):
```bash
git clone <url-del-repo>
cd modelo_rehabilitacion
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Instalar dependencias de la API** (opcional):
```bash
cd api
pip install -r requirements.txt
```

### Instalación con Docker

```bash
# Construir imagen
docker build -t modelo-rehabilitacion .

# Ejecutar contenedor
docker run -p 8000:8000 modelo-rehabilitacion
```

## 🚀 Uso

### Ejecutar la API

```bash
# Desde el directorio raíz
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000` con documentación automática en `http://localhost:8000/docs`.

### Ejecutar Análisis Exploratorio

```bash
# Abrir Jupyter Notebook
jupyter notebook EDA_UI_PRMD.ipynb
```

### Entrenar Modelos

```bash
# Abrir notebook de entrenamiento
jupyter notebook Entrenamiento_Modelo.ipynb
```

## 📊 Modelos Disponibles

### M07 - Rotación Interna/Externa de Hombro
- **Features:** 10 biomecánicos (ángulos, velocidad, simetría)
- **Arquitectura:** TCN con dilatación exponencial
- **Dataset:** UI-PRMD movimientos m07
- **Métricas:** Recall >90%, F1-score optimizado

### M09 - Abducción de Hombro
- **Features:** 10 biomecánicos (ángulos, velocidad, simetría)
- **Arquitectura:** TCN con bloques residuales
- **Dataset:** UI-PRMD movimientos m09
- **Métricas:** Recall >90%, F1-score optimizado

## 🔧 Configuración

### Variables de Entorno

Crear archivo `.env` en la raíz:

```env
PYTHONPATH=/path/to/modelo_rehabilitacion
TF_CPP_MIN_LOG_LEVEL=2
OLLAMA_HOST=http://localhost:11434
```

### Configuración de Modelos

Los modelos se cargan automáticamente desde las carpetas `_artifacts`. Para reentrenar:

1. Ejecutar `Entrenamiento_Modelo.ipynb`
2. Los nuevos modelos se guardan en `*_artifacts/`
3. Reiniciar la API para cargar los nuevos modelos

## 📈 Pipeline de ML

### Features Biomecánicos (10 canales)
1. **Ángulo de hombro** derecho
2. **Relación muñeca/codo** normalizada
3. **Relación codo/hombro** normalizada
4. **Elevación de hombro**
5. **Inclinación del tronco**
6. **Diferencia de simetría**
7. **Disponibilidad de simetría**
8. **Velocidad angular**
9. **Progreso de rango** de movimiento
10. **Ratio de puntos válidos**

### Arquitectura TCN
- **Capas convolucionales causales** para temporalidad
- **Dilatación exponencial** (1, 2, 4, 8)
- **Bloques residuales** con skip connections
- **Regularización** (dropout 0.15)
- **Optimización** Adam + early stopping

## 🧪 Testing

### Ejecutar Tests
```bash
# Tests básicos
python -m pytest tests/

# Tests de API
pytest api/tests/
```

### Validación de Modelos
```bash
# En Jupyter notebook
jupyter notebook Entrenamiento_Modelo.ipynb
# Ejecutar sección de validación
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama para feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Guías de Contribución
- Seguir PEP 8 para estilo de código
- Agregar tests para nuevas funcionalidades
- Actualizar documentación
- Usar type hints en Python

## 📚 Documentación Adicional

- [Documentación Técnica del Stack](docs/DOCUMENTACION_STACK_TECNOLOGICO_PROYECTO.ipynb)
- [Análisis Exploratorio](EDA_UI_PRMD.ipynb)
- [Entrenamiento de Modelos](Entrenamiento_Modelo.ipynb)
- [API Documentation](http://localhost:8000/docs) (cuando esté ejecutándose)

## 🔍 Referencias Académicas

- Bai, S., et al. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*
- Lea, C., et al. (2012). Temporal convolutional networks for action segmentation and detection. *CVPR*
- Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- **Marco** - Desarrollo principal
- **Equipo de Investigación** - Análisis biomecánico y validación clínica

## 🙏 Agradecimientos

- Dataset UI-PRMD por proporcionar datos de movimiento reales
- Comunidad de código abierto por las librerías utilizadas
- Investigadores en visión por computadora y rehabilitación

---

**Última actualización:** Abril 2026

⭐ Si este proyecto te resulta útil, ¡dale una estrella!

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
- Modelo por defecto en cliente: `phi3:mini`.

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
