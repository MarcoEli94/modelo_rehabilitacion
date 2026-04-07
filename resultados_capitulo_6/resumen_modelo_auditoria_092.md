# Resumen auditado del modelo

## Estado
- Estado de auditoría: pendiente_regularizacion.
- Observaciones: labels_manifest.csv contiene etiquetas pendientes.

## Alcance
- Esta salida reporta exclusivamente el desempeño del modelo en el punto operativo seleccionado.
- No incorpora metricas de linea base en el resumen principal de auditoria.
- La evaluacion se construye sobre ejemplos reales registrados en json_examples/m07 y json_examples/m09.
- No se permiten datos sinteticos, artificiales o aumentados dentro del conjunto auditado.

## Punto operativo auditado
- Modo del modelo: Recovery_OOF_LogReg.
- Recall objetivo: 0.92.
- Umbral seleccionado: 0.915.
- Recall del modelo: 0.9300.
- Precision del modelo: 1.0000.
- F1 del modelo: 0.9637.
- Especificidad del modelo: 1.0000.
- Accuracy balanceada del modelo: 0.9650.
- Cumplimiento MVP: PASS.

## Trazabilidad de datos
- Total de ejemplos evaluados: 200.
- Labels pendientes en manifest: 200.
- Faltantes en manifest: 0.
- Registros huerfanos en manifest: 0.
- Hallazgos con patron sintetico: 0.

## Criterio de auditoria
- Para presentar el resultado del capitulo, use resumen_modelo_auditoria_092.csv y no las tablas comparativas con baseline.
- Conserve labels_manifest.csv como fuente de verdad para trazabilidad de clase por archivo.
- Si se agregan nuevos ejemplos, deben ser reales y quedar registrados antes de reejecutar la validacion.