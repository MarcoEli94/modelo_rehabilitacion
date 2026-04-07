# Reporte de Integracion y Reproducibilidad (Recall objetivo 0.92)

## 1. Reproduccion del experimento
1. Ejecutar en orden las secciones de configuracion, carga de modelos y evaluacion del notebook.
2. Validar el modo de inferencia (`RECOVERY_MODE`) y confirmar en `recovery_model_status.csv`.
3. Ejecutar el barrido de umbral para objetivo de recall y exportar artefactos.
4. Verificar consistencia en archivos de salida dentro de `resultados_capitulo_6`.

## 2. Resultados y comparativos
- Modo de inferencia activo: **Recovery_OOF_LogReg**.
- Umbral seleccionado para objetivo de recall=0.92: **0.915**.
- Recall: **0.9300**.
- Precision: **1.0000**.
- F1: **0.9637**.
- Specificity: **1.0000**.
- Balanced Accuracy: **0.9650**.
- Cumplimiento MVP: **PASS**.

Comparativo por modo:
- Baseline: Recall=1.0000, Precision=0.5000, F1=0.6667, Specificity=0.0000.
- TCN_only: Recall=1.0000, Precision=0.5000, F1=0.6667, Specificity=0.0000.
- Recovery_OOF_LogReg: Recall=1.0000, Precision=0.9901, F1=0.9950, Specificity=0.9900.

## 3. Discusion (mejora respecto a baseline)
- El baseline presenta recall alto pero no discrimina correctamente la clase negativa, por lo que su utilidad clinica es limitada.
- El modo TCN_only mantiene el mismo patron del baseline y no logra mejoras en precision ni specificity.
- El modo Recovery_OOF_LogReg incrementa de manera marcada precision, F1 y specificity, permitiendo cumplir el criterio MVP.
- Mejora vs baseline: Delta Precision=0.4901, Delta F1=0.3284, Delta Specificity=0.9900.
- Conclusion: bajo el flujo actual, la estrategia de recovery provee un punto operativo reproducible y auditable con recall cercano al objetivo (0.92) y cumplimiento integral de metricas.

## 4. Figuras y tablas generadas
- `resultados_capitulo_6/figuras_integracion/fig_resumen_recall_objetivo_092.png`
- `resultados_capitulo_6/figuras_integracion/fig_barrido_recall_objetivo_092.png`
- `resultados_capitulo_6/comparativa_integrada_baseline_tcn_recovery.csv`
- `resultados_capitulo_6/resumen_recall_objetivo_092.csv`