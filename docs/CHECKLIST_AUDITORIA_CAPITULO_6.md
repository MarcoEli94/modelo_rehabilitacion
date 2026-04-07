# Checklist de Auditoria - Capitulo 6

Fecha de revision: ____________________
Responsable: ____________________
Version del repo (commit/tag): ____________________

## 1) Integridad de datos y trazabilidad

- [ ] Existe evidencia de origen de datos reales para m07 y m09.
- [ ] Se evita codigo de generacion artificial dentro del flujo oficial de validacion.
- [ ] Se documenta criterio de etiquetado usado en json_examples.
- [ ] Se conserva trazabilidad de archivos fuente usados en evaluacion.
- [ ] No hay dependencias ocultas fuera del repositorio para reproducir resultados.

Evidencia:
- Archivo principal de validacion: Capitulo_6_Validacion_Diseno_Experimental.ipynb
- Carpeta de ejemplos: json_examples
- Resultados exportados: resultados_capitulo_6

## 2) Reproducibilidad tecnica

- [ ] Semilla global definida y aplicada.
- [ ] Versiones de librerias registradas en salida del notebook.
- [ ] Rutas de carga de modelos y datos son relativas al proyecto.
- [ ] El notebook corre de arriba a abajo sin errores.
- [ ] Los artefactos de salida se generan en la misma estructura en cada corrida.

## 3) Metricas de exito (MVP)

Criterios objetivo:
- Recall >= 0.90
- Precision >= 0.75
- F1 >= 0.80
- Specificity >= 0.70

Validar en tabla de resultados consolidada:
- [ ] Recall cumple
- [ ] Precision cumple
- [ ] F1 cumple
- [ ] Specificity cumple
- [ ] Decision final PASS en resumen_validacion.json

## 4) Calidad estadistica minima

- [ ] Se reportan IC 95% para Recall y Precision.
- [ ] Se reporta comparacion contra baseline.
- [ ] Se reporta prueba pareada (McNemar o equivalente).
- [ ] Se incluyen matrices de confusion y tabla de edge cases.

## 5) Riesgo de sesgo y uso responsable

- [ ] Se incluye checklist de etica y sesgos.
- [ ] Se declara limitacion sobre variables sensibles si no existen.
- [ ] Se define supervision humana para uso de alto impacto.
- [ ] No se exponen datos sensibles en logs o reportes.

## 6) Estructura y orden profesional del repositorio

- [ ] Documentacion concentrada en docs.
- [ ] Exportaciones de modelo separadas en models/exports.
- [ ] README actualizado con rutas reales.
- [ ] No hay archivos legacy ambiguos en la raiz sin justificacion.

## 7) Evidencia para auditor

Adjuntar:
- [ ] Captura o export de tabla de metricas por escenario.
- [ ] resumen_validacion.json actualizado de la ultima corrida.
- [ ] indice_artefactos_reporte.json con lista de tablas y figuras.
- [ ] Registro de comando/flujo de ejecucion reproducible.

## 8) Criterio de aceptacion de auditoria interna

Aprobado si:
- [ ] Se cumplen los 4 umbrales de MVP en escenario consolidado real.
- [ ] Existe trazabilidad documental completa.
- [ ] El notebook es reproducible end-to-end.
- [ ] El paquete de evidencia esta completo.

Observaciones finales:

- __________________________________________________________
- __________________________________________________________
- __________________________________________________________
