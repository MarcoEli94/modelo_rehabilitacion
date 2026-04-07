# Instrucciones para agregar nuevos llamados de ejemplo

## Regla principal
Solo se admiten ejemplos reales. No registrar archivos sinteticos, generados artificialmente, interpolados o aumentados.

## Procedimiento
1. Identificar el ejercicio correcto: use m07 para rotacion interna/externa y m09 para abduccion de hombro.
2. Guardar el JSON en la carpeta correspondiente dentro de json_examples/m07 o json_examples/m09.
3. Mantener una convencion de nombre estable, por ejemplo example_036.json, sin espacios ni sufijos ambiguos.
4. Verificar que el JSON represente una captura real y que conserve la estructura esperada por el pipeline actual.
5. Registrar el archivo en json_examples/labels_manifest.csv con tres campos: exercise, file, label.
6. Usar label=1 para llamado correcto y label=0 para llamado incorrecto, de acuerdo con el criterio clinico del proyecto.
7. Reejecutar las secciones de carga, evaluacion, comparativa de modos y resumen auditado del notebook.
8. Confirmar que el archivo nuevo aparezca en cobertura_auditoria_json_examples.csv sin faltantes ni labels vacios.

## Controles de auditoria
- No mezclar ejemplos de m07 dentro de m09 ni viceversa.
- No borrar registros historicos de labels_manifest.csv; agregar nuevas filas de forma trazable.
- Si un archivo se reemplaza, documentar el cambio y mantener el nombre de archivo sincronizado con labels_manifest.csv.
- Si aparece un faltante en manifest o un patron sintetico, la corrida no debe considerarse apta para auditoria.