# Plan de implementación — Cerrar los 6 gaps

## Contexto

Se identificaron 6 gaps entre la implementación actual de FoldReport y el PLAN.md. Este plan cubre la implementación de todos ellos.

---

---

## Propuestos cambios

### 1. Demo multi-herramienta

Ampliar `examples/demo/` para incluir datos sintéticos de las 3 herramientas, no solo ColabFold.

#### [MODIFY] [make_demo.py](file:///d:/alphafold-report/examples/make_demo.py)

- Añadir funciones `_generate_af3_server_demo()`, `_generate_boltz_demo()` y `_generate_openfold3_demo()` siguiendo los mismos patrones que `make_fixtures.py` usa para tests
- Generar 2 modelos por herramienta → 8 predicciones totales en la demo (2 ColabFold + 2 AF3 + 2 Boltz + 2 OpenFold3)
- Regenerar `demo_report.html` y `demo_metrics.csv`

#### [MODIFY] [README.md](file:///d:/alphafold-report/README.md)

- Actualizar la sección "Quick start" para mencionar que la demo incluye datos de las 4 herramientas
- Añadir OpenFold3 a la tabla de herramientas soportadas

---

### 2. PAE interactivo (canvas + JS)

Reemplazar el heatmap PAE estático (PNG/matplotlib) por un heatmap interactivo en JavaScript con:

- **Hover**: tooltip mostrando `(residuo_alineado, residuo_scored, error_Å)`
- **Escala de color**: misma paleta que la versión estática (Greens_r, 0–30 Å)
- **Líneas de cadena**: divisiones por cadena superpuestas
- **Rendimiento**: canvas HTML5 para PAE grandes (matrices >500×500)

#### [MODIFY] [figures.py](file:///d:/alphafold-report/foldreport/figures.py)

- Mantener `plddt_figure()` como está (matplotlib PNG, funciona bien)
- Añadir `pae_data_for_js(pred)` → devuelve dict con la matriz PAE como lista, las posiciones de cadena, y metadata necesaria para el JS
- Modificar `make_figures()` para devolver también la clave `"pae_interactive"` con los datos JSON

#### [MODIFY] [builder.py](file:///d:/alphafold-report/foldreport/report/builder.py)

- En `_detail_card()`: reemplazar el `<img>` del PAE por un `<canvas>` con `data-pae` attribute conteniendo los datos JSON
- Pasar los datos PAE interactivos a cada tarjeta de detalle

#### [MODIFY] [template.html](file:///d:/alphafold-report/foldreport/report/template.html)

- Añadir JS para renderizar heatmaps en canvas:
  - Dibujar la matriz con colores mapeados
  - Superponer líneas de separación de cadenas
  - Manejar evento `mousemove` para tooltip
  - Tooltip flotante con `(residuo_i, residuo_j, error)`
- Añadir CSS para el tooltip y el canvas

---

### 3. Tabla filtrable

Añadir controles de filtrado sobre la tabla de ranking existente.

#### [MODIFY] [template.html](file:///d:/alphafold-report/foldreport/report/template.html)

- Añadir una barra de filtros encima de la tabla:
  - **Búsqueda por texto**: input que filtra por nombre de predicción
  - **Filtro por herramienta**: chips/botones toggle (ColabFold, AF3 Server, Boltz, All)
  - **Umbral de confianza**: slider o input numérico de confianza mínima
- JS que oculta/muestra filas `<tr>` según los filtros activos (combinables)
- CSS para los controles de filtro con el dark theme existente
- Contador de predicciones visibles vs. totales

---

### 4. Parser OpenFold3

OpenFold3 fue liberado en marzo 2026 (Apache 2.0, aqlaboratory/openfold-3). Su formato de salida es similar al AF3 Server pero con estructura de directorios propia.

**Formato de salida OpenFold3:**

```
output_dir/
└── <job_name>/
    ├── seed-<seed>_sample-<sample>/
    │   ├── <job_name>_model.cif          # estructura (pLDDT en B-factors)
    │   ├── <job_name>_confidences.json   # datos completos (atom_plddts, pae, pde)
    │   └── <job_name>_summary_confidences.json  # métricas escalares
    └── experiment_config.json
```

**Métricas disponibles:**

- `summary_confidences.json`: `plddt` (media), `ptm`, `iptm`, `ranking_score`, `has_clash`, `fraction_disordered`
- `confidences.json`: `atom_plddts` (per-atom), `pae` (NxN), `token_chain_ids`, `pde`
- pLDDT también en B-factors de la estructura

#### [NEW] [openfold3.py](file:///d:/alphafold-report/foldreport/parsers/openfold3.py)

- `can_handle()`: buscar directorios con patrón `seed-*_sample-*` que contengan `*_summary_confidences.json` + `*_confidences.json`
- `parse()`: por cada subdirectorio seed/sample:
  - Leer `*_summary_confidences.json` → pTM, ipTM, ranking_score, mean_plddt
  - Leer `*_confidences.json` → atom_plddts (agregar a per-residue via `aggregate_atom_plddt_to_residue` de base.py), pae (matriz NxN), token_chain_ids
  - Leer estructura CIF/PDB → chains, B-factor pLDDT como fallback
  - Devolver `list[Prediction]` con `source_tool="openfold3"`

> [!NOTE]
> Esto valida la función `aggregate_atom_plddt_to_residue` que ya existe en `base.py` pero que ningún parser usa actualmente — OpenFold3 reporta pLDDT per-atom, no per-residue.

#### [MODIFY] [parsers/\_\_init\_\_.py](file:///d:/alphafold-report/foldreport/parsers/__init__.py)

- Importar `OpenFold3Parser` y añadirlo a `ALL_PARSERS`

#### [NEW] [tests/data/openfold3/](file:///d:/alphafold-report/tests/data/openfold3/)

- Fixtures sintéticas fieles al layout de OpenFold3

#### [MODIFY] [make_fixtures.py](file:///d:/alphafold-report/tests/make_fixtures.py)

- Añadir `_generate_openfold3()` para crear las fixtures

#### [NEW] Tests en [test_other_parsers.py](file:///d:/alphafold-report/tests/test_other_parsers.py)

- Tests para `OpenFold3Parser`: can_handle, parse, métricas, pLDDT, PAE

---

### 5. Tests de edge cases

Añadir tests para casos límite no cubiertos actualmente.

#### [NEW] [test_edge_cases.py](file:///d:/alphafold-report/tests/test_edge_cases.py)

- **Archivos malformados**: JSON con claves faltantes, JSON inválido, PDB/CIF corrupto → verificar que el parser falla de forma limpia (excepción controlada o skip)
- **Proteína de una sola cadena**: predicción sin ipTM, una sola cadena → verificar que confidence_score funciona, informe se genera, métricas muestran N/A donde corresponde
- **pLDDT/PAE ausentes**: predicción sin datos de pLDDT o sin PAE → verificar que figuras devuelven None, informe muestra "not available"
- **Carpeta vacía**: directorio sin archivos reconocibles → verificar que `detect_parser()` devuelve None

#### [MODIFY] [make_fixtures.py](file:///d:/alphafold-report/tests/make_fixtures.py)

- Añadir generación de fixtures para edge cases:
  - `tests/data/single_chain/` — predicción ColabFold de una sola cadena
  - `tests/data/malformed/` — archivos con formato incorrecto
  - `tests/data/empty/` — directorio vacío

---

### 6. Fixtures mejoradas

Hacer las fixtures sintéticas más fieles a los datos reales.

#### [MODIFY] [make_fixtures.py](file:///d:/alphafold-report/tests/make_fixtures.py)

- Usar proteínas más grandes (56 residuos, como la demo, en lugar de 10)
- Añadir archivos auxiliares que las herramientas reales generan (log.txt, config, timings) para verificar que el parser los ignora correctamente
- Generar fixtures de single-chain para test de edge cases

---

## Verificación

### Tests automatizados

```bash
python -m pytest tests/ -v
```

- Los 17 tests existentes deben seguir pasando
- Los nuevos tests de edge cases deben pasar
- Test del report debe verificar que el PAE interactivo se genera (canvas element en el HTML)

### Verificación manual

- Abrir `examples/demo_report.html` y verificar:
  - Aparecen predicciones de ColabFold, AF3 Server y Boltz
  - El PAE es interactivo (hover muestra tooltip)
  - La tabla se puede filtrar por herramienta y por texto
  - La tabla se puede filtrar por umbral de confianza
  - Todo funciona offline
