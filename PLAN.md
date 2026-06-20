# FoldReport — Especificación de proyecto

> Brief para construir el proyecto con Claude Code. Pásale este fichero como contexto
> inicial. Cada fase tiene criterios de aceptación claros; trabájalas en orden y no
> empieces una hasta que la anterior pase sus criterios.

---

## 1. Visión en una frase

Apuntas la herramienta a una carpeta de predicciones de estructura —vengan de **ColabFold**,
del **AlphaFold 3 Server**, de **Boltz** o de **OpenFold3**— y obtienes un **único informe HTML
autocontenido** que rankea todas las predicciones por confianza y deja explorar cada una
(PAE interactivo, pLDDT por residuo, métricas de interfaz) sin instalar nada más ni abrir un notebook.

**La cuña que nadie cubre:** multi-herramienta + por lotes + informe navegable en un solo archivo.
Mantén esta frase como filtro: si una feature no sirve a "unificar salidas de varias
herramientas en un informe compartible", probablemente sobra para el MVP.

---

## 2. El problema (por qué existe esto)

- Ejecutar AlphaFold ya está resuelto (ColabFold, el AF3 Server). El cuello de botella se movió
  al **después**: tienes decenas o cientos de carpetas de salida y necesitas decidir *qué mirar*.
- Cada herramienta moderna escupe formatos y métricas ligeramente distintos (pLDDT, PAE, pTM,
  ipTM, mpDockQ) con estructuras de carpeta diferentes. Comparar entre ellas hoy es manual.
- Las herramientas existentes cubren piezas sueltas: hacen figuras de una predicción cada vez,
  o requieren un visor pesado, o solo entienden un formato. Ninguna resuelve
  "300 salidas de 3 herramientas, decididme cuáles importan".

## 3. Usuario objetivo

Biólogo estructural / bioinformático que ya tiene salidas de predicción y necesita triarlas e
interpretarlas rápido. Asume que sabe qué es pLDDT y PAE pero no quiere escribir scripts de
parsing cada vez.

---

## 4. Alcance del MVP

### DENTRO (v0.1 – v1.0)
- Ingesta de una carpeta (o varias) con predicciones.
- Parseo robusto de **ColabFold** primero; **AF3 Server** y **Boltz** después.
- Normalización a una representación interna común (ver §6).
- Tabla de métricas ordenable/filtrable: una fila por predicción/modelo.
- Figuras de calidad de publicación: PAE y pLDDT por residuo.
- Informe HTML **autocontenido** (un solo .html, sin dependencias externas en runtime) con
  visor 3D embebido y ranking por confianza.
- CLI de un comando: `foldreport <carpeta> -o informe.html`.

### FUERA (explícitamente, para no dispersarse)
- Ejecutar AlphaFold / inferencia (cero GPU, cero modelo). Solo se procesan salidas existentes.
- Predicción de efecto de mutaciones / variantes.
- Edición o reparación de estructuras.
- Backend con servidor, base de datos o cuentas de usuario. El entregable es CLI + HTML estático.
- Soporte de formatos legacy raros. Empieza por lo que la comunidad usa hoy.

---

## 5. Stack técnico

- **Lenguaje:** Python 3.10+.
- **Parsing estructura (mmCIF/PDB):** `gemmi` (preferido) o `biotite`.
- **Datos/tablas:** `pandas`.
- **Figuras estáticas:** `matplotlib`.
- **Visor 3D embebido en navegador (sin servidor):** `py3Dmol` o Mol*.
- **CLI:** `click` o `typer`.
- **Empaquetado:** `pyproject.toml`, instalable con `pip install .`. Objetivo: `pip install foldreport`.
- **Tests:** `pytest`.

Mantén las dependencias mínimas. Cada dependencia nueva es fricción de instalación, y la
instalación de un comando es clave para la adopción.

---

## 6. Diseño de la representación interna (el corazón del proyecto)

El error más caro sería acoplar el código a un formato concreto. Define **primero** una capa
intermedia limpia y haz que cada parser produzca exactamente esto. Así añadir una herramienta
nueva es escribir un parser, no tocar el resto.

Estructuras sugeridas (ajústalas, pero respeta el principio):

```python
@dataclass
class Prediction:
    name: str                      # identificador legible
    source_tool: str               # "colabfold" | "af3_server" | "boltz" | ...
    structure_path: Path           # ruta al .cif/.pdb
    chains: list[Chain]
    plddt: list[float]             # por residuo, orden canónico
    pae: np.ndarray | None         # matriz NxN o None si no hay
    metrics: PredictionMetrics
    raw_files: dict[str, Path]     # trazabilidad de dónde salió cada cosa

@dataclass
class PredictionMetrics:
    mean_plddt: float
    ptm: float | None
    iptm: float | None
    mpdockq: float | None
    n_chains: int
    n_residues: int
    # rellena None lo que la herramienta no provea; el informe debe tolerar huecos

# Contrato de parser: Path de carpeta -> list[Prediction]
class Parser(Protocol):
    def can_handle(self, path: Path) -> bool: ...
    def parse(self, path: Path) -> list[Prediction]: ...
```

Principios:
- Todo parser devuelve `list[Prediction]`. Nada aguas abajo conoce el formato original.
- Las métricas ausentes son `None`, nunca inventadas. El informe muestra "N/A".
- Detección automática de formato vía `can_handle()`; el usuario no debería tener que declararlo.

---

## 7. Estructura de ficheros sugerida

```
foldreport/
├── pyproject.toml
├── README.md
├── foldreport/
│   ├── __init__.py
│   ├── cli.py
│   ├── models.py            # dataclasses de §6
│   ├── parsers/
│   │   ├── __init__.py      # registro + autodetección
│   │   ├── base.py          # Protocol/ABC del parser
│   │   ├── colabfold.py
│   │   ├── af3_server.py
│   │   └── boltz.py
│   ├── metrics.py           # cálculo/normalización de métricas
│   ├── figures.py           # PAE, pLDDT (matplotlib)
│   └── report/
│       ├── builder.py       # ensambla el HTML
│       └── template.html    # plantilla autocontenida
├── tests/
│   ├── data/                # fixtures mínimas reales de cada herramienta
│   └── test_*.py
└── examples/
    └── demo/                # carpeta de ejemplo lista para `foldreport examples/demo`
```

---

## 8. Hoja de ruta por fases (con criterios de aceptación)

### Fase 1 — Parser ColabFold sólido (semanas 1–2)
- Implementa `models.py` y el `Parser` base.
- Implementa el parser de ColabFold completo.
- **Aceptación:** dada una carpeta real de ColabFold, `parse()` devuelve `Prediction`s con
  estructura, pLDDT, PAE y métricas correctas. Test con fixture real en `tests/data/`.

### Fase 2 — Figuras y tabla de métricas (semanas 3–4)
- `figures.py`: gráfico de PAE y de pLDDT por residuo, calidad de publicación.
- `metrics.py`: tabla normalizada (pandas) ordenable por cualquier métrica.
- **Aceptación:** desde una lista de `Prediction` se generan los PNG/figuras y un DataFrame
  con una fila por predicción. Huecos como `None` se manejan sin romper.

### Fase 3 — Informe HTML autocontenido (semanas 5–6)
- Visor 3D embebido (py3Dmol/Mol*) coloreado por pLDDT.
- Plantilla que junta: tabla rankeada arriba, detalle por predicción debajo.
- **Aceptación:** `foldreport <carpeta> -o informe.html` produce **un solo .html** que abre en
  el navegador sin conexión y sin ficheros adyacentes, con ranking por confianza funcional.

### Fase 4 — Segundo y tercer formato + pulido (semanas 7–8)
- Añade parser de AF3 Server (y Boltz si da tiempo). Esto valida que la abstracción de §6 aguanta.
- README con ejemplo reproducible copy-paste, datos de prueba en el repo, `pip install` de un comando.
- **Aceptación:** la misma orden funciona sobre carpetas de ≥2 herramientas distintas sin cambios,
  produciendo informes equivalentes. Un usuario nuevo logra un informe en <5 min desde el README.

---

## 9. Qué hace que se adopte (prioridad nº1 del autor)

Lo que separa la herramienta que la gente usa de la que muere en GitHub casi nunca es el código:

1. **Instalación de un comando** (`pip install foldreport`).
2. **README con un ejemplo que funciona copiando y pegando**, con datos incluidos en el repo.
3. **Resolver UN caso de uso del todo** antes que cinco a medias.
4. Coste de probar ≈ cero: apuntar a una carpeta y obtener algo útil en segundos.
5. Cuando esté maduro: preprint corto en bioRxiv y difusión donde está la comunidad estructural.

Regla de oro de scope: ante cualquier feature nueva, pregúntate si sirve a la frase de §1.
Si no, va al backlog, no al MVP.

---

## 10. Riesgo conocido

El espacio de análisis post-AlphaFold tiene varios actores (herramientas que hacen figuras de PAE,
visores de una predicción, plugins de visores pesados). Si esto es "una más que dibuja PAE", se
pierde. La defensa es la cuña: **unificar salidas de varias herramientas en un informe que se
comparte como un solo archivo.** Esa combinación es lo que hoy no existe.

---

## 11. Primeras instrucciones para Claude Code

1. Crea el esqueleto del repo según §7 con `pyproject.toml` instalable.
2. Implementa `models.py` (§6) y `parsers/base.py` con el contrato `Parser`.
3. Implementa `parsers/colabfold.py` y un test con una fixture mínima en `tests/data/colabfold/`.
   (Si no tienes una salida real a mano, genera una fixture sintética fiel al formato real de
   ColabFold y deja un TODO para sustituirla por una real.)
4. No avances a figuras hasta que el parser pase su test.