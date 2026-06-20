"""(B) submission-ready figures: shared PAE scale, vector export, colorblind palette."""

from __future__ import annotations

from foldreport import figures
from foldreport.parsers import parse_folder
from foldreport.report import build_report


def test_pae_uses_one_shared_scale_across_predictions(colabfold_dir):
    """Every PAE heatmap colours on the same ceiling, not its own max.

    A per-figure ``vmax`` is exactly the dishonest comparison we want to avoid: a darker
    panel must always mean lower error.
    """
    preds = parse_folder(colabfold_dir)
    datas = [figures.pae_data_for_js(p) for p in preds]
    maxes = {d["max_val"] for d in datas}
    assert maxes == {figures.PAE_MAX_ANGSTROM}
    # The true per-matrix maxima still differ (so the shared scale is doing real work).
    assert len({d["data_max"] for d in datas}) > 1


def test_pae_scale_is_overridable(colabfold_dir):
    pred = parse_folder(colabfold_dir)[0]
    assert figures.pae_data_for_js(pred, vmax=20.0)["max_val"] == 20.0
    assert figures.build_pae_figure(pred, vmax=20.0) is not None


def test_save_publication_figures_writes_vector_and_raster(tmp_path, colabfold_dir):
    pred = parse_folder(colabfold_dir)[0]
    written = figures.save_publication_figures(
        pred, tmp_path, dpi=300, formats=("pdf", "svg", "png")
    )
    # pLDDT + PAE in three formats each.
    assert len(written) == 6
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    suffixes = sorted({p.suffix for p in written})
    assert suffixes == [".pdf", ".png", ".svg"]
    # Vector PDF carries the PDF magic; embedded-font config does not break it.
    pdf = next(p for p in written if p.suffix == ".pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_publication_downloads_embed_as_decodable_data_uris(colabfold_dir):
    """Both figures are downloadable from the HTML as self-contained PNG + PDF URIs."""
    import base64

    pred = parse_folder(colabfold_dir)[0]
    downloads = figures.publication_downloads(pred)
    assert set(downloads) == {"plddt", "pae"}
    for fmt_uris in downloads.values():
        assert set(fmt_uris) == {"png", "pdf"}
        png = base64.b64decode(fmt_uris["png"].split(",", 1)[1])
        pdf = base64.b64decode(fmt_uris["pdf"].split(",", 1)[1])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert pdf[:4] == b"%PDF"


def test_report_offers_per_figure_downloads(tmp_path, colabfold_dir):
    preds = parse_folder(colabfold_dir)
    out = tmp_path / "dl.html"
    build_report(preds, out)
    text = out.read_text(encoding="utf-8")
    assert 'download="' in text
    assert "data:application/pdf;base64," in text
    # One PNG + one PDF link per figure, two figures per prediction.
    assert text.count('class="dl"') == len(preds) * 2 * 2


def test_save_publication_figures_skips_absent_metrics(tmp_path):
    from foldreport.models import Prediction, PredictionMetrics

    pred = Prediction(
        name="bare",
        source_tool="test",
        structure_path=__file__,
        plddt=[],
        pae=None,
        metrics=PredictionMetrics(),
    )
    assert figures.save_publication_figures(pred, tmp_path) == []


def test_colorblind_palette_differs_and_flows_into_report(tmp_path, colabfold_dir):
    default = figures.plddt_bands(colorblind=False)
    cb = figures.plddt_bands(colorblind=True)
    assert default != cb
    assert len(default) == len(cb) == 4

    preds = parse_folder(colabfold_dir)
    out = tmp_path / "cb.html"
    build_report(preds, out, colorblind=True)
    text = out.read_text(encoding="utf-8")
    # A colorblind-safe (Okabe-Ito) color reaches the legend / 3D viewer palette.
    assert "#0072B2" in text
    # The default AlphaFold "very high" blue is no longer injected as a band color.
    assert '"#0053D6"' not in text
