import csv
import html
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "applications"
PLOTS = OUT / "plots"
PDF = OUT / "application_section_preview.pdf"


def esc(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def color(value):
    if not value or value == "none":
        return 0, 0, 0
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0, 0, 0


def num(value, default=0.0):
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value))
    return float(match.group(0)) if match else default


def ypdf(y, height):
    return height - y


def text_width(text, size):
    return 0.52 * size * len(text)


def svg_page(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = num(root.attrib.get("width"), 1200)
    height = num(root.attrib.get("height"), 800)
    cmds = []

    def stroke(elem):
        r, g, b = color(elem.attrib.get("stroke", "#000000"))
        cmds.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        cmds.append(f"{num(elem.attrib.get('stroke-width'), 1):.4f} w")

    def fill(elem):
        r, g, b = color(elem.attrib.get("fill", "#000000"))
        cmds.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag == "rect":
            if elem.attrib.get("fill") and elem.attrib.get("fill") != "none":
                fill(elem)
                x = num(elem.attrib.get("x"), 0)
                y = num(elem.attrib.get("y"), 0)
                w = num(elem.attrib.get("width"), width)
                h = num(elem.attrib.get("height"), height)
                cmds.append(f"{x:.4f} {ypdf(y + h, height):.4f} {w:.4f} {h:.4f} re f")
        elif tag == "line":
            stroke(elem)
            x1, y1 = num(elem.attrib.get("x1")), num(elem.attrib.get("y1"))
            x2, y2 = num(elem.attrib.get("x2")), num(elem.attrib.get("y2"))
            cmds.append(f"{x1:.4f} {ypdf(y1, height):.4f} m {x2:.4f} {ypdf(y2, height):.4f} l S")
        elif tag == "polyline":
            pts = []
            for point in elem.attrib.get("points", "").split():
                if "," in point:
                    x, y = point.split(",", 1)
                    pts.append((float(x), float(y)))
            if not pts:
                continue
            stroke(elem)
            parts = [f"{pts[0][0]:.4f} {ypdf(pts[0][1], height):.4f} m"]
            parts.extend(f"{x:.4f} {ypdf(y, height):.4f} l" for x, y in pts[1:])
            parts.append("S")
            cmds.append(" ".join(parts))
        elif tag == "text":
            text = html.unescape("".join(elem.itertext()).strip())
            if not text:
                continue
            size = num(elem.attrib.get("font-size"), 11)
            font = "F2" if elem.attrib.get("font-weight") in ("700", "bold") else "F1"
            x = num(elem.attrib.get("x"))
            y = num(elem.attrib.get("y"))
            anchor = elem.attrib.get("text-anchor")
            if anchor == "middle":
                x -= text_width(text, size) / 2
            elif anchor == "end":
                x -= text_width(text, size)
            r, g, b = color(elem.attrib.get("fill", "#000000"))
            cmds.extend(["BT", f"/{font} {size:.4f} Tf", f"{r:.4f} {g:.4f} {b:.4f} rg", f"1 0 0 1 {x:.4f} {ypdf(y, height):.4f} Tm", f"({esc(text)}) Tj", "ET"])
    return width, height, "\n".join(cmds)


def text_page(title, paragraphs, rows=None):
    width, height = 612, 792
    cmds = ["1 1 1 rg", f"0 0 {width} {height} re f"]
    y = 748
    cmds.extend(["BT", "/F2 18 Tf", "0 0 0 rg", f"1 0 0 1 54 {y} Tm", f"({esc(title)}) Tj", "ET"])
    y -= 32
    for para in paragraphs:
        for line in textwrap.wrap(para, width=92):
            cmds.extend(["BT", "/F1 10 Tf", "0 0 0 rg", f"1 0 0 1 54 {y} Tm", f"({esc(line)}) Tj", "ET"])
            y -= 14
        y -= 8
    if rows:
        y -= 6
        for row in rows:
            line = "   ".join(row)
            cmds.extend(["BT", "/F1 8 Tf", "0 0 0 rg", f"1 0 0 1 54 {y} Tm", f"({esc(line[:120])}) Tj", "ET"])
            y -= 11
            if y < 60:
                break
    return width, height, "\n".join(cmds)


def read_csv_rows(path, limit=None):
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    return rows if limit is None else rows[:limit]


def write_pdf(pages, path):
    objects = []
    kids = []
    for idx, (width, height, content) in enumerate(pages):
        page_obj = 3 + idx * 2
        content_obj = page_obj + 1
        kids.append(f"{page_obj} 0 R")
        stream = content.encode("latin-1", errors="replace")
        objects.append((page_obj, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.4f} {height:.4f}] /Resources << /Font << /F1 1 0 R /F2 2 0 R >> >> /Contents {content_obj} 0 R >>".encode()))
        objects.append((content_obj, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"))
    all_objects = [
        (1, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        (2, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
    ] + objects
    catalog_num = max(n for n, _ in all_objects) + 1
    pages_num = catalog_num + 1
    all_objects.append((catalog_num, f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()))
    all_objects.append((pages_num, f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()))
    all_objects.sort(key=lambda x: x[0])

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    for n, obj in all_objects:
        offsets[n] = len(out)
        out.extend(f"{n} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    max_obj = max(offsets)
    out.extend(f"xref\n0 {max_obj + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for n in range(1, max_obj + 1):
        out.extend(f"{offsets.get(n, 0):010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {max_obj + 1} /Root {catalog_num} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(out)


def main():
    pages = []
    pages.append(text_page(
        "Application Section Preview",
        [
            "This PDF previews the revised application section using PDF-compatible figures. The TeX source now includes PDF figures rather than SVG files.",
            "The local TeX Live installation could not run pdflatex because pdflatex.ini/pdflatex.fmt are missing. This preview PDF is generated independently as a working artifact.",
            "The empirical discussion uses the updated plug-in CLT variance and treats fixed alpha=tau comparisons as diagnostics. The main capital comparison matches VaR, XP, and ES through the gain-loss ratio.",
        ],
    ))
    pages.append(text_page("Descriptive Statistics", [], read_csv_rows(OUT / "descriptive_statistics.csv")))
    pages.append(text_page("Fixed-Level Diagnostics", [], read_csv_rows(OUT / "risk_backtest_summary.csv")))
    pages.append(text_page("Gain-Loss-Matched Buffers", [], read_csv_rows(OUT / "matched_gain_loss_summary.csv")))
    for name in ["price_paths.svg", "drawdowns.svg", "risk_forecasts_BTC.svg", "risk_forecasts_SPX.svg", "omega_at_var.svg", "alpha_at_expectile.svg"]:
        pages.append(svg_page(PLOTS / name))
    write_pdf(pages, PDF)
    print(PDF)


if __name__ == "__main__":
    main()
