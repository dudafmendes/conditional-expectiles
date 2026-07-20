import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "output" / "applications" / "plots"


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def parse_color(value):
    if not value or value == "none":
        return (0.0, 0.0, 0.0)
    value = value.strip()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (0.0, 0.0, 0.0)


def number(value, default=0.0):
    if value is None:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", str(value))
    return float(match.group(0)) if match else default


def extract_size(root):
    width = number(root.attrib.get("width"), 1200)
    height = number(root.attrib.get("height"), 800)
    return width, height


def y_pdf(y, height):
    return height - y


def style_text(elem):
    size = number(elem.attrib.get("font-size"), 11)
    weight = elem.attrib.get("font-weight", "")
    font = "Helvetica-Bold" if weight in ("700", "bold") else "Helvetica"
    fill = parse_color(elem.attrib.get("fill", "#000000"))
    anchor = elem.attrib.get("text-anchor", "start")
    return size, font, fill, anchor


def text_width(text, size):
    return 0.52 * size * len(text)


def render_svg_to_pdf(svg_path, pdf_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width, height = extract_size(root)
    commands = []

    def set_stroke(elem):
        r, g, b = parse_color(elem.attrib.get("stroke", "#000000"))
        lw = number(elem.attrib.get("stroke-width"), 1.0)
        opacity = number(elem.attrib.get("opacity"), 1.0)
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        commands.append(f"{lw:.4f} w")
        if opacity < 1:
            commands.append(f"% opacity {opacity:.4f} ignored")

    def set_fill(elem):
        r, g, b = parse_color(elem.attrib.get("fill", "#000000"))
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag == "rect":
            fill = elem.attrib.get("fill")
            if fill and fill != "none":
                set_fill(elem)
                x = number(elem.attrib.get("x"), 0)
                y = number(elem.attrib.get("y"), 0)
                w = number(elem.attrib.get("width"), width)
                h = number(elem.attrib.get("height"), height)
                commands.append(f"{x:.4f} {y_pdf(y + h, height):.4f} {w:.4f} {h:.4f} re f")
        elif tag == "line":
            set_stroke(elem)
            x1 = number(elem.attrib.get("x1"))
            y1 = number(elem.attrib.get("y1"))
            x2 = number(elem.attrib.get("x2"))
            y2 = number(elem.attrib.get("y2"))
            commands.append(f"{x1:.4f} {y_pdf(y1, height):.4f} m {x2:.4f} {y_pdf(y2, height):.4f} l S")
        elif tag == "polyline":
            pts = elem.attrib.get("points", "").strip()
            if not pts:
                continue
            pairs = []
            for pt in pts.split():
                if "," not in pt:
                    continue
                x, y = pt.split(",", 1)
                pairs.append((float(x), float(y)))
            if not pairs:
                continue
            set_stroke(elem)
            first = pairs[0]
            parts = [f"{first[0]:.4f} {y_pdf(first[1], height):.4f} m"]
            parts.extend(f"{x:.4f} {y_pdf(y, height):.4f} l" for x, y in pairs[1:])
            parts.append("S")
            commands.append(" ".join(parts))
        elif tag == "text":
            text = html.unescape("".join(elem.itertext()).strip())
            if not text:
                continue
            size, font, fill, anchor = style_text(elem)
            x = number(elem.attrib.get("x"))
            y = number(elem.attrib.get("y"))
            if anchor == "middle":
                x -= text_width(text, size) / 2
            elif anchor == "end":
                x -= text_width(text, size)
            r, g, b = fill
            commands.append("BT")
            commands.append(f"/{font} {size:.4f} Tf")
            commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
            commands.append(f"1 0 0 1 {x:.4f} {y_pdf(y, height):.4f} Tm")
            commands.append(f"({pdf_escape(text)}) Tj")
            commands.append("ET")

    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    page = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.4f} {height:.4f}] "
        "/Resources << /Font << /Helvetica 4 0 R /Helvetica-Bold 5 0 R >> >> "
        "/Contents 6 0 R >>"
    ).encode("ascii")
    objects.append(page)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    output = bytearray()
    output.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{i} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    pdf_path.write_bytes(output)


def main():
    for svg in sorted(PLOT_DIR.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        render_svg_to_pdf(svg, pdf)
        print(f"{svg.name} -> {pdf.name}")


if __name__ == "__main__":
    main()
