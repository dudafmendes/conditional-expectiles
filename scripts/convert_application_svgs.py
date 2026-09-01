import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_DIR = ROOT / "output" / "applications" / "plots"


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


def text_runs(text, base_font):
    """Emit Greek alpha/tau through the built-in Symbol font; keep other text in Helvetica."""
    symbol_map = {"α": "a", "τ": "t"}
    runs = []
    active_font = None
    active_text = []
    for char in text:
        font = "Symbol" if char in symbol_map else base_font
        value = symbol_map.get(char, char)
        if active_font is not None and font != active_font:
            runs.append((active_font, "".join(active_text)))
            active_text = []
        active_font = font
        active_text.append(value)
    if active_text:
        runs.append((active_font, "".join(active_text)))
    return runs


def render_svg_to_pdf(svg_path, pdf_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width, height = extract_size(root)
    commands = []
    opacity_names = {}

    def set_opacity(elem):
        value = number(elem.attrib.get("opacity"), 1.0)
        value = max(0.0, min(1.0, value))
        if value == 1.0:
            commands.append("/GS0 gs")
            return
        if value not in opacity_names:
            opacity_names[value] = f"GS{len(opacity_names) + 1}"
        commands.append(f"/{opacity_names[value]} gs")

    def set_stroke(elem):
        r, g, b = parse_color(elem.attrib.get("stroke", "#000000"))
        lw = number(elem.attrib.get("stroke-width"), 1.0)
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG")
        commands.append(f"{lw:.4f} w")
        set_opacity(elem)

    def set_fill(elem):
        r, g, b = parse_color(elem.attrib.get("fill", "#000000"))
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        set_opacity(elem)

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
            stroke = elem.attrib.get("stroke")
            if stroke and stroke != "none":
                set_stroke(elem)
                x = number(elem.attrib.get("x"), 0)
                y = number(elem.attrib.get("y"), 0)
                w = number(elem.attrib.get("width"), width)
                h = number(elem.attrib.get("height"), height)
                commands.append(f"{x:.4f} {y_pdf(y + h, height):.4f} {w:.4f} {h:.4f} re S")
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
        elif tag == "polygon":
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
            set_fill(elem)
            first = pairs[0]
            parts = [f"{first[0]:.4f} {y_pdf(first[1], height):.4f} m"]
            parts.extend(f"{x:.4f} {y_pdf(y, height):.4f} l" for x, y in pairs[1:])
            parts.append("h f")
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
            commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
            x_cursor = x
            for run_font, run_text in text_runs(text, font):
                commands.append(f"/{run_font} {size:.4f} Tf")
                commands.append(f"1 0 0 1 {x_cursor:.4f} {y_pdf(y, height):.4f} Tm")
                commands.append(f"({pdf_escape(run_text)}) Tj")
                x_cursor += text_width(run_text, size)
            commands.append("ET")

    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"")  # The page resource dictionary is assembled after opacity states are known.
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>")
    objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    ext_states = [("GS0", "<< /Type /ExtGState /ca 1 /CA 1 >>")]
    ext_states.extend((name, f"<< /Type /ExtGState /ca {opacity:.4f} /CA {opacity:.4f} >>") for opacity, name in opacity_names.items())
    state_object_ids = []
    for _, state in ext_states:
        state_object_ids.append(len(objects) + 1)
        objects.append(state.encode("ascii"))
    state_resources = " ".join(f"/{name} {object_id} 0 R" for (name, _), object_id in zip(ext_states, state_object_ids))
    page = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.4f} {height:.4f}] "
        f"/Resources << /Font << /Helvetica 4 0 R /Helvetica-Bold 5 0 R /Symbol 6 0 R >> /ExtGState << {state_resources} >> >> "
        "/Contents 7 0 R >>"
    ).encode("ascii")
    objects[2] = page

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
    if len(sys.argv) == 3 and sys.argv[1] == "--directory":
        plot_dir = Path(sys.argv[2]).resolve()
    elif len(sys.argv) == 1:
        plot_dir = DEFAULT_PLOT_DIR
    else:
        raise SystemExit("Usage: python scripts/convert_application_svgs.py [--directory PATH]")
    for svg in sorted(plot_dir.glob("*.svg")):
        pdf = svg.with_suffix(".pdf")
        render_svg_to_pdf(svg, pdf)
        print(f"{svg.name} -> {pdf.name}")


if __name__ == "__main__":
    main()
