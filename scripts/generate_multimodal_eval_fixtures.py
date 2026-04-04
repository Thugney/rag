from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "fixtures" / "multimodal"

IMAGE_WIDTH = 1400
IMAGE_HEIGHT = 900
IMAGE_LINES = [
    "Invoice 4821",
    "Total Due NOK 120000",
    "Payment due 14 April 2026",
    "Approval owner Finance team",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multimodal evaluation fixtures for OCR, scanned PDF, and spreadsheet regression tests."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Target directory for generated fixtures.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_ocr_image(output_path: Path, lines: Sequence[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="rag-fixture-image-") as temp_root_raw:
        temp_root = Path(temp_root_raw)
        script_path = temp_root / "render_eval_image.ps1"
        script_path.write_text(
            """
param(
  [string]$OutputPath,
  [int]$Width,
  [int]$Height
)
Add-Type -AssemblyName System.Drawing
$bitmap = New-Object System.Drawing.Bitmap $Width, $Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.Clear([System.Drawing.Color]::White)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$titleFont = New-Object System.Drawing.Font("Arial", 42, [System.Drawing.FontStyle]::Bold)
$bodyFont = New-Object System.Drawing.Font("Arial", 28, [System.Drawing.FontStyle]::Regular)
$brush = [System.Drawing.Brushes]::Black
$graphics.DrawString("Invoice 4821", $titleFont, $brush, 70, 90)
$graphics.DrawString("Total Due NOK 120000", $bodyFont, $brush, 70, 220)
$graphics.DrawString("Payment due 14 April 2026", $bodyFont, $brush, 70, 290)
$graphics.DrawString("Approval owner Finance team", $bodyFont, $brush, 70, 360)
$bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$titleFont.Dispose()
$bodyFont.Dispose()
$graphics.Dispose()
$bitmap.Dispose()
            """.strip(),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-File",
                str(script_path),
                str(output_path),
                str(IMAGE_WIDTH),
                str(IMAGE_HEIGHT),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to render OCR image fixture: {result.stderr or result.stdout}")


def create_scanned_pdf(image_path: Path, output_path: Path) -> None:
    image_bytes = image_path.read_bytes()
    page_width = 612.0
    page_height = 792.0
    draw_width = 540.0
    draw_height = draw_width * (IMAGE_HEIGHT / IMAGE_WIDTH)
    draw_x = 36.0
    draw_y = (page_height - draw_height) / 2

    image_object = (
        f"<< /Type /XObject /Subtype /Image /Width {IMAGE_WIDTH} /Height {IMAGE_HEIGHT} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(image_bytes)} >>\n"
    ).encode("ascii")
    content_stream = (
        f"q\n{draw_width:.2f} 0 0 {draw_height:.2f} {draw_x:.2f} {draw_y:.2f} cm\n/Im0 Do\nQ\n"
    ).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ),
        image_object + b"stream\n" + image_bytes + b"\nendstream",
        f"<< /Length {len(content_stream)} >>\n".encode("ascii") + b"stream\n" + content_stream + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    output_path.write_bytes(pdf)


def create_csv_fixture(output_path: Path) -> None:
    output_path.write_text(
        "\n".join(
            [
                "Control,Owner,Status",
                "Token rotation,Maria,Complete",
                "Third-party review,Omar,Blocked",
                "DPO sign-off,Elin,In Progress",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def create_xlsx_fixture(output_path: Path) -> None:
    rows = [
        ["Control", "Owner", "Status"],
        ["MFA rollout", "Alice", "In Progress"],
        ["Backup review", "Bob", "Complete"],
        ["Vendor review", "Dan", "Blocked"],
    ]

    worksheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row, start=1):
            cell_reference = f"{column_name(column_index)}{row_index}"
            safe_value = escape_xml(value)
            cells.append(
                f'<c r="{cell_reference}" t="inlineStr"><is><t>{safe_value}</t></is></c>'
            )
        worksheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(worksheet_rows)
        + "</sheetData></worksheet>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Risk Register" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def column_name(index: int) -> str:
    name = ""
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def escape_xml(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_manifest(output_dir: Path) -> None:
    manifest = {
        "generated_at": None,
        "fixtures": {
            "invoice_ocr_image": "invoice_ocr.jpg",
            "invoice_scan_pdf": "invoice_scan.pdf",
            "risk_register_xlsx": "risk_register.xlsx",
            "security_controls_csv": "security_controls.csv",
        },
    }
    (output_dir / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_fixtures(output_dir: Path) -> Path:
    output_dir = ensure_output_dir(output_dir.resolve())
    create_ocr_image(output_dir / "invoice_ocr.jpg", IMAGE_LINES)
    create_scanned_pdf(output_dir / "invoice_ocr.jpg", output_dir / "invoice_scan.pdf")
    create_xlsx_fixture(output_dir / "risk_register.xlsx")
    create_csv_fixture(output_dir / "security_controls.csv")
    write_manifest(output_dir)
    return output_dir


def main() -> int:
    args = parse_args()
    output_dir = generate_fixtures(Path(args.output_dir))

    print(f"Generated multimodal fixtures in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
