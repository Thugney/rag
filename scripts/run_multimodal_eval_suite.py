from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunker import SmartChunker  # noqa: E402
from scripts.generate_multimodal_eval_fixtures import DEFAULT_OUTPUT_DIR, generate_fixtures  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate multimodal fixtures, run parser smoke checks, and optionally execute retrieval evaluation."
    )
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "evals" / "multimodal_retrieval_eval.json"),
        help="Dataset JSON to execute after fixture generation.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only generate fixtures and run parser smoke checks.",
    )
    return parser.parse_known_args()


def run_parser_smoke() -> None:
    chunker = SmartChunker()
    fixture_dir = DEFAULT_OUTPUT_DIR

    checks = [
        {
            "path": fixture_dir / "invoice_ocr.jpg",
            "parser_name": "image",
            "content_type": "image",
            "metadata_keys": ["ocr_source", "ocr_engine", "image_name"],
        },
        {
            "path": fixture_dir / "invoice_scan.pdf",
            "parser_name": "pdf",
            "content_type": "image",
            "metadata_keys": ["ocr_source", "image_index", "page_number"],
        },
        {
            "path": fixture_dir / "risk_register.xlsx",
            "parser_name": "spreadsheet",
            "content_type": "table",
            "metadata_keys": ["sheet_name", "row_start", "row_end"],
        },
        {
            "path": fixture_dir / "security_controls.csv",
            "parser_name": "spreadsheet",
            "content_type": "table",
            "metadata_keys": ["delimiter", "row_start", "row_end"],
        },
    ]

    for check in checks:
        parsed = chunker.parse_document(str(check["path"]))
        if not parsed.elements:
            raise RuntimeError(f"No parsed elements were produced for {check['path'].name}")

        matching_elements = [
            element
            for element in parsed.elements
            if element.parser_name == check["parser_name"] and element.content_type == check["content_type"]
        ]
        if not matching_elements:
            raise RuntimeError(
                f"{check['path'].name} did not produce an element with parser={check['parser_name']} "
                f"and content_type={check['content_type']}"
            )

        metadata = matching_elements[0].metadata
        missing_keys = [key for key in check["metadata_keys"] if key not in metadata]
        if missing_keys:
            raise RuntimeError(f"{check['path'].name} is missing expected metadata keys: {', '.join(missing_keys)}")


def main() -> int:
    args, passthrough = parse_args()
    generate_fixtures(DEFAULT_OUTPUT_DIR)
    run_parser_smoke()
    print("Parser smoke checks passed for multimodal fixtures.")

    if args.skip_retrieval:
        return 0

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_retrieval_eval.py"),
        "--dataset",
        str(Path(args.dataset).resolve()),
        *passthrough,
    ]
    result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
