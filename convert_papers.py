"""
convert_papers.py

Converts a folder of PDFs to markdown using marker-pdf.
Outputs clean markdown that can be fed as context to a coding model.

Usage:
    python convert_papers.py --input ./papers --output ./converted

    # Single file:
    python convert_papers.py --input ./papers/zhu_2024.pdf --output ./converted

    # With figure extraction (saves figure images alongside markdown):
    python convert_papers.py --input ./papers --output ./converted --figures

Dependencies:
    pip install marker-pdf

Notes:
    - First run downloads surya-ocr model weights (~1–2 GB). Subsequent runs are fast.
    - GPU strongly recommended for speed; CPU works but is slow (~2–5 min/paper).
    - Output is one .md file per PDF, plus optional figure images.
    - Equations are rendered as LaTeX inside $...$ blocks.
    - Tables are preserved as markdown tables.
"""

import argparse
import sys
import time
from pathlib import Path


def load_models():
    """Load marker model weights. Downloads on first run, cached thereafter."""
    print("Loading marker models (downloads on first run, ~1-2 GB)...")
    from marker.models import create_model_dict
    models = create_model_dict()
    print("Models loaded.\n")
    return models


def convert_pdf(pdf_path: Path, output_dir: Path, models: dict, extract_figures: bool = False) -> Path:
    """
    Convert a single PDF to markdown.

    Args:
        pdf_path:        Path to the input PDF.
        output_dir:      Directory to write the output markdown file.
        models:          Marker model dict from create_model_dict().
        extract_figures: If True, save extracted figure images alongside the markdown.

    Returns:
        Path to the output markdown file.
    """
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered

    print(f"Converting: {pdf_path.name}")
    t0 = time.time()

    # Build converter
    converter = PdfConverter(artifact_dict=models, config={"use_llm": False})

    # Run conversion
    rendered = converter(str(pdf_path))

    # Extract markdown text
    markdown, _, images = text_from_rendered(rendered)

    # Write markdown output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_md = output_dir / (pdf_path.stem + ".md")
    out_md.write_text(markdown, encoding="utf-8")

    # Optionally save extracted figures
    if extract_figures and images:
        fig_dir = output_dir / (pdf_path.stem + "_figures")
        fig_dir.mkdir(exist_ok=True)
        for img_name, img in images.items():
            img_path = fig_dir / img_name
            img.save(str(img_path))
        print(f"  Saved {len(images)} figures → {fig_dir.relative_to(output_dir)}/")

    elapsed = time.time() - t0
    size_kb = out_md.stat().st_size / 1024
    print(f"  Done in {elapsed:.1f}s → {out_md.name} ({size_kb:.0f} KB)")

    return out_md


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs to markdown using marker-pdf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a PDF file, or a directory containing PDF files."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for converted markdown files."
    )
    parser.add_argument(
        "--figures", "-f", action="store_true",
        help="Extract and save figure images alongside the markdown."
    )
    parser.add_argument(
        "--pattern", "-p", default="*.pdf",
        help="Glob pattern when --input is a directory (default: '*.pdf')."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Collect PDFs to convert
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"Error: {input_path} is not a PDF file.", file=sys.stderr)
            sys.exit(1)
        pdfs = [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob(args.pattern))
        if not pdfs:
            print(f"No files matching '{args.pattern}' found in {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) to convert.\n")

    # Load models once, reuse across all files
    models = load_models()

    # Convert each PDF
    converted = []
    failed = []
    for pdf in pdfs:
        try:
            out = convert_pdf(pdf, output_dir, models, extract_figures=args.figures)
            converted.append(out)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append((pdf, e))

    # Summary
    print(f"\n{'='*50}")
    print(f"Converted {len(converted)}/{len(pdfs)} files → {output_dir}/")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for pdf, err in failed:
            print(f"  {pdf.name}: {err}")


if __name__ == "__main__":
    main()
