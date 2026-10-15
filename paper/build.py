from pathlib import Path
import shutil
import subprocess
import tempfile

from pypdf import PdfReader


PAPER_DIR = Path(__file__).resolve().parent
SOURCE_NAME = "Paper_Equipo-QG_TP_IA.tex"
INPUT_FILES = (
    SOURCE_NAME,
    "llncs.cls",
    "splncs04.bst",
    "refs.bib",
)
MAX_PAGES = 12


def build_paper():
    with tempfile.TemporaryDirectory(prefix="lncs-paper-") as temporary_directory:
        build_dir = Path(temporary_directory)
        for filename in INPUT_FILES:
            shutil.copy2(PAPER_DIR / filename, build_dir / filename)

        subprocess.run(["tectonic", SOURCE_NAME], cwd=build_dir, check=True)

        built_pdf = build_dir / Path(SOURCE_NAME).with_suffix(".pdf").name
        page_count = len(PdfReader(built_pdf).pages)
        if not 0 < page_count <= MAX_PAGES:
            raise ValueError(f"Paper must have 1-{MAX_PAGES} pages; got {page_count}.")

        output_pdf = PAPER_DIR / built_pdf.name
        shutil.copy2(built_pdf, output_pdf)

    print(f"Generated {output_pdf} ({page_count} pages)")


if __name__ == "__main__":
    build_paper()
