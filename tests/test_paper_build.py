import re
import subprocess
import sys
import unittest
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD_SCRIPT = PAPER / "build.py"
PDF = PAPER / "Paper_Equipo-QG_TP_IA.pdf"
LEGACY_AUXILIARIES = (
    "Paper_Equipo-QG_TP_IA.aux",
    "Paper_Equipo-QG_TP_IA.bbl",
    "Paper_Equipo-QG_TP_IA.blg",
    "Paper_Equipo-QG_TP_IA.log",
)


class PaperBuildTest(unittest.TestCase):
    def test_build_writes_lncs_pdf_and_preserves_existing_auxiliaries(self):
        existing = {
            PAPER / name: (PAPER / name).read_bytes()
            for name in LEGACY_AUXILIARIES
            if (PAPER / name).is_file()
        }

        subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=ROOT, check=True)

        self.assertTrue(PDF.is_file())
        reader = PdfReader(PDF)
        self.assertGreater(len(reader.pages), 0)
        self.assertLessEqual(len(reader.pages), 12)
        text = " ".join(" ".join(page.extract_text() or "" for page in reader.pages).split())
        self.assertIn("Liu", text)
        self.assertIn("Breiman", text)
        self.assertIn("scikit-learn", text)
        authors_and_legajos = (
            ("Laura Cornaló Bassan", "00000"),
            ("Matías Fernandez", "24282"),
            ("Mariano Insaurralde", "24684"),
            ("Philippe Maurel Garcete", "25613"),
            ("Franco Sanchez", "27310"),
        )
        for author, legajo in authors_and_legajos:
            self.assertRegex(text, re.escape(author) + r"\s*1\s*\[" + legajo + r"\]")
        self.assertNotIn("et al.", text)
        for path, contents in existing.items():
            self.assertEqual(path.read_bytes(), contents, msg=f"Modified {path.name}")


if __name__ == "__main__":
    unittest.main()
