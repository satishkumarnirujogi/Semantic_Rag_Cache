"""Download the German immigration-law PDFs used in the corpus.

Usage:  python scripts/download_german_law_docs.py [target_dir]
Target directory: first argument, else $DOCS_DIR, else data/pdfs.
"""
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URLS = [
    "https://www.landtag.ltsh.de/export/sites/ltsh/beauftragte/fb/Dokumente/Aufenthaltsrecht-fuer-internationale-Studierende-Deutsch-Englisch.pdf",
    "https://www.tum.de/fileadmin/w00bfo/www/Studium/Internationale_Studierende/Servicestelle_fuer_Zuwanderung_und_Einbuergerung.pdf",
    "https://service.berlin.de/dienstleistung/305244/pdf/",
    "https://bamf.de/SharedDocs/Anlagen/EN/EMN/Studien/wp47-emn-studierende-drittstaaten.pdf",
    "https://www.studierendenwerke.de/fileadmin/user_upload/Stockbilder/PDFs/SIK/202406_Studierende_Fachkraefteeinwanderung.pdf",
    "https://www.stwhh.de/fileadmin/user_upload/Beratung/__Downloads/Flyer_Beratungsangebote_E.pdf",
    "https://welcome.hamburg.de/resource/blob/848992/a1527763678536ee06cfd9a46ecf74f9/amt-m-m31-mobilitaet-nach-der-eu-studentenrichtlinie-im-rahmen-von-erasmus-data.pdf",
    "https://uni-erfurt.de/fileadmin/einrichtung/willy-brandt-school/Brandt_School_Documents/Student_Health_Insurance_in_Germany_2024.pdf",
    "https://www.ostfalia.de/fileadmin/user_upload/Fakultaeten/f/Formulare_und_Dokumente/ASTP/Health-Insurance-Infos_ASTP.pdf",
    "https://www.Uni-Saarland.de/fileadmin/upload/studium/flyer/sprachen/financing-your-studies.pdf",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def filename_for(url: str, response) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    if "filename=" in disposition:
        return disposition.split("filename=")[-1].strip("\"'; ")
    name = urlparse(url).path.rstrip("/").split("/")[-1]
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else os.getenv("DOCS_DIR", PROJECT_ROOT / "data" / "pdfs"))
    target.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as resp:
                body = resp.read()
                name = filename_for(url, resp)
            if not body.startswith(b"%PDF"):
                print(f"Skipped {url}: response is not a PDF (blocked or moved?)")
                continue
            (target / name).write_bytes(body)
            print(f"Downloaded {name} ({len(body) / 1024:.0f} KB)")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    print(f"\nFiles in {target}:")
    for f in sorted(target.glob("*.pdf")):
        print(f"- {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
