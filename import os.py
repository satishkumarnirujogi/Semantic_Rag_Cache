import os
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

destination_dir = Path(r"C:\Users\niruj\german_law_rag_docs")
destination_dir.mkdir(parents=True, exist_ok=True)

urls = [
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

# Set a standard User-Agent so academic and government servers don't block the request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def get_filename(url: str, response) -> str:
    # 1. Check Content-Disposition header for filename if available
    content_disp = response.headers.get("Content-Disposition", "")
    if "filename=" in content_disp:
        return content_disp.split("filename=")[-1].strip('"\'; ')

    # 2. Extract from URL path
    path = urlparse(url).path.rstrip("/")
    filename = path.split("/")[-1]

    # 3. Add .pdf extension if missing (e.g., service.berlin.de/.../305244/pdf/)
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"

    return filename


for url in urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            filename = get_filename(url, response)
            target_path = destination_dir / filename

            print(f"Downloading: {filename}...")
            with open(target_path, "wb") as f:
                f.write(response.read())

    except Exception as e:
        print(f"Failed to download {url}: {e}")

print(f"\nDone. Downloaded files in {destination_dir}:")
for file in destination_dir.glob("*.pdf"):
    print(f"- {file.name} ({file.stat().st_size / 1024:.1f} KB)")