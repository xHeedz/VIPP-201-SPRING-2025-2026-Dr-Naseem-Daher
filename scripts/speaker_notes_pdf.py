"""
Printable speaker notes for the long presentation (one block per slide), from scripts/speaker_notes_long.py.

    python scripts/speaker_notes_pdf.py
writes ../26-10-2026/Speaker Notes.html and, with Google Chrome installed, Speaker Notes.pdf
"""
import html
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speaker_notes_long import NOTES  # noqa: E402

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "26-10-2026"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 11.5pt; line-height: 1.45; color: #231a1c; max-width: 720px; margin: 28px auto; }
h1 { font-size: 18pt; color: #7a1f2b; margin-bottom: 2px; }
.slide { border-top: 1px solid #e9e2e3; padding: 10px 0 6px; page-break-inside: avoid; }
.n { color: #7a1f2b; font-weight: bold; font-size: 10pt; letter-spacing: 0.5px; }
h2 { font-size: 13.5pt; margin: 2px 0 4px; }
.shows { color: #6b6b6b; font-size: 10pt; margin: 0 0 6px; }
.say { background: #fbf3f4; border-left: 3px solid #7a1f2b; padding: 6px 10px; margin: 4px 0; }
.asked { font-size: 10.5pt; margin: 6px 0 0 13px; }
.small { color: #6b6b6b; font-size: 10pt; }
"""


def main():
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Speaker Notes</title><style>{CSS}</style></head><body>",
             "<h1>speaker notes: progress presentation</h1>",
             f"<p class='small'>Hadi Al Shmaissani, VIPP 301A, Dr. Daher. {len(NOTES)} slides, about 30 s to 1 min each. "
             "Grey line: what is on the screen. Pink box: what to say. Last line: a likely question.</p>"]
    for i, n in enumerate(NOTES, 1):
        parts.append(f"<div class='slide'><div class='n'>SLIDE {i}</div><h2>{html.escape(n['title'])}</h2>"
                     f"<p class='shows'>{html.escape(n['shows'])}</p><div class='say'>{html.escape(n['say'])}</div>")
        if n.get("asked"):
            q, _, a = n["asked"].partition("? ")
            parts.append(f"<p class='asked'><b>If asked: {html.escape(q)}?</b> {html.escape(a)}</p>")
        parts.append("</div>")
    parts.append("</body></html>")
    out = os.path.join(OUT_DIR, "Speaker Notes.html")
    open(out, "w").write("\n".join(parts))
    print("written", out)
    if os.path.exists(CHROME):
        pdf = os.path.join(OUT_DIR, "Speaker Notes.pdf")
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        "file://" + out], check=True, capture_output=True)
        print("written", pdf)


if __name__ == "__main__":
    main()
