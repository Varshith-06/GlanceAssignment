"""writeup.md -> writeup.pdf via python-markdown + headless Edge.

Run:  python deliverable/make_pdf.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt;
       line-height: 1.45; max-width: 52em; margin: 2em auto; color: #1a1a1a; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13.5pt; margin-top: 1.6em; border-bottom: 1px solid #bbb; }
code, pre { font-family: Consolas, monospace; font-size: 9pt;
            background: #f5f5f5; }
pre { padding: 8px; border: 1px solid #ddd; overflow-x: hidden;
      white-space: pre-wrap; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; }
th { background: #eee; }
blockquote { color: #444; border-left: 3px solid #ccc; margin-left: 0;
             padding-left: 1em; }
"""


def main() -> None:
    md = (HERE / "writeup.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = HERE / "writeup.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = HERE / "writeup.pdf"
    subprocess.run([EDGE, "--headless", "--disable-gpu",
                    f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer",
                    html_path.as_uri()], check=True, timeout=120)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
