import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path


def render(source):
    metadata = dict(re.findall(r"^% (\w+): (.+)$", source.read_text(), re.M))
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            ["tectonic", "--only-cached", "--outdir", directory, str(source)],
            check=True,
            cwd=source.parent,
        )
        pdf = Path(directory, source.stem + ".pdf")
        svg = pdf.with_suffix(".svg")
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        text = svg.read_text().split("\n", 1)[1]

    for identifier in re.findall(r'\bid="([^"]+)"', text):
        text = text.replace(f'id="{identifier}"', f'id="{source.stem}-{identifier}"')
        text = text.replace(f'#{identifier}"', f'#{source.stem}-{identifier}"')
        text = text.replace(f"#{identifier})", f"#{source.stem}-{identifier})")
    text = text.replace("rgb(0%, 0%, 0%)", "currentColor")
    view_width = float(re.search(r'viewBox="0 0 ([\d.]+)', text).group(1))
    width = metadata.get("width", f"{min(42, view_width / 10):.1f}rem")
    title_id = source.stem + "-title"
    desc_id = source.stem + "-desc"
    attributes = (
        f'style="display: block; width: min(100%, {width}); '
        'height: auto; margin: 1.5rem auto;" role="img" '
        f'aria-labelledby="{title_id} {desc_id}" '
    )
    text = text.replace("<svg ", "<svg " + attributes, 1)
    start = text.index(">") + 1
    text = (
        text[:start]
        + f'\n<title id="{title_id}">{escape(metadata["title"])}</title>'
        + f'\n<desc id="{desc_id}">{escape(metadata["description"])}</desc>'
        + text[start:]
    )
    source.with_suffix(".svg").write_text(text)
    if metadata.get("variants") == "dark":
        source.with_name(source.stem + "-dark.svg").write_text(
            text.replace("currentColor", "#cfd0d0")
        )
    print("Rendered", source.name, flush=True)


sources = [Path(arg).resolve() for arg in sys.argv[1:]]
if not sources:
    sources = sorted(Path(__file__).parent.glob("*.tex"))
for source in sources:
    render(source)
