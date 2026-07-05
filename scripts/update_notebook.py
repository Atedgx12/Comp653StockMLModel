"""Add Slide 5 (cross-sectional probe) to notebooks/03_presentation.ipynb."""
import json, os

NB = os.path.join(os.path.dirname(__file__), "..", "notebooks", "03_presentation.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Only add if the slide isn't already there
if any("Slide 5" in "".join(c["source"]) for c in nb["cells"]):
    print("Slide 5 already present — nothing to do.")
    raise SystemExit(0)


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ── Slide 5 markdown ──────────────────────────────────────────────────────────
SLIDE5_MD = """\
---
## Slide 5 — Cross-Sectional vs Absolute Direction

### What the probe measures
The cross-sectional label strategy used in the main pipeline marks a stock as a 1 (outperformer)
only when its next-day return lands in the top 20 percent of the full universe that day, and a 0
only when it lands in the bottom 20 percent. This is compared against the simpler absolute
direction label: 1 if the stock closes higher than it opened, 0 otherwise.

### Why it matters
Absolute direction is essentially a coin flip on most stocks most days. The majority-class
baseline already sits at roughly 53 percent because markets drift upward on average, which means
a naive always-long prediction beats random chance without learning anything. Cross-sectional
labels remove the market-wide drift by definition, giving the model a cleaner learning target
and a tighter majority-class baseline near 50 percent.

### Interpretation
A positive accuracy minus baseline value for the cross-sectional probe confirms that the model
extracts relative-strength signal that generalizes out of sample. A larger delta for cross-sectional
than for absolute direction validates the design choice made in the main pipeline.\
"""

# ── Cell: run the probe (uses cached output when available) ───────────────────
PROBE_RUN = """\
import subprocess, sys, os, re

probe_script = os.path.join(\"..\", \"scripts\", \"cross_section.py\")
out_file     = os.path.join(\"..\", \"outputs\", \"cross_section_out.txt\")

if os.path.exists(out_file):
    with open(out_file) as fh:
        output = fh.read()
else:
    result = subprocess.run(
        [sys.executable, probe_script],
        capture_output=True, text=True,
        cwd=os.path.join(\"..\", \"scripts\"),
    )
    output = result.stdout + result.stderr
    os.makedirs(os.path.join(\"..\", \"outputs\"), exist_ok=True)
    with open(out_file, \"w\") as fh:
        fh.write(output)

print(output)\
"""

# ── Cell: pretty-print the parsed probe table ─────────────────────────────────
PROBE_PARSE = """\
sections = {}
cur = None
for line in output.strip().splitlines():
    if \"ABSOLUTE\" in line:
        cur = \"absolute\"; sections[cur] = []
    elif \"CROSS\" in line:
        cur = \"cross_sectional\"; sections[cur] = []
    elif cur and \"=\" in line:
        sections[cur].append(line.strip())

print(\"Absolute direction vs cross-sectional outperformance\")
print(\"-\" * 56)
for key, rows in sections.items():
    print(f\"\\n  {key.replace('_', ' ').title()}\")
    for r in rows:
        print(f\"    {r}\")\
"""

# ── Cell: bar chart comparing the two strategies ──────────────────────────────
PROBE_BAR = """\
import matplotlib.pyplot as plt, numpy as np, re

def _parse_delta(text, marker):
    idx = text.find(marker)
    if idx == -1:
        return 0.0
    chunk = text[idx:]
    for line in chunk.splitlines():
        if \"accuracy - baseline\" in line:
            m = re.search(r\"[+-]?\\d+\\.\\d+\", line)
            return float(m.group()) if m else 0.0
    return 0.0

delta_abs = _parse_delta(output, \"ABSOLUTE\")
delta_cs  = _parse_delta(output, \"CROSS\")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    [\"Absolute direction\", \"Cross-sectional\"],
    [delta_abs, delta_cs],
    color=[\"#1f77b4\", \"#2ca02c\"], width=0.45,
)
ax.axhline(0, color=\"black\", linewidth=0.8)
ax.set_ylabel(\"Accuracy minus majority-class baseline\")
ax.set_title(\"Signal strength: absolute vs cross-sectional label strategy\")
for bar, v in zip(bars, [delta_abs, delta_cs]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (0.001 if v >= 0 else -0.003),
            f\"{v:+.4f}\", ha=\"center\", va=\"bottom\", fontsize=11)
plt.tight_layout()
out = os.path.join(\"..\", \"outputs\", \"cross_section_bar.png\")
plt.savefig(out, dpi=150)
plt.show()
print(f\"Saved {out}\")\
"""

nb["cells"].extend([
    md_cell(SLIDE5_MD),
    code_cell(PROBE_RUN),
    code_cell(PROBE_PARSE),
    code_cell(PROBE_BAR),
])

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook updated: {len(nb['cells'])} cells")
