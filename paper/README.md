# Paper and presentation

```bash
conda activate iese
python figures/make_figures.py    # regenerate both charts from live runs
pdflatex paper.tex  && pdflatex paper.tex
pdflatex slides.tex && pdflatex slides.tex
```

| File | What it is |
|---|---|
| `paper.pdf` | 5 pages, ACM `sigconf`. Abstract, design, evaluation, principles, limitations, positioning. |
| `slides.pdf` | 36 pages (22 frames, 16:9). Follows the demo narrative in `../DEMO.md`. |
| `figures/make_figures.py` | Produces both charts by *running the system*, not from transcribed numbers. |

**The figures are not hand-drawn.** `make_figures.py` builds a governor, extracts
rules from the lending policy, mines the synthetic pack, and plots what actually
happened. Re-running it after a code change re-measures rather than re-illustrates
— so a regression shows up in the paper.

Chart palette `#1C9970` / `#C4761A` was checked with the data-viz validator
(light mode): lightness band, chroma floor, CVD separation (protan ΔE 9.4),
normal-vision floor (ΔE 20.2) and surface contrast all pass. Both series are also
directly labelled, so identity never rests on colour alone.

## Numbers, and one correction

Every figure in the paper comes from a script here or in the repository root.
One is worth flagging because an earlier draft got it wrong: the headline ratio
was reported as 66.7, which divided *rule applications* by reviews. Several rules
fire per decision, so that overstated the claim by exactly that factor. The
decision-denominated figure is **36.8**, and the dashboard now reports both
separately.
