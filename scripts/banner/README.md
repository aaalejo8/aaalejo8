# Animated profile header

Generators for the animated banner and the three link tiles shown at the top of the profile README.

```bash
pip install -r scripts/banner/requirements.txt
python scripts/banner/generate.py          # regenerate assets/animated/*.svg
python scripts/banner/clean_background.py  # only if the source photo changes
```

| File | Purpose |
| --- | --- |
| `generate.py` | Builds `assets/animated/banner.svg` (landscape + particle name + subtitle) and `link-<name>-<dark/light>.svg` (arrow + icon tiles). Tune `LOOP`, `PARTICLES`, texts and timings at the top of the file. |
| `clean_background.py` | Removes the title baked into `assets/source/salta-original.png` → `assets/source/salta-clean.png`. |
| `fonts/` | Poppins (SIL Open Font License, see `OFL.txt`). Text is converted to SVG paths, so viewers don't need the font. |
| `icons/` | Icons from [skillicons.dev](https://skillicons.dev) (MIT), embedded in the tiles. |

All SVGs are self-contained (SMIL + CSS, no JavaScript, background photo embedded as a data URI), so they animate inside GitHub's `<img>` rendering. With `prefers-reduced-motion: reduce` the particles and moving layers are hidden and a static name/subtitle is shown.

Everything shares one 14 s loop: embers gather into the name (0–3.9 s) → subtitle wipe (4–5.2 s) → arrows draw (5–6.6 s) → hold with light sweep and sparks → name dissolves upward (~11.6 s) → restart.

The particle technique (optimal-transport matching with `scipy.optimize.linear_sum_assignment` and one SMIL `animateTransform` per particle) is inspired by [macu-dev/macu-dev](https://github.com/macu-dev/macu-dev).
