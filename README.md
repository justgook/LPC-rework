# LPC Revised

Creative-commons licensed game assets in 32-pixel, three-quarters (top-down) perspective. Inspired by classics like Chrono Trigger and Link to the Past.

![LPC Revised - Four Seasons](/GithubReadme.png)

![Backslash Animation](/GithubCharacterDemo.gif)

## Getting Started

This repository stores only **source sprites** (hand-drawn base color) and **palette definitions**. All color variants (~50,000+ files) are generated locally after cloning.

### Requirements

- Python 3.8+
- [Pillow](https://pillow.readthedocs.io/)
- [NumPy](https://numpy.org/)

### Generate Color Variants

```bash
# Clone the repository
git clone <repo-url>
cd LPC-rework

# Generate all color variants
python scripts/generate_colors.py
```

This reads the source sprites and `Ramps.json` palette definitions, then produces every color variant through pixel-level recoloring. Generation covers Body, Head, Hair, Eyes, Clothing, Wings, Head Accessories (Helms, Plumage, Glasses), and Children.

### Script Options

```bash
# Preview what would be generated (no file writes)
python scripts/generate_colors.py --dry-run

# Generate only specific items or palette types
python scripts/generate_colors.py --item "Hair"
python scripts/generate_colors.py --palette body

# Verify generated files match existing ones
python scripts/generate_colors.py --verify

# Verbose output (per-item progress, mismatch details)
python scripts/generate_colors.py --verify -v

# Remove all generated color variant directories
python scripts/generate_colors.py --clean
```

Available palette types: `body`, `clothing`, `clothing_alt`, `hair`, `eye`, `metal`

## Repository Structure

```
Characters/
  Body/           # Body sprites (skin tones + alternate fantasy colors)
  Head/           # Head sprites, eyes, noses
  Hair/           # 27 hair styles
  Clothing/       # Clothing items organized by body type and slot
  Head Accessories/  # Helms, plumage, glasses
  Children/       # Child-sized body and head variants
  Props/          # Shields, swords (color generation pending)
  _ Guides & Palettes/  # Palette definitions and reference guides

FX/               # Visual effects
Objects/          # World objects
Structure/        # Buildings and structures
Terrain/          # Terrain tiles
scripts/          # Color generation script
```

### How Color Generation Works

Each sprite item has a **source** (base color) set of PNG files at its root level. The script reads `Ramps.json`, which defines 6-value color ramps (lightest to darkest) for each named color in categories like Skin, Clothing, Hair, Eyes, and Metal. It maps source ramp colors to target ramp colors pixel-by-pixel, producing a recolored PNG for every color in the category.

Source colors by category:
- **Skin**: Ivory
- **Clothing**: Dove
- **Hair**: Blonde
- **Eyes**: Blue
- **Metal**: Steel

## Assets

Assets span Fantasy, Modern, and (eventually) Sci-Fi themes. Each asset is listed in a `Credits.txt` file in its directory with the original author and relevant information.

## License

All assets are licensed under **Creative Commons Attribution 3.0 (CC-BY-3.0)** or **OGA-BY 3.0**. You may freely use, modify, and distribute these assets provided you give appropriate attribution. OGA-BY removes CC-BY 3.0's restriction on technical protection measures, allowing use in commercial games on platforms like iOS.

