#!/usr/bin/env python3
"""Generate all color variants for LPC-Revised character sprites.

This script reads source (base-color) sprite sheets and recolors them using
palette ramps from Ramps.json to produce every color variant. It replaces the
need to store ~62,000+ recolored PNGs in git — only the ~1,050 hand-drawn
source sprites and palette definitions are tracked.

Usage:
    python scripts/generate_colors.py                  # generate everything
    python scripts/generate_colors.py --dry-run        # preview without writing
    python scripts/generate_colors.py --item "Hair"    # only items matching pattern
    python scripts/generate_colors.py --palette body   # only body/skin palette
    python scripts/generate_colors.py --verify         # compare against existing files
    python scripts/generate_colors.py --clean          # remove generated color dirs

Requirements:
    Python 3.8+, Pillow, NumPy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy is required. Install with: pip install numpy", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = REPO_ROOT / "Characters"
RAMPS_PATH = CHARACTERS_DIR / "_ Guides & Palettes" / "Color Palettes" / "Ramps.json"

# Palette categories in Ramps.json
SKIN_TONES = ["Ivory", "Porcelain", "Peach", "Tan", "Tawny", "Honey", "Bronze", "Brown", "Coffee"]

# Source color name per palette category (the color the root-level source sprites are painted in)
SKIN_SOURCE = "Ivory"
# Source for items whose root sprites are painted in "neutral" clothing colors (Dove)
CLOTHING_SOURCE = "Dove"
# Source for body/head _Alternate Colors (root sprites are painted in Ivory skin tones,
# which equals Clothing/Ivory in the Ramps.json)
BODY_ALT_SOURCE = "Ivory"
HAIR_SOURCE = "Blonde"
EYE_SOURCE = "Blue"
METAL_SOURCE = "Steel"

# ---------------------------------------------------------------------------
# Palette loading
# ---------------------------------------------------------------------------

_ramps_cache = None


def load_ramps() -> dict:
    """Load Ramps.json (cached)."""
    global _ramps_cache
    if _ramps_cache is None:
        with open(RAMPS_PATH) as f:
            _ramps_cache = json.load(f)
    return _ramps_cache


def rgba_to_hex(entry: dict) -> str:
    """Convert Ramps.json RGBA entry to '#rrggbb'."""
    r = int(round(entry["r"] * 255))
    g = int(round(entry["g"] * 255))
    b = int(round(entry["b"] * 255))
    return "#%02x%02x%02x" % (r, g, b)


def get_ramp_hex(category: str, color_name: str) -> list[str]:
    """Get a color ramp as hex values (lightest to darkest) from Ramps.json."""
    ramps = load_ramps()
    return [rgba_to_hex(e) for e in ramps[category][color_name]]


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b)."""
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def build_color_map(
    source_ramp: list[str], target_ramp: list[str]
) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    """Build a pixel color mapping from source ramp to target ramp.

    Both ramps must have the same length. Each position maps source[i] -> target[i].
    Only entries where source != target are included.
    """
    mapping = {}
    seen = set()
    for s, t in zip(source_ramp, target_ramp):
        src_rgb = hex_to_rgb(s)
        tgt_rgb = hex_to_rgb(t)
        if src_rgb in seen:
            continue
        seen.add(src_rgb)
        if src_rgb != tgt_rgb:
            mapping[src_rgb] = tgt_rgb
    return mapping


# ---------------------------------------------------------------------------
# Image recoloring (numpy-accelerated)
# ---------------------------------------------------------------------------


def _apply_color_map_np(arr: np.ndarray, color_map: dict) -> np.ndarray:
    """Apply color map to an RGBA numpy array. Returns new array.

    Pixels with alpha <= 1 are left unchanged.
    """
    out = arr.copy()
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    # Mask: only recolor pixels with alpha > 5 (nearly invisible pixels are skipped,
    # matching the behavior of the original lpctools recoloring)
    visible = alpha > 5

    for src_rgb, tgt_rgb in color_map.items():
        # Match pixels where R, G, B all match source AND pixel is visible
        match = (
            visible
            & (rgb[:, :, 0] == src_rgb[0])
            & (rgb[:, :, 1] == src_rgb[1])
            & (rgb[:, :, 2] == src_rgb[2])
        )
        out[:, :, 0][match] = tgt_rgb[0]
        out[:, :, 1][match] = tgt_rgb[1]
        out[:, :, 2][match] = tgt_rgb[2]

    return out


def recolor_image(src_path: Path, dst_path: Path, color_map: dict) -> None:
    """Recolor a PNG image using the given color map."""
    img = Image.open(src_path).convert("RGBA")
    arr = np.array(img)
    out = _apply_color_map_np(arr, color_map)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(dst_path, "PNG")


# ---------------------------------------------------------------------------
# Image cache for verify (avoids re-reading source for each color)
# ---------------------------------------------------------------------------

_src_img_cache: dict[str, np.ndarray] = {}


def _load_src_cached(path: Path) -> np.ndarray:
    """Load and cache source image as numpy array."""
    key = str(path)
    if key not in _src_img_cache:
        _src_img_cache[key] = np.array(Image.open(path).convert("RGBA"))
    return _src_img_cache[key]


def clear_src_cache():
    """Clear the source image cache."""
    _src_img_cache.clear()


# ---------------------------------------------------------------------------
# Item discovery and classification
# ---------------------------------------------------------------------------


class SpriteItem:
    """A sprite item directory containing source PNGs and color variant subdirectories."""

    __slots__ = ("path", "palette_type", "ramps_category", "source_color",
                 "color_names", "use_alternate_colors_dir", "source_color_dir")

    def __init__(self, path: Path, palette_type: str, ramps_category: str,
                 source_color: str, color_names: list[str],
                 use_alternate_colors_dir: bool = False,
                 source_color_dir: str | None = None):
        self.path = path
        self.palette_type = palette_type
        self.ramps_category = ramps_category
        self.source_color = source_color
        self.color_names = color_names
        self.use_alternate_colors_dir = use_alternate_colors_dir
        # If set, source PNGs are in this subdirectory instead of root
        self.source_color_dir = source_color_dir

    def __repr__(self):
        rel = self.path.relative_to(REPO_ROOT)
        return f"SpriteItem({rel}, {self.palette_type}, {len(self.color_names)} colors)"

    @property
    def short_name(self) -> str:
        """Short display name for logging."""
        try:
            return str(self.path.relative_to(CHARACTERS_DIR))
        except ValueError:
            return str(self.path)


def get_source_pngs(item_dir: Path) -> list[Path]:
    """Get root-level PNG files (source sprites), excluding _Base.png."""
    return sorted(
        f for f in item_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".png" and not f.name.startswith("_")
    )


def _skin_colors() -> list[str]:
    return list(load_ramps()["Skin"].keys())


def _clothing_colors() -> list[str]:
    return list(load_ramps()["Clothing"].keys())


def _clothing_colors_no_skin() -> list[str]:
    """Clothing colors excluding skin tone names (for body/head _Alternate Colors)."""
    skin = set(load_ramps()["Skin"].keys())
    return [c for c in load_ramps()["Clothing"].keys() if c not in skin]


def _hair_colors() -> list[str]:
    return list(load_ramps()["Hair"].keys())


def _eye_colors() -> list[str]:
    return list(load_ramps()["Eyes"].keys())


def _metal_colors() -> list[str]:
    return list(load_ramps()["Metal"].keys())


def discover_items() -> list[SpriteItem]:
    """Walk Characters/ and discover all sprite items."""
    items: list[SpriteItem] = []

    # --- Body ---
    body_dir = CHARACTERS_DIR / "Body"
    if body_dir.is_dir():
        for subdir in sorted(body_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            if "Wing" in subdir.name:
                # Wings use clothing palette, flat structure
                items.append(SpriteItem(
                    subdir, "clothing", "Clothing", CLOTHING_SOURCE,
                    _clothing_colors(),
                ))
                behind = subdir / "_ Behind"
                if behind.is_dir():
                    items.append(SpriteItem(
                        behind, "clothing", "Clothing", CLOTHING_SOURCE,
                        _clothing_colors(),
                    ))
            else:
                # Body types: skin tones (flat) + alt colors (_Alternate Colors/)
                items.append(SpriteItem(
                    subdir, "body", "Skin", SKIN_SOURCE,
                    _skin_colors(),
                ))
                items.append(SpriteItem(
                    subdir, "clothing_alt", "Clothing", BODY_ALT_SOURCE,
                    _clothing_colors_no_skin(), use_alternate_colors_dir=True,
                ))

    # --- Head ---
    head_dir = CHARACTERS_DIR / "Head"
    if head_dir.is_dir():
        for subdir in sorted(head_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            if "Eyes" in subdir.name:
                items.append(SpriteItem(
                    subdir, "eye", "Eyes", EYE_SOURCE,
                    _eye_colors(),
                ))
            elif "Head" in subdir.name or "Nose" in subdir.name:
                items.append(SpriteItem(
                    subdir, "body", "Skin", SKIN_SOURCE,
                    _skin_colors(),
                ))
                items.append(SpriteItem(
                    subdir, "clothing_alt", "Clothing", BODY_ALT_SOURCE,
                    _clothing_colors_no_skin(), use_alternate_colors_dir=True,
                ))

    # --- Hair ---
    hair_dir = CHARACTERS_DIR / "Hair"
    if hair_dir.is_dir():
        for subdir in sorted(hair_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            items.append(SpriteItem(
                subdir, "hair", "Hair", HAIR_SOURCE,
                _hair_colors(),
            ))
            behind = subdir / "_Behind"
            if behind.is_dir():
                items.append(SpriteItem(
                    behind, "hair", "Hair", HAIR_SOURCE,
                    _hair_colors(),
                ))

    # --- Clothing ---
    clothing_dir = CHARACTERS_DIR / "Clothing"
    if clothing_dir.is_dir():
        for body_type in sorted(clothing_dir.iterdir()):
            if not body_type.is_dir():
                continue
            for category in sorted(body_type.iterdir()):
                if not category.is_dir():
                    continue
                for item_dir in sorted(category.iterdir()):
                    if not item_dir.is_dir() or item_dir.name.startswith("_"):
                        continue
                    items.append(SpriteItem(
                        item_dir, "clothing", "Clothing", CLOTHING_SOURCE,
                        _clothing_colors(),
                    ))

    # --- Head Accessories ---
    ha_dir = CHARACTERS_DIR / "Head Accessories"
    if ha_dir.is_dir():
        for subdir in sorted(ha_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            if "Eyepatch" in subdir.name:
                continue
            elif "Plumage" in subdir.name:
                items.append(SpriteItem(
                    subdir, "clothing", "Clothing", CLOTHING_SOURCE,
                    _clothing_colors(),
                ))
            elif "Helm" in subdir.name or "Eyewear" in subdir.name or "Glasses" in subdir.name:
                items.append(SpriteItem(
                    subdir, "metal", "Metal", METAL_SOURCE,
                    _metal_colors(),
                ))

    # --- Children ---
    children_dir = CHARACTERS_DIR / "Children"
    if children_dir.is_dir():
        for category_name in ["Body", "Head"]:
            cat_dir = children_dir / category_name
            if not cat_dir.is_dir():
                continue
            # Check for item subdirectories (like "Head 01 - Default")
            item_subdirs = [
                d for d in sorted(cat_dir.iterdir())
                if d.is_dir() and not d.name.startswith("_")
                and d.name not in _skin_colors() and d.name not in _clothing_colors()
            ]
            if item_subdirs:
                for item_dir in item_subdirs:
                    items.append(SpriteItem(
                        item_dir, "body", "Skin", SKIN_SOURCE,
                        _skin_colors(),
                    ))
                    items.append(SpriteItem(
                        item_dir, "clothing_alt", "Clothing", BODY_ALT_SOURCE,
                        _clothing_colors_no_skin(), use_alternate_colors_dir=True,
                    ))
            else:
                items.append(SpriteItem(
                    cat_dir, "body", "Skin", SKIN_SOURCE,
                    _skin_colors(),
                ))
                items.append(SpriteItem(
                    cat_dir, "clothing_alt", "Clothing", BODY_ALT_SOURCE,
                    _clothing_colors_no_skin(), use_alternate_colors_dir=True,
                ))

    # --- Props ---
    # NOTE: Props (Sword, Shield) are excluded for now. They use palette mappings
    # that don't fully match Ramps.json (especially Wood), and many _Behind color
    # variants were never generated. Props will be handled in a future refactor
    # when more props are added.

    return items


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_item(item: SpriteItem, dry_run: bool = False, verify: bool = False,
                  verbose: bool = False) -> dict:
    """Generate all color variants for a single sprite item."""
    source_ramp = get_ramp_hex(item.ramps_category, item.source_color)
    stats = {"generated": 0, "skipped": 0, "verified": 0, "mismatched": 0, "errors": 0}

    # Determine source PNG files
    if item.source_color_dir:
        src_dir = item.path / item.source_color_dir
        if not src_dir.is_dir():
            stats["errors"] += 1
            return stats
        source_pngs = get_source_pngs(src_dir)
    else:
        source_pngs = get_source_pngs(item.path)

    if not source_pngs:
        return stats

    # Pre-load source images into cache for verify mode
    if verify:
        for sp in source_pngs:
            _load_src_cached(sp)

    for color_name in item.color_names:
        # Skip the source color itself
        if color_name == item.source_color:
            stats["skipped"] += len(source_pngs)
            continue
        # Skip if using a color dir as source and this IS that dir
        if item.source_color_dir and color_name == item.source_color_dir:
            stats["skipped"] += len(source_pngs)
            continue

        target_ramp = get_ramp_hex(item.ramps_category, color_name)
        color_map = build_color_map(source_ramp, target_ramp)

        if not color_map:
            stats["skipped"] += len(source_pngs)
            continue

        # Output directory
        if item.use_alternate_colors_dir:
            out_dir = item.path / "_Alternate Colors" / color_name
        else:
            out_dir = item.path / color_name

        for src_png in source_pngs:
            dst_path = out_dir / src_png.name

            if dry_run:
                stats["generated"] += 1
                continue

            if verify:
                if dst_path.exists():
                    if _verify_file_np(src_png, dst_path, color_map):
                        stats["verified"] += 1
                    else:
                        stats["mismatched"] += 1
                        if verbose:
                            print(f"    MISMATCH: {color_name}/{src_png.name}")
                else:
                    stats["mismatched"] += 1
                    if verbose:
                        print(f"    MISSING:  {color_name}/{src_png.name}")
                stats["generated"] += 1
                continue

            try:
                recolor_image(src_png, dst_path, color_map)
                stats["generated"] += 1
            except Exception as e:
                print(f"  ERROR: {src_png} -> {dst_path}: {e}", file=sys.stderr)
                stats["errors"] += 1

    return stats


def _verify_file_np(src_path: Path, existing_path: Path, color_map: dict) -> bool:
    """Check if existing file matches what we'd generate (numpy-accelerated)."""
    try:
        src_arr = _load_src_cached(src_path)
        existing = np.array(Image.open(existing_path).convert("RGBA"))

        if src_arr.shape != existing.shape:
            return False

        # Apply color map to source
        expected = _apply_color_map_np(src_arr, color_map)

        # Compare only pixels where either has alpha > 5
        src_alpha = src_arr[:, :, 3]
        ex_alpha = existing[:, :, 3]
        compare_mask = (src_alpha > 5) | (ex_alpha > 5)

        if not compare_mask.any():
            return True

        return np.array_equal(expected[compare_mask], existing[compare_mask])
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def clean_generated(dry_run: bool = False) -> int:
    """Remove all generated color variant directories."""
    ramps = load_ramps()
    all_color_names = set()
    for category in ramps.values():
        for name in category:
            all_color_names.add(name)

    removed = 0
    # Walk bottom-up so we can remove directories after their contents
    for root, dirs, _files in os.walk(CHARACTERS_DIR, topdown=False):
        root_path = Path(root)
        if "_ Guides & Palettes" in str(root_path):
            continue

        # Remove _Alternate Colors directories
        if root_path.name == "_Alternate Colors":
            if dry_run:
                print(f"  Would remove: {root_path.relative_to(REPO_ROOT)}")
            else:
                shutil.rmtree(root_path)
            removed += 1
            continue

        # Remove color-named directories inside item directories
        if root_path.name in all_color_names:
            parent = root_path.parent
            if parent.name not in ("Characters", "") and "_ Guides" not in str(parent):
                if dry_run:
                    print(f"  Would remove: {root_path.relative_to(REPO_ROOT)}")
                else:
                    shutil.rmtree(root_path)
                removed += 1

    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate color variants for LPC-Revised character sprites.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files.")
    parser.add_argument("--verify", action="store_true",
                        help="Compare output against existing files.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all generated color variant directories.")
    parser.add_argument("--item", type=str, default=None,
                        help="Filter: only items whose path contains this string.")
    parser.add_argument("--palette", type=str, default=None,
                        choices=["body", "clothing", "clothing_alt", "hair", "eye", "metal"],
                        help="Filter: only items using this palette type.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-item progress and mismatch details.")
    args = parser.parse_args()

    print("Discovering sprite items...")
    all_items = discover_items()

    items = all_items
    if args.item:
        pattern = args.item.lower()
        items = [i for i in items if pattern in str(i.path).lower()]
    if args.palette:
        items = [i for i in items if i.palette_type == args.palette]

    print(f"Found {len(items)} sprite items (of {len(all_items)} total).")

    if args.clean:
        print("\nCleaning generated directories...")
        removed = clean_generated(dry_run=args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {removed} directories.")
        return

    by_type: dict[str, int] = {}
    for item in items:
        by_type[item.palette_type] = by_type.get(item.palette_type, 0) + 1
    for ptype, count in sorted(by_type.items()):
        print(f"  {ptype}: {count} items")

    mode = "DRY RUN" if args.dry_run else ("VERIFY" if args.verify else "GENERATING")
    print(f"\n--- {mode} ---")

    start = time.time()
    total = {"generated": 0, "skipped": 0, "verified": 0, "mismatched": 0, "errors": 0}

    for i, item in enumerate(items):
        item_start = time.time()

        if args.verbose:
            print(f"\n  [{i+1}/{len(items)}] {item.short_name} ({item.palette_type})")

        stats = generate_item(item, args.dry_run, args.verify, args.verbose)

        for k, v in stats.items():
            total[k] += v

        item_elapsed = time.time() - item_start
        elapsed = time.time() - start

        if args.verbose:
            parts = []
            if stats["generated"]:
                parts.append(f"{stats['generated']} files")
            if stats["verified"]:
                parts.append(f"{stats['verified']} OK")
            if stats["mismatched"]:
                parts.append(f"{stats['mismatched']} MISMATCH")
            if stats["errors"]:
                parts.append(f"{stats['errors']} ERR")
            print(f"    {', '.join(parts)} ({item_elapsed:.1f}s)")
        else:
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                elapsed = time.time() - start
                mismatch_info = f" ({total['mismatched']} mismatched)" if total["mismatched"] else ""
                print(f"  [{i+1}/{len(items)}] {elapsed:.1f}s - {total['generated']} files{mismatch_info}")

        # Clear source cache between items to limit memory usage
        clear_src_cache()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Generated: {total['generated']}")
    if total["skipped"]:
        print(f"  Skipped (same as source): {total['skipped']}")
    if total["verified"]:
        print(f"  Verified OK: {total['verified']}")
    if total["mismatched"]:
        print(f"  MISMATCHED: {total['mismatched']}")
    if total["errors"]:
        print(f"  ERRORS: {total['errors']}")


if __name__ == "__main__":
    main()
