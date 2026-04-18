"""
Generate curling board images from shot location data.

Recreates single-shot board diagrams for data quality verification
by comparing reconstructed images to the original scraped data.

Usage:
    python generate_board_image.py --event 1 --match 1 --end 1 --shot 5 -o board.png
"""

import argparse
import math

import pandas as pd

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Board rendering constants
# ---------------------------------------------------------------------------

# Output image dimensions (pixels).
# Height is derived from the Y range to preserve physical aspect ratio.
IMAGE_WIDTH = 400

# Normalised coordinate range displayed on the board.
# The house centre is at (0, 0); y-positive is toward the hog line / delivery
# end (bottom of image), y-negative is toward the back line / hack (top of
# image).
# In real units: 1 normalised unit = HOUSE_RADIUS px at 300 DPI = 6 feet
# (the radius of the 12-foot ring).
#
# Derived from extract_shot_data.py geometry constants
# (HOUSE_CX=161, HOUSE_CY=171, HOUSE_RADIUS=112, STONE_Y_MAX_FRAC=0.93):
#
#   X extent:  ±HOUSE_CX / HOUSE_RADIUS = ±161/112 ≈ ±1.44
#   Y_MIN:     -HOUSE_CY / HOUSE_RADIUS  = -171/112 ≈ -1.53  (top of PDF crop)
#   Y_MAX:     4.60 provides visual buffer beyond the hog line for guards
#
# Physical reference lines (normalised):
#   Back line = -1.0  ( 6 ft behind tee:  1 × 6 ft)
#   Hog line  = +3.5  (21 ft in front of tee: 3.5 × 6 ft)
X_MIN, X_MAX = -1.44, 1.44   # width  ≈ 2.88 normalised units
Y_MIN, Y_MAX = -1.53, 4.60   # height ≈ 6.13 normalised units

IMAGE_HEIGHT = round(IMAGE_WIDTH * (Y_MAX - Y_MIN) / (X_MAX - X_MIN))  # ≈ 853

# Physical reference lines
BACK_LINE_Y = -1.0   # 6 ft behind tee line  (1 normalised unit = 6 ft)
HOG_LINE_Y = 3.5     # 21 ft in front of tee line

# House ring radii (normalised: 12-foot ring = 1.0)
TWELVE_FOOT_RADIUS = 1.0
EIGHT_FOOT_RADIUS = 2.0 / 3.0   # ≈ 0.667
FOUR_FOOT_RADIUS = 1.0 / 3.0    # ≈ 0.333
BUTTON_RADIUS = 0.5 / 6.0       # ≈ 0.083

# Stone radius in normalised coordinates (a real stone is ~11.25 in diameter,
# the 12-foot ring is 12 ft = 144 in, so stone radius ≈ 5.625 / 72 ≈ 0.078)
STONE_RADIUS = 0.078

# Colours
COLOUR_BACKGROUND = (255, 255, 255)
COLOUR_RING_12 = (170, 170, 230)   # blue / lilac
COLOUR_RING_8 = (255, 255, 255)    # white
COLOUR_RING_4 = (230, 160, 160)    # light red / pink
COLOUR_BUTTON = (255, 255, 255)    # white
COLOUR_RING_OUTLINE = (100, 100, 100)
COLOUR_LINE = (180, 180, 180)      # centre / tee lines
COLOUR_HOG_LINE = (200, 80, 80)    # hog line
COLOUR_BACK_LINE = (140, 140, 200) # back line
COLOUR_RED_STONE = (220, 40, 40)
COLOUR_RED_OUTLINE = (160, 20, 20)
COLOUR_YELLOW_STONE = (240, 200, 60)
COLOUR_YELLOW_OUTLINE = (180, 150, 30)

MAX_STONES_PER_TEAM = 8


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def _norm_to_pixel(nx, ny):
    """Convert normalised board coordinates to pixel coordinates.

    y increases downward in pixel space: the back line (small negative y)
    maps near the top of the image and the hog line (large positive y) maps
    near the bottom, matching the standard top-down view with the button
    near the top of the image.
    """
    px = (nx - X_MIN) / (X_MAX - X_MIN) * IMAGE_WIDTH
    py = (ny - Y_MIN) / (Y_MAX - Y_MIN) * IMAGE_HEIGHT
    return px, py


def _norm_radius_to_pixels(r):
    """Convert a normalised radius to pixel radius (using x-axis scale)."""
    return r / (X_MAX - X_MIN) * IMAGE_WIDTH


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_ring(draw, cx, cy, radius, fill, outline):
    """Draw a filled circle centred at pixel (cx, cy)."""
    r = _norm_radius_to_pixels(radius)
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=fill,
        outline=outline,
        width=1,
    )


def _draw_stone(draw, nx, ny, fill, outline):
    """Draw a stone at normalised position (nx, ny)."""
    px, py = _norm_to_pixel(nx, ny)
    r = _norm_radius_to_pixels(STONE_RADIUS)
    draw.ellipse(
        [px - r, py - r, px + r, py + r],
        fill=fill,
        outline=outline,
        width=1,
    )


# ---------------------------------------------------------------------------
# Stone extraction from shot data dict
# ---------------------------------------------------------------------------

def _extract_stones(shot_data, team_num):
    """Return list of (x, y) tuples for stones belonging to *team_num* (1 or 2).

    Empty/missing positions are skipped.
    """
    stones = []
    prefix = f"team{team_num}_stone"
    for i in range(1, MAX_STONES_PER_TEAM + 1):
        x_key = f"{prefix}{i}_x"
        y_key = f"{prefix}{i}_y"
        x_val = shot_data.get(x_key, "")
        y_val = shot_data.get(y_key, "")
        if x_val != "" and y_val != "":
            try:
                stones.append((float(x_val), float(y_val)))
            except (ValueError, TypeError):
                continue
    return stones


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_board_image(shot_data, image_width=IMAGE_WIDTH,
                         image_height=IMAGE_HEIGHT):
    """Generate a curling board image for a single shot.

    Parameters
    ----------
    shot_data : dict
        A row from ``shot_locations.csv`` containing stone position columns
        (``team1_stone1_x``, ``team1_stone1_y``, …, ``team2_stone8_y``).
    image_width : int, optional
        Width of the output image in pixels (default 400).
    image_height : int, optional
        Height of the output image in pixels (default IMAGE_HEIGHT ≈ 613).

    Returns
    -------
    PIL.Image.Image
        The rendered board image.
    """
    img = Image.new("RGB", (image_width, image_height), COLOUR_BACKGROUND)
    draw = ImageDraw.Draw(img)

    # House centre in pixel coordinates
    hx, hy = _norm_to_pixel(0, 0)

    # --- Hog line -----------------------------------------------------------
    x_left, _ = _norm_to_pixel(X_MIN, HOG_LINE_Y)
    x_right, _ = _norm_to_pixel(X_MAX, HOG_LINE_Y)
    _, hog_py = _norm_to_pixel(0, HOG_LINE_Y)
    draw.line([(x_left, hog_py), (x_right, hog_py)],
              fill=COLOUR_HOG_LINE, width=2)

    # --- Back line ----------------------------------------------------------
    x_left, _ = _norm_to_pixel(X_MIN, BACK_LINE_Y)
    x_right, _ = _norm_to_pixel(X_MAX, BACK_LINE_Y)
    _, back_py = _norm_to_pixel(0, BACK_LINE_Y)
    draw.line([(x_left, back_py), (x_right, back_py)],
              fill=COLOUR_BACK_LINE, width=2)

    # --- Centre line (vertical, full height) --------------------------------
    cx_px, _ = _norm_to_pixel(0, Y_MIN)
    _, y_top = _norm_to_pixel(0, Y_MIN)
    _, y_bot = _norm_to_pixel(0, Y_MAX)
    draw.line([(cx_px, y_top), (cx_px, y_bot)], fill=COLOUR_LINE, width=1)

    # --- Tee line (horizontal, through house centre) ------------------------
    x_left, _ = _norm_to_pixel(X_MIN, 0)
    x_right, _ = _norm_to_pixel(X_MAX, 0)
    draw.line([(x_left, hy), (x_right, hy)], fill=COLOUR_LINE, width=1)

    # --- House rings (draw from largest to smallest) ------------------------
    _draw_ring(draw, hx, hy, TWELVE_FOOT_RADIUS, COLOUR_RING_12,
               COLOUR_RING_OUTLINE)
    _draw_ring(draw, hx, hy, EIGHT_FOOT_RADIUS, COLOUR_RING_8,
               COLOUR_RING_OUTLINE)
    _draw_ring(draw, hx, hy, FOUR_FOOT_RADIUS, COLOUR_RING_4,
               COLOUR_RING_OUTLINE)
    _draw_ring(draw, hx, hy, BUTTON_RADIUS, COLOUR_BUTTON,
               COLOUR_RING_OUTLINE)

    # --- Stones -------------------------------------------------------------
    team1_stones = _extract_stones(shot_data, 1)
    team2_stones = _extract_stones(shot_data, 2)

    print(f"[debug] event_id={shot_data.get('event_id')}  match_id={shot_data.get('match_id')}  "
          f"end={shot_data.get('end_number')}  shot={shot_data.get('shot_number')}  "
          f"team={shot_data.get('team_code')}  player={shot_data.get('player_name')}")
    print(f"[debug] shot_type={shot_data.get('shot_type')}  turn={shot_data.get('turn')}  "
          f"accuracy={shot_data.get('accuracy')}")
    print(f"[debug] team1 stones ({len(team1_stones)}):")
    for i, (sx, sy) in enumerate(team1_stones, 1):
        dist = math.sqrt(sx * sx + sy * sy)
        ring = ("button" if dist < BUTTON_RADIUS else
                "4-ft"   if dist < FOUR_FOOT_RADIUS else
                "8-ft"   if dist < EIGHT_FOOT_RADIUS else
                "12-ft"  if dist < TWELVE_FOOT_RADIUS else
                "outside house")
        print(f"[debug]   stone {i}: x={sx:+.3f}  y={sy:+.3f}  dist={dist:.3f}  ({ring})")
    print(f"[debug] team2 stones ({len(team2_stones)}):")
    for i, (sx, sy) in enumerate(team2_stones, 1):
        dist = math.sqrt(sx * sx + sy * sy)
        ring = ("button" if dist < BUTTON_RADIUS else
                "4-ft"   if dist < FOUR_FOOT_RADIUS else
                "8-ft"   if dist < EIGHT_FOOT_RADIUS else
                "12-ft"  if dist < TWELVE_FOOT_RADIUS else
                "outside house")
        print(f"[debug]   stone {i}: x={sx:+.3f}  y={sy:+.3f}  dist={dist:.3f}  ({ring})")

    for sx, sy in team1_stones:
        _draw_stone(draw, sx, sy, COLOUR_RED_STONE, COLOUR_RED_OUTLINE)
    for sx, sy in team2_stones:
        _draw_stone(draw, sx, sy, COLOUR_YELLOW_STONE, COLOUR_YELLOW_OUTLINE)

    return img


def generate_board_image_from_parquet(parquet_path, event_id, match_id,
                                      end_number, shot_number):
    """Look up a specific shot in *parquet_path* and generate a board image.

    Parameters
    ----------
    parquet_path : str
        Path to ``shot_locations.parquet``.
    event_id, match_id, end_number, shot_number : int or str
        Identifiers for the shot to render.

    Returns
    -------
    PIL.Image.Image
        The rendered board image.

    Raises
    ------
    ValueError
        If no matching row is found.
    """
    event_id = int(event_id)
    match_id = int(match_id)
    end_number = int(end_number)
    shot_number = int(shot_number)

    df = pd.read_parquet(parquet_path)
    mask = (
        (df["event_id"] == event_id)
        & (df["match_id"] == match_id)
        & (df["end_number"] == end_number)
        & (df["shot_number"] == shot_number)
    )
    matches = df[mask]
    if matches.empty:
        raise ValueError(
            f"No shot found for event_id={event_id}, match_id={match_id}, "
            f"end_number={end_number}, shot_number={shot_number}"
        )

    # Convert to plain dict; replace NaN/None with "" so _extract_stones
    # can use its existing `!= ""` guard.
    row = {
        k: ("" if (v is None or (isinstance(v, float) and math.isnan(v))) else v)
        for k, v in matches.iloc[0].to_dict().items()
    }
    return generate_board_image(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a curling board image from shot location data."
    )
    parser.add_argument(
        "--parquet", default="output/shot_locations.parquet",
        help="Path to shot_locations.parquet (default: output/shot_locations.parquet)",
    )
    parser.add_argument("--event", required=True, help="Event ID")
    parser.add_argument("--match", required=True, help="Match ID")
    parser.add_argument("--end", required=True, help="End number")
    parser.add_argument("--shot", required=True, help="Shot number")
    parser.add_argument(
        "-o", "--output", default="board.png",
        help="Output image path (default: board.png)",
    )
    args = parser.parse_args()

    img = generate_board_image_from_parquet(
        args.parquet, args.event, args.match, args.end, args.shot,
    )
    img.save(args.output)
    print(f"Board image saved to {args.output}")


if __name__ == "__main__":
    main()
