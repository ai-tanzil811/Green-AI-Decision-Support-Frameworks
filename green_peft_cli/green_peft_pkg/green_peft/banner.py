
from __future__ import annotations

import os
import sys

__author__ = 'Ashraful Islam Tanzil'
__affiliation__ = 'United International University'
__github__ = 'https://github.com/ai-tanzil811'
__dataset_doi__ = '10.34740/KAGGLE/DSV/20178095'
__model_doi__ = '10.57967/hf/10690'

# Google Signature 24-bit RGB ANSI Palette
_G_BLUE   = '\033[38;2;66;133;244m'   # #4285F4 Primary Accent
_G_RED    = '\033[38;2;234;67;53m'    # #EA4335 Alert / Caution
_G_YELLOW = '\033[38;2;251;188;5m'   # #FBBC05 Highlight / Warnings
_G_GREEN  = '\033[38;2;52;168;83m'    # #34A853 Success / Validated
_CYAN     = '\033[38;2;24;183;218m'   # #18B7DA Info Links
_GRAY     = '\033[38;2;154;160;166m'  # #9AA0A6 Secondary / Muted
_BOLD     = '\033[1m'
_RESET    = '\033[0m'


def _supports_color(stream) -> bool:
    """Colour only when writing to a real terminal that is not opted out."""
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return bool(getattr(stream, 'isatty', lambda: False)())


def _supports_unicode() -> bool:
    """Check if the execution environment supports UTF-8 box characters."""
    encoding = getattr(sys.stdout, 'encoding', '') or ''
    if os.name == 'nt':
        return encoding.lower().replace('-', '') in ('utf8', 'utf-8')
    return True


def render(version: str = '0.3.0', model_status: str = 'VALIDATED', color: bool | None = None) -> str:
    """Return the Google CLI-style colored banner as a string."""
    use_color = _supports_color(sys.stderr) if color is None else color
    use_unicode = _supports_unicode()

    # Color tokens
    gb, gr, gy, gg, cy, d, b, r = (
        (_G_BLUE, _G_RED, _G_YELLOW, _G_GREEN, _CYAN, _GRAY, _BOLD, _RESET)
        if use_color else ('', '', '', '', '', '', '', '')
    )

    # Box-drawing glyphs with pure ASCII fallbacks
    tl = '╭' if use_unicode else '+'
    tr = '╮' if use_unicode else '+'
    bl = '╰' if use_unicode else '+'
    br = '╯' if use_unicode else '+'
    h  = '─' if use_unicode else '-'
    v  = '│' if use_unicode else '|'
    dot = '●' if use_unicode else '*'

    # Google-style 4-color title branding: G(Blue) r(Red) e(Yellow) e(Green) n(Blue) PEFT(Blue+Bold)
    colored_logo = f"{gb}G{r}{gr}r{r}{gy}e{r}{gg}e{r}{gb}n{r}{gb}{b}PEFT{r}"
    ver_str = f"v{version}" if version else "v0.3.0"

    # Status Chip Badge
    if model_status == 'VALIDATED':
        status_chip = f"{gg}{dot} {model_status}{r}"
    elif model_status:
        status_chip = f"{gy}{dot} {model_status}{r}"
    else:
        status_chip = f"{d}{dot} READY{r}"

    # Header Card Top Line
    card_width = 68
    top_border = f"{gb}{tl}{h * card_width}{tr}{r}"
    bottom_border = f"{gb}{bl}{h * card_width}{br}{r}"

    # Structured Banner Lines
    lines = [
        top_border,
        f"{gb}{v}{r}  {colored_logo} {d}[{ver_str}]{r}  {d}—{r}  {b}Green AI Decision Support Framework{r}  {status_chip}  {gb}{v}{r}",
        f"{gb}{v}{r}  {d}Google Cloud / Research CLI Architecture{r}" + " " * 27 + f"{gb}{v}{r}",
        f"{gb}{h * (card_width + 2)}{r}",
        f"  {gb}{b}AUTHOR{r}       {b}{__author__}{r} {d}({__affiliation__}){r}",
        f"  {cy}{b}RESOURCES{r}    {cy}{__github__}{r}",
        f"  {gg}{b}DATA DOI{r}     {d}{__dataset_doi__}{r}",
        f"  {gy}{b}MODEL DOI{r}    {d}{__model_doi__}{r}",
        bottom_border,
    ]

    return '\n'.join(lines)


def print_banner(version: str = '', model_status: str = '', stream=None) -> None:
    """Print the banner to stderr by default, so piping stdout stays machine-readable."""
    if os.environ.get('GREENPEFT_NO_BANNER'):
        return
    out = stream if stream is not None else sys.stderr
    use_color = _supports_color(out)
    print(render(version=version, model_status=model_status, color=use_color), file=out)