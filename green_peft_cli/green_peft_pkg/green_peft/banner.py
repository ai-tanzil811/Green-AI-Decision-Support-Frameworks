"""Author / project banner printed at the top of every CLI invocation.

Kept strictly ASCII: Windows consoles default to cp1252, where any non-ASCII
character in this banner renders as a literal question mark.
"""
from __future__ import annotations

import os
import sys

__author__ = 'Ashraful Islam Tanzil'
__affiliation__ = 'United International University'
__github__ = 'https://github.com/ai-tanzil811'
__dataset_doi__ = '10.34740/KAGGLE/DSV/20178095'
__model_doi__ = '10.57967/hf/10690'

ASCII_ART = r"""
   ____                     ____  _____ _____ _____
  / ___|_ __ ___  ___ _ __ |  _ \| ____|  ___|_   _|
 | |  _| '__/ _ \/ _ \ '_ \| |_) |  _| | |_    | |
 | |_| | | |  __/  __/ | | |  __/| |___|  _|   | |
  \____|_|  \___|\___|_| |_|_|   |_____|_|     |_|
"""

_GREEN = '\033[32m'
_DIM = '\033[2m'
_BOLD = '\033[1m'
_RESET = '\033[0m'


def _supports_color(stream) -> bool:
    """Colour only when writing to a real terminal that is not opted out.

    Honours NO_COLOR (https://no-color.org) and FORCE_COLOR, and stays quiet when the
    output is redirected to a file or piped, so captured output has no escape codes.
    """
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return bool(getattr(stream, 'isatty', lambda: False)())


def render(version: str = '', model_status: str = '', color: bool | None = None) -> str:
    """Return the banner as a string. `color` overrides terminal auto-detection."""
    use_color = _supports_color(sys.stdout) if color is None else color
    g, d, b, r = (_GREEN, _DIM, _BOLD, _RESET) if use_color else ('', '', '', '')

    title = 'GreenPEFT' + (f' v{version}' if version else '')
    lines = [
        f'{g}{ASCII_ART.strip(chr(10))}{r}',
        '',
        f'  {b}{title}{r} - sustainability-aware PEFT decision support',
        f'  {d}{__author__} | {__affiliation__}{r}',
        f'  {d}{__github__}{r}',
    ]
    if model_status:
        marker = '[ok]' if model_status == 'VALIDATED' else '[!]'
        lines.append(f'  {d}surrogate status: {marker} {model_status}{r}')
    lines.append(f'  {d}data doi {__dataset_doi__} | model doi {__model_doi__}{r}')
    lines.append('')
    return '\n'.join(lines)


def print_banner(version: str = '', model_status: str = '', stream=None) -> None:
    """Print the banner to stderr by default, so piping stdout stays machine-readable.

    Suppressed entirely by GREENPEFT_NO_BANNER=1.
    """
    if os.environ.get('GREENPEFT_NO_BANNER'):
        return
    out = stream if stream is not None else sys.stderr
    use_color = _supports_color(out)
    print(render(version=version, model_status=model_status, color=use_color), file=out)
