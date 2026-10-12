"""Google palette on dark slate (spec section 9 / Krug ch 12).

Brand colours are used for borders and bars only: as text on #1E1E2E they reach just 4.2-5.4:1
(red fails AA). Text uses Google's own dark-theme tints, which pass WCAG AAA (>= 7:1):
blue 7.8, green 8.4, yellow 11.7, red 9.0, hint grey 7.4, body 12.4, headers 16.4."""
from rich.theme import Theme

# brand: borders, bars, focus rings (WCAG non-text contrast needs 3:1)
BLUE = "#4285F4"
GREEN = "#34A853"
YELLOW = "#FBBC05"
RED = "#EA4335"
SURFACE = "#1E1E2E"

# text on SURFACE (all >= 7:1)
TEXT_BLUE = "#8AB4F8"
TEXT_GREEN = "#81C995"
TEXT_YELLOW = "#FDD663"
TEXT_RED = "#F6AEA9"
HEADER = "#FFFFFF"
BODY = "#D9E0EE"
MUTED = "#A6ADC8"

THEME = Theme({
    "accent": f"bold {TEXT_BLUE}",
    "ok": f"bold {TEXT_GREEN}",
    "warn": f"bold {TEXT_YELLOW}",
    "danger": f"bold {TEXT_RED}",
    "header": f"bold {HEADER}",
    "body": BODY,
    "muted": MUTED,
    "key": f"bold {TEXT_BLUE}",
    "crumb": TEXT_BLUE,
    "crumb.here": f"bold {HEADER}",
})

DIFFICULTY_STYLE = {"Easy": "ok", "Medium": "warn", "Hard": "danger"}
