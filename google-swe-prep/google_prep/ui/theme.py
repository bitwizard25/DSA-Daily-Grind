"""Google-inspired terminal palette (spec section 9A)."""
from rich.theme import Theme

BLUE = "#4285F4"
GREEN = "#34A853"
YELLOW = "#FBBC05"
RED = "#EA4335"
SURFACE = "#1E1E2E"
HEADER = "#FFFFFF"
BODY = "#D9E0EE"
MUTED = "#6C7086"

THEME = Theme({
    "accent": f"bold {BLUE}",
    "ok": f"bold {GREEN}",
    "warn": f"bold {YELLOW}",
    "danger": f"bold {RED}",
    "header": f"bold {HEADER}",
    "body": BODY,
    "muted": MUTED,
    "key": f"bold {BLUE}",
})

DIFFICULTY_STYLE = {"Easy": "ok", "Medium": "warn", "Hard": "danger"}
