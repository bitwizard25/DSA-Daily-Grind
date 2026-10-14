"""Single source of 'now'. GOOGLE_PREP_NOW (ISO format) pins it for demos and tests."""
import os
from datetime import datetime


def now() -> datetime:
    value = os.environ.get("GOOGLE_PREP_NOW")
    return datetime.fromisoformat(value) if value else datetime.now()
