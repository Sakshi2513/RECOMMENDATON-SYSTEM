"""Text cleaning. The same function cleans the training data and the text typed into the app,
so the model always sees text in the same form."""

import re

BOILERPLATE = [r"during the inspection,? the following observations were noted:?", r"private\s*&\s*confidential"]
FLAT = re.compile(r"\b(flat|unit|apartment|apt)s?\.?\s*(no\.?|number|#)?\s*[-:.]?\s*\d+\s*[a-z]?\b")
ROOM = re.compile(r"\broom\s*(no\.?|number|#)?\s*[-:.]?\s*\d+\b")
FLOOR = re.compile(r"\b\d+\s*(st|nd|rd|th)\s+floors?\b|\bfloors?\s*(no\.?)?\s*[-:.]?\s*\d+\b")
PHONE = re.compile(r"(?<!\d)(\+?91[- ]?)?[6-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"https?://\S+|www\.\S+")
LIST_NUMBER = re.compile(r"(?:(?<=\s)|^)\(?\d{1,2}[\).]\s+(?=\S)")


def clean(text, typo_map):
    """Lowercase, remove contact details and boilerplate, mask flat/room/floor numbers, fix common typos."""
    t = text or ""
    t = EMAIL.sub(" ", PHONE.sub(" ", URL.sub(" ", t)))
    t = t.lower()
    for pattern in BOILERPLATE:
        t = re.sub(pattern, " ", t)
    t = LIST_NUMBER.sub("; ", " " + t)
    t = FLOOR.sub(" <floor> ", ROOM.sub(" <room> ", FLAT.sub(" <flat> ", t)))
    t = t.replace("�", " ").replace(" ", " ")
    t = re.sub(r"[^\w<>&%/,.;:()\-+ ]", " ", t)
    t = "".join(typo_map.get(w, w) for w in re.split(r"(\s+)", t))
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(\s*[;:]\s*)+", "; ", t).strip(" ;:,.-")
    return re.sub(r"\s+", " ", t)


def mask_pii(text):
    """Remove phone numbers, e-mail addresses and links before text is stored."""
    t = EMAIL.sub("[email]", text or "")
    t = URL.sub("[link]", t)
    return PHONE.sub("[phone]", t)
