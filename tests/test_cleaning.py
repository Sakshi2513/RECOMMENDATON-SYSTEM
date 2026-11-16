import json

from recsys import config
from recsys.cleaning import clean, mask_pii

TYPOS = json.loads(config.TYPO_PATH.read_text(encoding="utf-8"))


def test_masks_flat_room_floor_numbers():
    out = clean("Dampness in Flat No-302, Room 4, 3rd floor", {})
    assert "<flat>" in out and "<room>" in out and "<floor>" in out
    assert "302" not in out


def test_removes_contact_details():
    out = clean("call 9123456780 or mail a.b@x.com, see https://x.y/z", {})
    assert "9123456780" not in out and "@" not in out and "http" not in out


def test_fixes_typos_and_lowercases():
    assert (
        clean("Dampness in Celing of Bathoom", {"celing": "ceiling", "bathoom": "bathroom"})
        == "dampness in ceiling of bathroom"
    )
    assert clean("tile joints kitchn", TYPOS) == "tile joints kitchen"


def test_empty_and_none():
    assert clean("", {}) == "" and clean(None, {}) == ""


def test_mask_pii_for_storage():
    assert mask_pii("call 9123456780 mail x@y.com") == "call [phone] mail [email]"
