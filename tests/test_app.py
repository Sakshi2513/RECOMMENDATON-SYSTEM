"""End-to-end checks of the Streamlit page, using Streamlit's headless test runner."""

from unittest import mock

import pytest
from streamlit.testing.v1 import AppTest

APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)


def run():
    return AppTest.from_file(APP, default_timeout=120).run()


def click(at, label):
    [b for b in at.button if b.label == label][0].click()
    at.run()


def generate(at, neg, pos):
    at.text_area(key="neg").set_value(neg)
    at.text_area(key="pos").set_value(pos)
    at.run()
    click(at, "Generate recommendation")


def test_generate_and_approve_saves_to_log():
    at = run()
    generate(at, "Dampness at skirting level", "Tile joints open in bathroom, nahani trap gaps")
    assert not at.exception
    assert [c for c in at.checkbox if c.value], "at least one action should be suggested"
    click(at, "Approve")
    assert [s.value for s in at.success] == ["Saved to the correction log."]


def test_empty_input_warns():
    at = run()
    click(at, "Generate recommendation")
    assert at.warning and not at.exception


def test_load_example_fills_both_boxes():
    at = run()
    click(at, "Load example")
    assert at.text_area(key="neg").value and at.text_area(key="pos").value


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    at = run()
    assert not [b for b in at.button if b.label == "Generate recommendation"]
    at.text_input[0].set_value("wrong")
    at.button[0].click()
    at.run()
    assert at.error
    at.text_input[0].set_value("secret")
    at.button[0].click()
    at.run()
    assert [b for b in at.button if b.label == "Generate recommendation"]


def test_refuses_to_open_unprotected_on_streamlit_cloud():
    with mock.patch("pathlib.Path.exists", return_value=True):
        at = run()
    assert at.error and not [b for b in at.button if b.label == "Generate recommendation"]


def test_admin_panel_requires_admin_password(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "boss")
    at = run()
    at.text_input(key="admin_pw").set_value("nope")
    at.run()
    assert at.error
    at.text_input(key="admin_pw").set_value("boss")
    at.run()
    assert any("saved corrections" in m.value for m in at.markdown)
