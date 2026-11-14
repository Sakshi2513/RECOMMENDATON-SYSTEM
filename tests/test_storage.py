from recsys.storage import CorrectionLog


def test_add_masks_personal_data_and_exports(tmp_path):
    log = CorrectionLog(tmp_path)
    log.add(surveyor="R. Sharma 9123456780", negative="call 9123456780", positive="mail a@b.com", actions=["plumbing_repair"],
            shown_text="1. x", final_text="1. y", status="edited", model_version="abc")
    assert log.count() == 1
    csv_text = log.to_csv().decode("utf-8-sig")
    assert "9123456780" not in csv_text and "a@b.com" not in csv_text
    assert "[phone]" in csv_text and "[email]" in csv_text and "plumbing_repair" in csv_text
