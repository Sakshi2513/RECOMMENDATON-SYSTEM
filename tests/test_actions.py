from recsys.actions import ACTIONS, LABEL, STEP, action_flags


def test_every_action_in_app_has_label_and_step():
    for name in LABEL:
        assert name in ACTIONS and name in STEP


def test_flags():
    flags = dict(zip(ACTIONS, action_flags("Tile joint grouting and plumbing repair of nahani trap"), strict=True))
    assert flags["tile_joint_grouting"] == 1 and flags["plumbing_repair"] == 1
    assert flags["painting"] == 0
    assert sum(action_flags("")) == 0
