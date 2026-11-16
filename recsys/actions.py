"""The repair actions the model predicts, how they are recognised in recommendation text, and the
standard sentence shown in the app for each. The keyword rules create the training labels; they are not
surveyor decisions (see docs)."""

import re

# action -> regular expression that marks it in a recommendation text (used to build training labels)
ACTION_RULES = {
    "tile_joint_grouting": r"grout",
    "epoxy_grouting": r"epoxy",
    "chemical_coating": r"chemical|coating|\bcoat\b|brush|app membrane|sbs",
    "brick_bat_coba": r"brick\s*bat|coba",
    "plumbing_repair": r"plumb|pipe|nahani|nahin|trap|drain|flush|sanitary|wc\b|commode",
    "crack_filling": r"crack|fill(?:ing)?\b|hair\s*line",
    "sealant": r"sealant|sealent|silicon|caulk|\bpu\b|polyurethane",
    "plaster_repair": r"plaster|chipping|putty|patch",
    "painting": r"paint|primer|re-?coat",
    "floor_breaking": r"break|dismantl|chipping out|remove",
    "retiling": r"re-?til|new tile|tile (?:replace|fix)|replace.*tile|tiling",
    "injection": r"inject|pressure",
    "outlet_patch": r"outlet|patch",
    "fillet": r"fillet|filet|watta|angle",
    "inspection_check": r"check|inspect|monitor|further|investigat|test",
}
ACTIONS = list(ACTION_RULES)
_COMPILED = {name: re.compile(rule, re.I) for name, rule in ACTION_RULES.items()}

LABEL = {
    "tile_joint_grouting": "Tile joint grouting",
    "epoxy_grouting": "Epoxy grouting",
    "chemical_coating": "Chemical coating",
    "plumbing_repair": "Plumbing repair",
    "crack_filling": "Crack filling",
    "sealant": "Sealant application",
    "plaster_repair": "Plaster repair",
    "painting": "Painting",
    "floor_breaking": "Floor breaking",
    "retiling": "Retiling",
    "outlet_patch": "Outlet patching",
    "inspection_check": "Further inspection",
}

STEP = {
    "tile_joint_grouting": "Grout the open tile joints.",
    "epoxy_grouting": "Carry out epoxy grouting of the tile joints.",
    "chemical_coating": "Apply a waterproofing chemical coating to the affected surface.",
    "plumbing_repair": "Check and repair the plumbing, nahani trap and pipe leakage.",
    "crack_filling": "Fill the cracks and gaps.",
    "sealant": "Apply sealant at the joints and junctions.",
    "plaster_repair": "Repair the plaster and patch the damaged surface.",
    "painting": "Repaint the affected surface once it has dried.",
    "floor_breaking": "Break open the affected floor portion to carry out the repair.",
    "retiling": "Retile the affected area.",
    "outlet_patch": "Patch the outlet and drain junctions.",
    "inspection_check": "Carry out a further inspection and monitor the area.",
}


def action_flags(recommendation: str) -> list:
    """1/0 for every action in ACTIONS, from a recommendation text."""
    return [1 if _COMPILED[a].search(recommendation or "") else 0 for a in ACTIONS]
