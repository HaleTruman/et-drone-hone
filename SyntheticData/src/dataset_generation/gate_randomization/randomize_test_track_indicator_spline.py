"""Randomize only the BP_Track_Indicator_Test_Spline actor's spline points."""

from pathlib import Path
from importlib import reload
import random
import sys


SCRIPT_PATH = Path(
    globals().get(
        "__file__",
        r"C:\Users\brend\Projects\UE_5\TestProject\ML\SyntheticData\src\dataset_generation\gate_randomization\randomize_test_track_indicator_spline.py",
    )
).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unreal

from SyntheticData.src.dataset_generation.config import TRACK_MAX_WAYPOINTS, TRACK_MIN_WAYPOINTS
from SyntheticData.src.dataset_generation.gate_randomization import spline_actor
from SyntheticData.src.dataset_generation.gate_randomization.track import generate_track_layout
from SyntheticData.src.dataset_generation.unreal_editor import invalidate_viewports, load_level_if_needed, require_unreal_editor_python


reload(spline_actor)
TARGET_SPLINE_ACTOR_LABEL = "BP_Track_Indicator_Test_Spline"


def main():
    require_unreal_editor_python()
    load_level_if_needed()

    actor = spline_actor.find_actor_by_label(TARGET_SPLINE_ACTOR_LABEL)
    if not actor:
        raise RuntimeError(f"Could not find actor '{TARGET_SPLINE_ACTOR_LABEL}'.")

    point_count_seed = random.randint(TRACK_MIN_WAYPOINTS, TRACK_MAX_WAYPOINTS)
    layout = generate_track_layout(point_count_seed)
    spline_points = layout["spline_points"]

    with unreal.ScopedEditorTransaction("Randomize Track Indicator Test Spline"):
        actor.modify()
        spline_component = spline_actor.actor_spline_component(actor)
        spline_component.modify()
        spline_actor.update_spline_component_points(spline_component, spline_points)
        spline_actor.rebuild_track_meshes(actor, spline_component)

    invalidate_viewports()
    unreal.log(
        f"Randomized only '{TARGET_SPLINE_ACTOR_LABEL}' with {len(spline_points)} spline point(s) "
        f"from random seed count {point_count_seed}."
    )
    return actor

if __name__ == "__main__":
    main()
