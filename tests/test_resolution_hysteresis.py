import pytest

from lidar_mapping.temporal.resolution_hysteresis import (
    ResolutionHysteresis,
)


def test_first_request_applied_immediately():
    controller = ResolutionHysteresis(coarsening_frames=3)

    assert controller.update((0, 0), 0.5) == 0.5


def test_refinement_is_immediate():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 1.0)

    assert controller.update((0, 0), 0.25) == 0.25


def test_coarsening_requires_three_requests():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 0.25)

    assert controller.update((0, 0), 1.0) == 0.25
    assert controller.update((0, 0), 1.0) == 0.25
    assert controller.update((0, 0), 1.0) == 1.0


def test_stable_request_resets_coarsening_counter():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 0.25)
    controller.update((0, 0), 1.0)
    controller.update((0, 0), 0.25)

    assert controller.update((0, 0), 1.0) == 0.25
    assert controller.update((0, 0), 1.0) == 0.25


def test_cells_have_independent_states():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 0.25)
    controller.update((1, 0), 0.5)

    controller.update((0, 0), 1.0)
    controller.update((0, 0), 1.0)
    controller.update((0, 0), 1.0)

    assert controller.get_state((0, 0)) == 1.0
    assert controller.get_state((1, 0)) == 0.5


def test_update_frame():
    controller = ResolutionHysteresis(coarsening_frames=3)

    result = controller.update_frame({
        (0, 0): 0.25,
        (1, 0): 0.5,
    })

    assert result == {
        (0, 0): 0.25,
        (1, 0): 0.5,
    }


def test_remove_cell():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 0.25)
    controller.remove_cell((0, 0))

    assert controller.get_state((0, 0)) is None
    assert controller.active_cell_count() == 0


def test_reset():
    controller = ResolutionHysteresis(coarsening_frames=3)

    controller.update((0, 0), 0.25)
    controller.update((1, 0), 0.5)

    controller.reset()

    assert controller.active_cell_count() == 0


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_resolution_rejected(value):
    controller = ResolutionHysteresis(coarsening_frames=3)

    with pytest.raises(ValueError):
        controller.update((0, 0), value)
