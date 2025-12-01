import numpy as np

from projeto_toi.datasets import npz_dataset as dset


def _extract_core(x, h, w, target_size=dset.TARGET_SIZE):
    pad_h = target_size - h
    pad_w = target_size - w
    top = pad_h // 2
    left = pad_w // 2
    return x[:, top : top + h, left : left + w]


def test_make_channels_from_masks_preserves_stats():
    cube = np.zeros((4, 2, 3), dtype=float)
    cube[0] = 1.0
    cube[1] = 3.0   # before -> media 2.0
    cube[2] = 2.0
    cube[3] = 6.0   # during -> media 4.0
    mask_before = np.array([1, 1, 0, 0], dtype=int)
    mask_during = np.array([0, 0, 1, 1], dtype=int)

    z = {"cube": cube, "mask_before": mask_before, "mask_during": mask_during}
    X = dset.make_channels_from_masks(z)
    assert X.shape == (4, dset.TARGET_SIZE, dset.TARGET_SIZE)

    core = _extract_core(X, h=2, w=3)
    np.testing.assert_allclose(core[0], np.full((2, 3), 2.0))
    np.testing.assert_allclose(core[1], np.full((2, 3), 4.0))
    np.testing.assert_allclose(core[2], np.full((2, 3), 2.0))  # diff
    np.testing.assert_allclose(core[3], cube.std(axis=0))


def test_make_channels_from_cube_basic_stats():
    cube = np.stack(
        [
            np.full((2, 3), 1.0),
            np.full((2, 3), 2.0),
            np.full((2, 3), 3.0),
        ]
    )  # T=3, H=2, W=3
    X = dset.make_channels_from_cube(cube)
    assert X.shape == (4, dset.TARGET_SIZE, dset.TARGET_SIZE)

    core = _extract_core(X, h=2, w=3)
    np.testing.assert_allclose(core[0], np.full((2, 3), 2.0))               # median
    np.testing.assert_allclose(core[1], np.full((2, 3), cube.std(axis=0)))  # std
    np.testing.assert_allclose(core[2], np.full((2, 3), 1.0))               # min
    np.testing.assert_allclose(core[3], np.full((2, 3), 1.0))               # median - min
