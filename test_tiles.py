import pytest

import generate


@pytest.mark.parametrize("filename", ['ra1_tiles.json', 'td_tiles.json'], ids=["RA1", "TD"])
def test_tiles_json(filename):
    for pattern, template, icon in generate.import_tiles_file(filename):
        assert template.shape == icon.shape
        assert template.ndim == 3
        assert pattern.ndim == 3
