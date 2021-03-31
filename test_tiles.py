import pytest

import generate


@pytest.mark.parametrize("filename", ['ra1_tiles.json', 'td_tiles.json'], ids=["RA1", "TD"])
def test_tiles_json(filename):
    for level in generate.import_tiles_file(filename):
        pattern_size = level[0][0].shape[1:]
        template_size = level[0][1].shape[1:]

        for pattern, template, icon in level:
            assert template.shape == icon.shape
            assert template.ndim == 3
            assert pattern.ndim == 3
            assert pattern.shape[1] > template.shape[1] and pattern.shape[2] > template.shape[2]
            assert pattern_size == pattern.shape[1:]
            assert template_size == template.shape[1:]
