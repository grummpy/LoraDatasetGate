import pytest

from loradatasetgate.samples import generate


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / "dataset"
    generate(root)
    return root
