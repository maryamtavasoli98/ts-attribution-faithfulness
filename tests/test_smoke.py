import importlib


def test_package_imports():
    assert importlib.import_module("ts_attribution_faithfulness").__version__
