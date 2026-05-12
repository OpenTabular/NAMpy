import nampy.configs as configs


def test_dedicated_config_classes_are_exported():
    expected = {
        "DefaultEnsembleTreeNAMConfig",
        "DefaultGPNAMConfig",
        "DefaultQNAMConfig",
        "DefaultTreeNAMConfig",
    }

    assert expected <= set(configs.__all__)
    for name in expected:
        assert hasattr(configs, name)
        assert getattr(configs, name)().__class__.__name__ == name
