from app.core import model_registry


def test_registry_non_empty():
    assert len(model_registry.list_models()) >= 10


def test_get_model_by_id():
    m = model_registry.get_model("ast-audioset")
    assert m is not None
    assert m.license == "BSD-3-Clause"
    assert m.commercial_ok is True


def test_permissive_models_only_commercial_ok():
    for m in model_registry.permissive_models():
        assert m.commercial_ok is True


def test_all_models_have_license_and_source():
    for m in model_registry.list_models():
        assert m.license
        assert m.source
        assert m.modality
