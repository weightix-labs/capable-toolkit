import capable_toolkit as ct


def test_all_public_exports_present():
    expected = {
        "jailbreak", "expert", "optimize", "think", "vision",
        "merge", "distill", "watermark", "verify_watermark",
        "poison", "immunize", "extract_schema", "validate_output",
        "CapableModel", "Transform", "load_model",
        "resolve_source", "ResolvedSource",
        "CapableError", "SourceResolutionError", "BackendUnavailable",
        "OperationError", "SchemaError", "__version__",
    }
    assert expected.issubset(set(ct.__all__))
    for name in expected:
        assert hasattr(ct, name), f"missing export: {name}"


def test_version():
    assert ct.__version__ == "0.1.0"


def test_all_functions_are_callable():
    for name in ["jailbreak", "expert", "optimize", "think", "vision",
                 "merge", "distill", "watermark", "poison", "immunize",
                 "extract_schema"]:
        assert callable(getattr(ct, name))
