from stockml import __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1
