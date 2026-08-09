import subprocess
import sys
from importlib.metadata import version

from stockml import __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1


def test_version_matches_package_metadata():
    assert __version__ == version("stockml")


def test_stockml_models_do_not_eagerly_load_lightgbm():
    code = "import sys; import stockml.models; assert 'lightgbm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
