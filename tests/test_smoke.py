import subprocess
import sys

from stockml import __version__


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 1


def test_stockml_models_do_not_eagerly_load_lightgbm():
    code = "import sys; import stockml.models; assert 'lightgbm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
