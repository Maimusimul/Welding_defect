import pytest
from PyQt5.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication exists with offscreen platform."""
    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv + ["-platform", "offscreen"])
    return app
