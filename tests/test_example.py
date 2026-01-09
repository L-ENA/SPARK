"""
Example test module
"""
import pytest
from spark import __version__


def test_version():
    """Test that version is defined"""
    assert __version__ == "0.1.0"


def test_example():
    """Example test case"""
    assert 1 + 1 == 2


class TestExample:
    """Example test class"""

    def test_method(self):
        """Example test method"""
        result = True
        assert result is True
