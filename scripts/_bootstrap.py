"""Make `kupe_asr` importable when a script is run as a file path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
