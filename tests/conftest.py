import pytest

import app.main
from app.usage import tokens


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    # Tests never download tiktoken's encoding (an 18 s network fetch the first time): apps
    # started with their lifespan skip the background load, and counts are chars/4 unless a
    # test sets an encoding itself.
    monkeypatch.setattr(app.main, "load_encoding_in_background", lambda: None)
    monkeypatch.setattr(tokens, "_encoding", None)
