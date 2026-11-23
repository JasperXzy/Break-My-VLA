import hashlib
from base64 import b64encode
from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts.colab_m2_smolvla import ARTIFACT_MARKER, receive_artifacts


def test_artifact_stream_checks_hash_and_reconstructs_files(tmp_path):
    names = ("baseline.gif", "object_nudge.gif", "lighting_shift.gif", "int8_low_steps.gif")
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"test data")
        archive.writestr("metrics.json", "{}")
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    encoded = b64encode(data).decode()
    chunks = [
        f"{ARTIFACT_MARKER}0 {encoded[:200]}",
        f"{ARTIFACT_MARKER}1 {encoded[200:]}",
    ]
    with pytest.raises(ValueError, match="checksum"):
        receive_artifacts([f"{ARTIFACT_MARKER}sha256 wrong", *chunks], tmp_path)
    assert not list(tmp_path.iterdir())

    receive_artifacts([f"{ARTIFACT_MARKER}sha256 {digest}", *chunks], tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted((*names, "metrics.json"))
    assert (tmp_path / "metrics.json").read_text() == "{}"
