"""Children (the llama-servers) end with Localive, even when it is killed outright."""

import os
import subprocess
import sys
import time

import pytest

PARENT = """
import subprocess, sys, time
from localive import children
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         preexec_fn=children.preexec())
children.bind(child)
print(child.pid, flush=True)
time.sleep(60)
"""


def alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout  # fmt: skip
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="Windows and Linux only")
def test_child_dies_when_localive_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    assert alive(child)
    parent.kill()  # like closing the console window: no clean shutdown
    parent.wait()
    deadline = time.monotonic() + 10
    while alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not alive(child)
