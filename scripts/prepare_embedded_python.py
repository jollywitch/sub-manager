from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


EMBEDDED_VERSION = "3.13.8"
EMBEDDED_URL = f"https://www.python.org/ftp/python/{EMBEDDED_VERSION}/python-{EMBEDDED_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_pth_configuration(runtime_dir: Path) -> None:
    pth_files = sorted(runtime_dir.glob("python*._pth"))
    for pth_file in pth_files:
        lines = pth_file.read_text(encoding="utf-8").splitlines()
        cleaned = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        if "Lib\\site-packages" not in cleaned:
            cleaned.append("Lib\\site-packages")
        if "import site" not in cleaned:
            cleaned.append("import site")
        pth_file.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> int:
    project_root = _project_root()
    output_dir = project_root / "build" / "runtime-python"
    archive_path = output_dir / "python-embed.zip"
    python_exe = output_dir / "python.exe"
    get_pip_path = output_dir / "get-pip.py"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading embedded Python runtime from {EMBEDDED_URL}")
    _download_file(EMBEDDED_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    archive_path.unlink(missing_ok=True)

    if not python_exe.exists():
        raise RuntimeError(f"Embedded Python executable not found after extraction: {python_exe}")

    _ensure_pth_configuration(output_dir)

    print(f"Downloading get-pip from {GET_PIP_URL}")
    _download_file(GET_PIP_URL, get_pip_path)
    try:
        env = os.environ.copy()
        result = subprocess.run(
            [str(python_exe), str(get_pip_path), "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=env,
        )
    finally:
        get_pip_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Installing pip into embedded Python failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )

    print(f"Embedded Python prepared at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
