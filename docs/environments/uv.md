# uv

`uv` can be used as an alternative Python dependency manager.

## Setup

```bash
git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git
cd python-ismrmrd-server
uv sync
```

If required:

```bash
uv python install 3.14
uv sync
```

Optional dependency groups:

```bash
uv sync --extra test
uv sync --extra nii
```

Run scripts through uv:

```bash
uv run python main.py -v
uv run python client.py -G dataset -o phantom_img.h5 phantom_raw.h5
```

Note: `siemens_to_ismrmrd` is not installed by `uv sync`.

## Windows installation

Install uv in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install Git if needed:

```powershell
winget install --id Git.Git -e --source winget
```

`uv` manages Python packages only. Install non-Python tools such as Git and `dos2unix` separately.

Install the simulation helpers when needed:

```bash
uv pip install git+https://github.com/ismrmrd/ismrmrd-python-tools.git
```

If the workflow requires Siemens `.dat` conversion, install `siemens_to_ismrmrd` separately through Conda or Docker.
