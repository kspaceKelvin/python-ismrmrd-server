# Conda

Conda, mamba, or micromamba can be used to create the project environment.

Conda is a Python environment manager that is useful for creating and maintaining Python packages and their dependencies. It is available either as part of the larger [Anaconda](https://www.anaconda.com/) framework, or separately as part of [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Although not required, it is helpful for setting up an environment for the Python ISMRMD client/server. [Mamba](https://mamba.readthedocs.io/en/latest/) and [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) are drop-in replacements for Conda that are often faster at resolving dependencies. The following instructions are for micromamba, but `micromamba` can be replaced with `conda` if preferred.

The dependency files are [environment.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment.yml) and [environment_windows.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment_windows.yml).

## Basic setup

```bash
git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git
cd python-ismrmrd-server
micromamba create -f environment.yml
micromamba activate mrd
```

On Windows, use [environment_windows.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment_windows.yml).

## Notes

- On Windows, install `ismrmrd` with pip if it is not present:

```bash
pip install ismrmrd
```

- Optional simulation helper package:

```bash
uv pip install git+https://github.com/ismrmrd/ismrmrd-python-tools.git
```

The [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository contains useful simulation libraries, including the code used by [generate_cartesian_shepp_logan_dataset.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/generate_cartesian_shepp_logan_dataset.py). Install it by cloning the repository and running `pip install .` with the trailing dot:

```bash
git clone https://github.com/ismrmrd/ismrmrd-python-tools.git
cd ismrmrd-python-tools
pip3 install .
```

## Full setup

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or another Conda-compatible tool.

On Windows, open PowerShell. Clone the repository and create the environment:

```powershell
git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git
cd python-ismrmrd-server
micromamba create -f environment_windows.yml
micromamba activate mrd
```

For Linux and macOS, use [environment.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment.yml) instead. To use the environment later, run `micromamba activate mrd`.

If `conda.anaconda.org` is blocked, replace the `conda-forge` channel in [environment_windows.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment_windows.yml) with `https://prefix.dev/conda-forge`. The repository does not use Anaconda's `defaults` channel.

Anaconda changed its terms of service in 2024. Miniconda, Miniforge, and micromamba remain free to use; the `conda-forge` channel remains open source.

## Conda download errors

If micromamba reports an error such as:

```text
critical libmamba Multiple errors occurred:
Download error (23) Failed writing received data to disk/application
Subdir conda-forge/noarch not loaded!
```

The `anaconda.org` domain may be blocked. Verify this by opening [anaconda.org/conda-forge](https://anaconda.org/conda-forge) in a browser. A [conda-forge mirror is hosted by prefix.dev](https://prefix.dev/blog/towards_a_vendor_lock_in_free_conda_experience); use `https://prefix.dev/conda-forge` as the channel in [environment_windows.yml](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/environment_windows.yml) if necessary.
