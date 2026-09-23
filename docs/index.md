# Python ISMRMRD Server

This repository is a reference implementation of the [ISMRM Raw Data (MRD) network streaming format](https://ismrmrd.readthedocs.io/en/latest/index.html).  It enables the integration of MR image reconstruction and analysis algorithms in a vendor neutral, cross-platform format that can be readily distributed through the use of container formats like [Docker](https://www.docker.com).

## Start Here

- New users should begin with [Quickstart](getting-started/quickstart.md).
- For setup options, use the pages under **Environments**.
- If you want to add your own processing, read [Custom Modules](getting-started/custom-modules.md).

## Example Applications

The following are examples of algorithms that are implemented using this repo's framework:

- [Variational Network (VarNet) Image Reconstruction](https://github.com/kspaceKelvin/VarNet-MRD-App) utilizes a variational network to reconstruct highly accelerated raw k-space data into images.  It utilizes the PyTorch packages and includes pre-trained network weights.

- [LCModel Spectroscopy Analysis](https://github.com/kspaceKelvin/LCModel-MRD-App) is available as an MRD-compatible container application.  It integrates a compiled executable of the popular [LCModel](https://lcmodel.com/lcmodel.shtml) software and outputs include RGB images of the classic LCModel PostScript images and metabolite concentrations stored in the image metadata.
