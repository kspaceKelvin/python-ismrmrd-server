# Quickstart

Use [GitHub Codespaces](https://github.com/features/codespaces) with the [RunClientServerRecon.ipynb](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/RunClientServerRecon.ipynb) notebook to run a complete client/server reconstruction workflow.

GitHub Codespace is a cloud-hosted, web-oriented service that allows a complete development environment to be set up with a single click.  It uses Docker images that are set up using the [`devcontainer.json`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/.devcontainer/devcontainer.json) file in this repo.

## Start with GitHub Codespaces

A GitHub account is required to use GitHub Codespaces.  Free personal usage includes 120 hours per month of compute time. 

1. **Open the repository in Codespaces**
    - Navigate to the [python-ismrmrd-server repository](https://github.com/kspaceKelvin/python-ismrmrd-server) on GitHub
    - Click the green **Code** button → select **Codespaces** → click **Create codespace on master**
    - Wait for the environment to initialize (typically <2 minutes)
    - When complete, a web version of Visual Studio Code will be shown
    - The environment is pre-configured and all required Python libraries are installed

2. **Open RunClientServerRecon.ipynb**
    - In VS Code, open the file browser and navigate to `RunClientServerRecon.ipynb` in the workspace root
    - Click on the notebook to open it

3. **Execute the notebook cells**
    - The notebook contains the following workflow steps:
        - Generate sample raw k-space data (Shepp-Logan phantom)
        - Visualize the k-space data
        - Start the MRD server (this is done in the Run and Debug tab of Visual Studio code, not within the notebook itself)
        - Send data for reconstruction using the client
        - Display the results
    - Run each cell in order by clicking the play button or pressing `Ctrl+Enter`
    - Output images and metadata are displayed in the output cells

4. **View results**
    - Reconstructed images are displayed in the output cells
    - Reconstruction metadata is also shown
    - Generated data files are saved to the `data/` directory

## Next steps

- Explore the [architecture documentation](../architecture/code-design.md) to understand the server components
- Implement an [example custom module](../getting-started/custom-modules.md) for image filtering
- See the **environments** section for alternative setup options ([Conda](../environments/conda.md), [uv](../environments/uv.md), [Docker](../environments/docker.md))
