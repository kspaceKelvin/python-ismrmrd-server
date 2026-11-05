# Calling External Binaries

<!-- snippet: external-binary-call, subprocess-run, lcmodel -->
It is often desirable to invoke external executables as part of a processing pipeline.  This can be done using `subprocess.run`, then capture stdout/stderr for logging and troubleshooting.  Note that input data for the external executable must be written to a file separately and the result must be parsed afterwards.  When packaging into a Docker image, the binaries (and its dependencies) must be [copied into the image](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e/Dockerfile#L58).  The [LCModel-MRD-App](https://github.com/kspaceKelvin/LCModel-MRD-App) example implements this functionality to integrate [LCModel](https://lcmodel.com/lcmodel.shtml).

```python
cmd = "some_external_program"
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    logging.error(f"External program call failed with message:\n{result.stderr}")
    return None
```

**Notes:**
- `capture_output=True` and `text=True` make it easy to persist logs as strings.
- Handle failures explicitly by checking `result.returncode` and surfacing `result.stderr`.

## Reference
- **Source file:** `lcmodel.py`
- **Permalink:** [lcmodel.py#L293-L298](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e/lcmodel.py#L293-L298)
- **Search anchor:** `subprocess.run`

