# Saving Incoming Data

The server can save incoming data streams for later offline use.

This is useful when an MRI scanner is the client, because saved image data is not normally stored in the scanner's raw data file and can otherwise require offline simulation of the scanner reconstruction.

## Enable saving globally

```bash
python main.py -s -S /tmp
```

- `-s` enables saving
- `-S` sets the destination folder

The default storage path is under `/tmp/share/saved_data`.

Saving is disabled by default. Data files are named using the current date and time.

## Enable per session

Use client config `savedataonly` to save incoming data without processing.

In `savedataonly` mode, incoming raw or image data is saved, no processing is performed, and no images are sent back to the client.

Saved files use MRD `.h5` format and can be supplied later to `client.py` for offline simulations. The startup script [start-fire-python-server-with-data-storage.sh](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/start-fire-python-server-with-data-storage.sh) enables `-s` and stores files under `/tmp/share/saved_data`.
