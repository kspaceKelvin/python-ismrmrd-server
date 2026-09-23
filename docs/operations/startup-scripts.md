# Startup Scripts

These scripts are used in FIRE/chroot workflows:

- [`start-fire-python-server.sh`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/start-fire-python-server.sh)
- [`sync-code-and-start-fire-python-server.sh`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/sync-code-and-start-fire-python-server.sh)
- [`start-fire-python-server-with-data-storage.sh`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/start-fire-python-server-with-data-storage.sh)

## Typical use

- Use `start-fire-python-server.sh` for stable startup.
- Use `sync-code-and-start-fire-python-server.sh` during rapid development when syncing host code.
- Use `start-fire-python-server-with-data-storage.sh` when you need to persist incoming data for offline analysis.

## Script details

`start-fire-python-server.sh` accepts one optional log-file argument. If omitted, logging output is discarded.

`sync-code-and-start-fire-python-server.sh` copies files from `/tmp/share/code/` to `/opt/code/python-ismrmrd-server/` before starting the server. In FIRE, the shared host folder is typically `%CustomerIceProgs%\fire\share\code\`. This is useful for development, but should not be used in stable projects because it overwrites files.

`start-fire-python-server-with-data-storage.sh` starts the server with incoming data storage enabled. Files are written under `/tmp/share/saved_data`, which maps to `%CustomerIceProgs%\fire\share\saved_data` on the Windows host. Raw data can be large and may exhaust available disk space.
