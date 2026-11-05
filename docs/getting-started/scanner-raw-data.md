# Scanner Raw Data

Raw data from MRI scanners can be converted to MRD using converters such as:

- `siemens_to_ismrmrd`
- `ge_to_ismrmrd`
- `philips_to_ismrmrd`
- `bruker_to_ismrmrd`

For Siemens `.dat` files:

```bash
siemens_to_ismrmrd -Z -f gre.dat -o gre_raw.h5
```

Then run a reconstruction session:

```bash
python client.py -c invertcontrast -o gre_img.h5 gre_raw_2.h5
```

`invertcontrast.py` is a basic example and does not implement advanced reconstruction for undersampled acquisitions.

Start the server first with [main.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/main.py), then run the client with [client.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/client.py).

The converter projects are available from GitHub: [siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd), [ge_to_ismrmrd](https://github.com/ismrmrd/ge_to_ismrmrd), [philips_to_ismrmrd](https://github.com/ismrmrd/philips_to_ismrmrd), and [bruker_to_ismrmrd](https://github.com/ismrmrd/bruker_to_ismrmrd).

For a multi-RAID Siemens file, `-Z` converts all measurements. Several output files may be created, such as `gre_raw_1.h5` and `gre_raw_2.h5`; dependency data is typically in earlier files and the main acquisition is in the last numbered file.

The options used above are:

```text
-Z  Convert all acquisitions from a multi-measurement (multi-RAID) file
-f  Input Siemens raw data .dat file
-o  Output MRD raw data .h5 file
```

If using the uv environment, install `siemens_to_ismrmrd` separately because it is not installed by `uv sync`.
