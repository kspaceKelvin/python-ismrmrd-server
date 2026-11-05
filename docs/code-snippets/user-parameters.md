# Extracting User Parameters from MRD Header

<!-- snippet: user-parameters, mrd-header, userParameterLong, userParameterDouble -->

The [MRD Header](https://ismrmrd.readthedocs.io/en/latest/mrd_header.html) contains a section for userParameters, which are long, double, or short values.  These are often populated through XSLT transformation of a vendor's proprietary format using an [xsl stylesheet](https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/parameter_maps/IsmrmrdParameterMap_Siemens.xsl).  They can be accessed using mrdhelper.py's functions [get_userParameterLong_value](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L52-L58), [get_userParameterDouble_value](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L60-L66), [get_userParameterString_value](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L68-L74), [get_userParameterBase64_value](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L76-L82).

The function signatures are:
```python
def get_userParameterLong_value(metadata, name):
def get_userParameterDouble_value(metadata, name):
def get_userParameterString_value(metadata, name):
def get_userParameterBase64_value(metadata, name):
```

**Usage example:**
```python
dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms
```

## Reference

- **Source file:** `spectroscopy.py`
- **Permalink:** [spectroscopy.py#L247](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/spectroscopy.py#L247)
- **Search anchor:** `get_userParameterDouble_value`
