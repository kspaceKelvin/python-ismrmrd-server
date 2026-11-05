# Reading JSON Config Parameters

<!-- snippet: json-config, get_json_config_param, configuration, type-casting -->

A [JSON config](https://github.com/kspaceKelvin/python-ismrmrd-server#31-additional-json-config) can be used to pass run-time parameters to the module.  The [mrdhelper.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrdhelper.py) module has a [`get_json_config_param`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L145) helper function to parse these parameters from within the module.  Examples are provided in [invertcontrast.json](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.json) and [filter.json](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/filter.json).

The `get_json_config_param` function signature is:
```python
def get_json_config_param(config, key, default=None, type='str'):
```

**Usage examples:**
```python
# String parameter (default type)
options = mrdhelper.get_json_config_param(config, 'options')

# Boolean with default
sendOriginal = mrdhelper.get_json_config_param(config, 'sendOriginal', default=False, type='bool')

# Integer with default
filterSize = mrdhelper.get_json_config_param(config, 'filterSize', default=0, type='int')
```

## Reference

- **Source file:** `mrdhelper.py` — `get_json_config_param()`
- **Permalink:** invertcontrast.py [L288](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L288), [L415](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L415), [filter.py#L307](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea28c9e36e8c1634ef94b31be9d353c4b2b/custom/filter.py#L307)
- **Search anchor:** `get_json_config_param`
