# JSON Config

The client can pass additional runtime configuration to a selected module through a JSON file.  This is compatible with the Siemens Open Recon user interface parameter configuration.

## Convention

- File name: `<module>.json`
- Example: [`invertcontrast.json`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.json)

The client looks for this file in the current working directory and sends options with the request.

Example options in `invertcontrast.json` include ROI overlays, colormap selection, and RGB output.

## Example

```json
{
	"version": "1.1.0",
	"parameters": {
		"options": "roi",
		"sendOriginal": "False"
	}
}
```

The module reads values with `mrdhelper.get_json_config_param(config, key, default, type)`. The helper supports string, boolean, integer, and other requested value types and applies the supplied default when a parameter is absent.
