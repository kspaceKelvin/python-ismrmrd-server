# Storing Data Between Runs

<!-- snippet: data-between-runs, persistent-storage, reference-data, pickle, LoadReferenceData -->

It is often necessary to use information from multiple acquisitions during an analysis.  As the server may be restarted in between these acquisitions, it is recommended to save the necessary data outside of local variables.  The [LCModel-MRD-App example](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e814e265f4a0ffcc1d1d5efd334a7f6a8c/lcmodel.py#L133-L160) saves water-unsuppressed reference data and their MRD header to disk.  A later run with water suppressed acquisitions retrieves that reference for combined analysis.

In this model, a writable folder must be shared by both runs.  When running inside a container, use a mounted folder or persistent volume as the container may not be reused between runs.  In the below example, the variable `storeFolder` (set to `/tmp/share/store`) is used as the writable data store.

## Save data in the first run

Data is stored as pickled files with the file name containing information required for easy selection without opening each file.  It contains the study date/time, an ID unique to the current subject's study instance, the protocol name, and a keyword.  The data stored in this example are the MRD header and images, but additional data can be added as well.

```python
import os
import pickle
import re
from datetime import datetime, timedelta

storeFolder = os.path.join('data', 'store')
os.makedirs(storeFolder, exist_ok=True)

studyDateTime = datetime.combine(
    mrdHeader.studyInformation.studyDate.to_date(),
    mrdHeader.studyInformation.studyTime.to_time(),
)
studyInstanceUID = mrdHeader.studyInformation.studyInstanceUID
keyword = 'lcmodel'  # Use the same keyword in both runs; no underscores
protocolName = mrdHeader.measurementInformation.protocolName
protocolName = re.sub(r'[^A-Za-z0-9_.-]', '-', protocolName)

name = (
    f"{studyDateTime:%Y-%m-%d_%H-%M-%S}_"
    f"{studyInstanceUID}_{keyword}_{protocolName}.pkl"
)
with open(os.path.join(storeFolder, name), 'wb') as file:
    pickle.dump({'mrdHeader': mrdHeader, 'images': images}, file)
```

The complete MRD header and image data and metadata are saved for future use.  After saving, the module then can continue analysis if possible, or close the session without returning data.

## Retrieve data in a later run

The following helper is adapted from [`LoadReferenceData()`](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e814e265f4a0ffcc1d1d5efd334a7f6a8c/lcmodel.py#L509-L574).  It matches the study UID and keyword, limits the reference age (24 hours by default), and loads the closest preceding reference.  It returns `None` when no match is available.

```python
def LoadReferenceData(storeFolder, studyDateTime, studyInstanceUID, keyword,
                      maxTimeDeltaHours=24):
    if not os.path.isdir(storeFolder):
        return None

    fileMatch = None
    bestAge = None
    pattern = (
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_'
        r'(?P<uid>[.\d]+)_(?P<keyword>[^_]+)_(?P<protocol>.+)\.pkl$'
    )

    for name in sorted(os.listdir(storeFolder)):
        fullPath = os.path.join(storeFolder, name)
        match = re.fullmatch(pattern, name)
        if not match or not os.path.isfile(fullPath):
            continue
        if match['uid'] != studyInstanceUID or match['keyword'] != keyword:
            continue

        fileDateTime = datetime.strptime(match['timestamp'], '%Y-%m-%d_%H-%M-%S')
        age = studyDateTime - fileDateTime
        if not timedelta(0) <= age <= timedelta(hours=maxTimeDeltaHours):
            continue
        if bestAge is None or age < bestAge:
            bestAge = age
            fileMatch = fullPath

    if fileMatch is None:
        return None
    with open(fileMatch, 'rb') as file:
        return pickle.load(file)
```

In the later run, extract the `studyDateTime` and `studyInstanceUID` from **that run's** header, and use the same `storeFolder` and `keyword` as the saving run:

```python
refData = LoadReferenceData(storeFolder, studyDateTime, studyInstanceUID, keyword)
if refData is None:
    logging.warning('No matching reference data found')
else:
    referenceHeader = refData['mrdHeader']
    referenceImages = refData['images']
    referenceData = referenceImages[0].data
```

**Notes:**

- No garbage collection is performed on `storeFolder` in these functions.  Use caution if saving large datasets and ensure that sufficient disk space is available.

## Reference

- **Source:** [LCModel-MRD-App](https://github.com/kspaceKelvin/LCModel-MRD-App) — `process_image()` and `LoadReferenceData()`
- **Save/load call permalink:** [Reference handling](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e814e265f4a0ffcc1d1d5efd334a7f6a8c/lcmodel.py#L133-L160)
- **Retrieval helper permalink:** [Reference matching and loading](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e814e265f4a0ffcc1d1d5efd334a7f6a8c/lcmodel.py#L509-L574)
- **Search anchors:** `pickle.dump`, `LoadReferenceData`