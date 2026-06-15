# Data

This folder stores the short-pulse voltage-response data used in this repository.

The data are organized by battery material-capacity group. Each group folder contains Excel files corresponding to different pulse-width settings.

## Data organization

The expected folder structure is:

```text
data/
├── 10Ah LMO/
├── 15Ah NMC/
├── 21Ah NMC/
├── 24Ah LMO/
├── 25Ah LMO/
├── 26Ah LMO/
├── 35Ah LFP/
└── 68Ah LFP/
```

Each folder corresponds to one battery material-capacity group. For example:

```text
15Ah NMC/
```

contains pulse-response files for the 15 Ah NMC group.

The expected file naming format is:

```text
<Material>_<Capacity>_W_<PulseWidth>.xlsx
```

For example:

```text
NMC_15Ah_W_30.xlsx
NMC_15Ah_W_50.xlsx
NMC_15Ah_W_70.xlsx
NMC_15Ah_W_100.xlsx
NMC_15Ah_W_300.xlsx
NMC_15Ah_W_500.xlsx
NMC_15Ah_W_700.xlsx
NMC_15Ah_W_1000.xlsx
NMC_15Ah_W_3000.xlsx
NMC_15Ah_W_5000.xlsx
```

Here, `W_<PulseWidth>` indicates the pulse width in milliseconds.

## Pulse-width settings

The default pulse-width list is:

```text
30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000 ms
```

Most data folders are expected to contain one Excel file for each pulse-width setting.

## Excel sheet structure

Each Excel file contains multiple sheets for different SOC-related data partitions.

The expected sheets include:

```text
SOC ALL
SOC TEST RANDOM
SOC5
SOC10
SOC15
...
SOC90
```

The sheet meanings are:

* `SOC ALL`: all available records for the corresponding material-capacity group and pulse-width file;
* `SOC TEST RANDOM`: randomly distributed SOC records used for random-SOC testing;
* `SOC5`, `SOC10`, ..., `SOC90`: records grouped by nominal SOC level.

Do not rename the sheet names unless the corresponding data-loading code is also updated.

## Column structure

Each sheet follows the same column format. The expected columns include:

```text
File_Name
Mat
No.
ID
Qn
Q
SOH
Pt
SOC
SOCR
U1
U2
...
U41
```

The main columns used in the current framework are:

| Column      | Description                                             |
| ----------- | ------------------------------------------------------- |
| `File_Name` | Source file name or original record name                |
| `Mat`       | Battery material type, such as `NMC`, `LMO` or `LFP`    |
| `No.`       | Cell or sample index within the material-capacity group |
| `ID`        | Battery or sample identifier                            |
| `Qn`        | Nominal capacity                                        |
| `Q`         | Measured or available capacity                          |
| `SOH`       | State of health label                                   |
| `Pt`        | Pulse test condition recorded in the data table         |
| `SOC`       | State of charge label                                   |
| `U1`–`U41`  | Short-pulse voltage-response features                   |

Although the Excel files contain several metadata columns, the main model inputs and targets are:

```text
Input features:
    U1, U2, ..., U41

Prediction targets:
    material-capacity class
    SOC
    SOH
```

The material-capacity class is constructed from the material type and nominal capacity, for example:

```text
NMC_15Ah
LMO_24Ah
LFP_35Ah
```

## Data availability

If the full dataset is not included directly in this GitHub repository, please download or obtain the dataset separately and place it under this `data/` directory following the structure shown above.

The repository expects the data path to be:

```text
data/
```

relative to the repository root.

## Notes

* Keep the folder names, file names and sheet names unchanged unless the loading scripts are also modified.
* Keep the Excel column names unchanged, especially `SOC`, `SOH` and `U1`–`U41`.
* Large generated files, caches and model outputs should be stored under `results/`, not inside `data/`.
* Python cache folders such as `__pycache__/` should not be committed to the repository.
