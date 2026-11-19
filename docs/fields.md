# Observed sample fields

Generated from local files. Fields can change with product version.

## PACE_OCI.20240615T182549.L2.OC_BGC.V1_0_0.NRT.nc

- title: OCI Level-2 Data
- processing_version: V1.0.0
- time_coverage_start: 2024-06-15T18:25:49.138Z
- time_coverage_end: 2024-06-15T18:30:49.021Z

| Field | Shape | Storage | Units | Meaning |
| --- | --- | --- | --- | --- |
| `sensor_band_parameters/wavelength` | (286,) | int32 | nm | wavelengths |
| `sensor_band_parameters/vcal_gain` | (286,) | float32 | — | Vicarious Calibration Gain |
| `sensor_band_parameters/vcal_offset` | (286,) | float32 | mW cm^-2 um^-1 sr^-1 | Vicarious Calibration Offset |
| `sensor_band_parameters/F0` | (286,) | float32 | W m^-2 um^-1 | Mean Solar Flux |
| `sensor_band_parameters/aw` | (286,) | float32 | m^-1 | Band-pass averaged absorption coefficient for seawater |
| `sensor_band_parameters/bbw` | (286,) | float32 | m^-1 | Band-pass averaged backscattering coefficient for seawater |
| `sensor_band_parameters/k_oz` | (286,) | float32 | cm^-1 | Ozone Absorption cross-sections |
| `sensor_band_parameters/k_no2` | (286,) | float32 | cm^2 molecule^-1 | NO2 Absorption cross-sections |
| `sensor_band_parameters/Tau_r` | (286,) | float32 | — | Rayleigh Optical Thickness |
| `scan_line_attributes/year` | (1710,) | int32 | years | Scan year |
| `scan_line_attributes/day` | (1710,) | int32 | days | Scan day of year |
| `scan_line_attributes/msec` | (1710,) | int32 | milliseconds | Scan time, milliseconds of day |
| `scan_line_attributes/detnum` | (1710,) | int8 | — | Detector Number (zero-based) |
| `scan_line_attributes/mside` | (1710,) | int8 | — | Mirror Side (zero-based) |
| `scan_line_attributes/slon` | (1710,) | float32 | degrees_east | Starting Longitude |
| `scan_line_attributes/clon` | (1710,) | float32 | degrees_east | Center Longitude |
| `scan_line_attributes/elon` | (1710,) | float32 | degrees_east | Ending Longitude |
| `scan_line_attributes/slat` | (1710,) | float32 | degrees_north | Starting Latitude |
| `scan_line_attributes/clat` | (1710,) | float32 | degrees_north | Center Latitude |
| `scan_line_attributes/elat` | (1710,) | float32 | degrees_north | Ending Latitude |
| `scan_line_attributes/csol_z` | (1710,) | float32 | degree | Center Solar Zenith Angle |
| `geophysical_data/chlor_a` | (1710, 1272) | float32 | mg m^-3 | Chlorophyll Concentration, OCI Algorithm |
| `geophysical_data/carbon_phyto` | (1710, 1272) | float32 | mg m^-3 | Phytoplankton Carbon |
| `geophysical_data/poc` | (1710, 1272) | int16 | mg m^-3 | Particulate Organic Carbon, D. Stramski, 2022 (hybrid version) |
| `geophysical_data/chlor_a_unc` | (1710, 1272) | float32 | mg m^-3 | Uncertainty in chlorophyll a concentration |
| `geophysical_data/carbon_phyto_unc` | (1710, 1272) | float32 | mg m^-3 | Phytoplankton Carbon standard uncertainty |
| `geophysical_data/l2_flags` | (1710, 1272) | int32 | — | Level-2 Processing Flags |
| `navigation_data/longitude` | (1710, 1272) | float32 | degrees_east | Longitude |
| `navigation_data/latitude` | (1710, 1272) | float32 | degrees_north | Latitude |
| `navigation_data/cntl_pt_cols` | (1272,) | int32 | — | Number of Pixel Control Points |
| `navigation_data/cntl_pt_rows` | (1710,) | int32 | — | Number of Scan Control Points |
| `navigation_data/tilt` | (1710,) | float32 | degree | Sensor tilt angle |

## PACE_OCI.20240730T181157.L2.OC_AOP.V2_0.NRT.nc

- title: OCI Level-2 Data AOP
- processing_version: 2.0
- time_coverage_start: 2024-07-30T18:11:57.134Z
- time_coverage_end: 2024-07-30T18:16:57.016Z

| Field | Shape | Storage | Units | Meaning |
| --- | --- | --- | --- | --- |
| `sensor_band_parameters/wavelength` | (286,) | int32 | nm | wavelengths |
| `sensor_band_parameters/wavelength_3d` | (184,) | int32 | nm | wavelengths |
| `sensor_band_parameters/vcal_gain` | (286,) | float32 | — | Vicarious Calibration Gain |
| `sensor_band_parameters/vcal_offset` | (286,) | float32 | mW cm^-2 um^-1 sr^-1 | Vicarious Calibration Offset |
| `sensor_band_parameters/F0` | (286,) | float32 | W m^-2 um^-1 | Mean Solar Flux |
| `sensor_band_parameters/aw` | (286,) | float32 | m^-1 | Band-pass averaged absorption coefficient for seawater |
| `sensor_band_parameters/bbw` | (286,) | float32 | m^-1 | Band-pass averaged backscattering coefficient for seawater |
| `sensor_band_parameters/k_oz` | (286,) | float32 | cm^-1 | Ozone Absorption cross-sections |
| `sensor_band_parameters/k_no2` | (286,) | float32 | cm^2 molecule^-1 | NO2 Absorption cross-sections |
| `sensor_band_parameters/Tau_r` | (286,) | float32 | — | Rayleigh Optical Thickness |
| `scan_line_attributes/year` | (1710,) | int32 | years | Scan year |
| `scan_line_attributes/day` | (1710,) | int32 | days | Scan day of year |
| `scan_line_attributes/msec` | (1710,) | int32 | milliseconds | Scan time, milliseconds of day |
| `scan_line_attributes/detnum` | (1710,) | int8 | — | Detector Number (zero-based) |
| `scan_line_attributes/mside` | (1710,) | int8 | — | Mirror Side (zero-based) |
| `scan_line_attributes/slon` | (1710,) | float32 | degrees_east | Starting Longitude |
| `scan_line_attributes/clon` | (1710,) | float32 | degrees_east | Center Longitude |
| `scan_line_attributes/elon` | (1710,) | float32 | degrees_east | Ending Longitude |
| `scan_line_attributes/slat` | (1710,) | float32 | degrees_north | Starting Latitude |
| `scan_line_attributes/clat` | (1710,) | float32 | degrees_north | Center Latitude |
| `scan_line_attributes/elat` | (1710,) | float32 | degrees_north | Ending Latitude |
| `scan_line_attributes/csol_z` | (1710,) | float32 | degree | Center Solar Zenith Angle |
| `geophysical_data/Rrs` | (1710, 1272, 184) | int16 | sr^-1 | Remote sensing reflectance |
| `geophysical_data/Rrs_unc` | (1710, 1272, 184) | int16 | sr^-1 | Uncertainty in remote sensing reflectance |
| `geophysical_data/aot_865` | (1710, 1272) | int16 | — | Aerosol optical thickness at 865 nm |
| `geophysical_data/angstrom` | (1710, 1272) | int16 | — | Aerosol Angstrom exponent, 443 to 865 nm |
| `geophysical_data/avw` | (1710, 1272) | float32 | nm | Apparent Visible Wavelength |
| `geophysical_data/l2_flags` | (1710, 1272) | int32 | — | Level-2 Processing Flags |
| `navigation_data/longitude` | (1710, 1272) | float32 | degrees_east | Longitude |
| `navigation_data/latitude` | (1710, 1272) | float32 | degrees_north | Latitude |
| `navigation_data/tilt` | (1710,) | float32 | degree | Sensor tilt angle |
