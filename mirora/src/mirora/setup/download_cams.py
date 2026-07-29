# Download and save CAMS data to a specified directory. 
# Requires a valid CDS API key: https://cds.climate.copernicus.eu/how-to-api

import logging
from datetime import datetime, timedelta
from zipfile import ZipFile
from pathlib import Path

import cdsapi
import xarray as xr

logger = logging.getLogger(__name__)

from mirora.utils import get_cams_forecast_path

def verify_one_download(cams_save_dir: Path, dt: datetime) -> bool:
    """
    Verifies that the CAMS download for time `dt` exists in the directory `cams_save_dir`.
    """
    cams_forecast_dir = get_cams_forecast_path(cams_save_dir, dt)
    atmos_vars_path = cams_forecast_dir / "data_plev.nc"
    surf_vars_path = cams_forecast_dir / "data_sfc.nc"
    try:
        xr.open_dataset(atmos_vars_path, engine="netcdf4", decode_timedelta=True)
        xr.open_dataset(surf_vars_path, engine="netcdf4", decode_timedelta=True)
    except Exception:
        logger.warning(f"CAMS data missing at time {dt}")
        return False
    return True

def verify_all_downloads(cams_save_dir: Path, start: datetime, end: datetime) -> bool:
    cur_time = start
    all_good = True
    while cur_time <= end:
        if not verify_one_download(cams_save_dir, cur_time):
            all_good = False
        cur_time += timedelta(hours=12)
    return all_good

def download_cams(cams_save_dir: Path, dt: datetime) -> None:
    dataset = "cams-global-atmospheric-composition-forecasts"
    client = cdsapi.Client()

    request = {
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "mean_sea_level_pressure",
                "particulate_matter_1um",
                "particulate_matter_2.5um",
                "particulate_matter_10um",
                "total_column_carbon_monoxide",
                "total_column_nitrogen_dioxide",
                "total_column_nitrogen_monoxide",
                "total_column_ozone",
                "total_column_sulphur_dioxide",
                "carbon_monoxide",
                "geopotential",
                "nitrogen_dioxide",
                "nitrogen_monoxide",
                "ozone",
                "specific_humidity",
                "sulphur_dioxide",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind"
            ],
            "pressure_level": [
                "50", "100", "150",
                "200", "250", "300",
                "400", "500", "600",
                "700", "850", "925",
                "1000"
            ],
            "leadtime_hour": [
                "0",
                "12",
                "24",
                "36",
                "48",
                "60",
                "72",
                "84",
                "96",
                "108",
                "120"
            ],
            "type": ["forecast"],
            "data_format": "netcdf_zip"
        }
    
    request["date"] = ["/".join([dt.strftime(r"%Y-%m-%d")] * 2)]
    request["time"] = [dt.strftime(r"%H:%M")]

    forecast_dir = get_cams_forecast_path(cams_save_dir, dt)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    zip_path = forecast_dir.with_suffix(".zip")
    client.retrieve(dataset, request, zip_path)
    with ZipFile(zip_path, 'r') as forecast_zip:
        forecast_zip.extractall(path=forecast_dir)
