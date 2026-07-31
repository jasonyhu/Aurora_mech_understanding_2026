"""
Download FINN v2.5 MODIS+VIIRS gridded CO fire emissions from NCAR GDEX
for specific fire event date ranges.

Uses the GDEX subset API to request only the dates/regions needed.
Token should be stored in ~/.gdex_token (from https://gdex.ucar.edu/accounts/profile/).
"""

import json
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import consts

FINN_DIR = consts.SCRATCH_PATH / "finn"
FINN_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_FILE = Path.home() / ".gdex_token"
API_BASE = "https://gdex.ucar.edu/api"

# Use continental bounding boxes to cover both regional and continental views
FIRE_EVENTS = [
    {"name": "central_russia_aug2022", "start": "202208190000", "end": "202208250000",
     "nlat": 75, "slat": 35, "wlon": 10, "elon": 80},
    {"name": "siberia_yakutia_jul2022", "start": "202207100000", "end": "202207160000",
     "nlat": 75, "slat": 40, "wlon": 90, "elon": 180},
    {"name": "alaska_jul2022", "start": "202207030000", "end": "202207090000",
     "nlat": 75, "slat": 40, "wlon": -170, "elon": -80},
    {"name": "amazon_sep2022", "start": "202209060000", "end": "202209120000",
     "nlat": 10, "slat": -35, "wlon": -85, "elon": -35},
    {"name": "south_america_jun2022", "start": "202206150000", "end": "202206210000",
     "nlat": -10, "slat": -55, "wlon": -85, "elon": -40},
    {"name": "spain_jul2022", "start": "202207150000", "end": "202207210000",                                 
   "nlat": 50, "slat": 30, "wlon": -15, "elon": 10}, 
]


def get_token() -> str:
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    token = input("Enter your GDEX API token (from https://gdex.ucar.edu/accounts/profile/): ").strip()
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    return token


def submit_request(event: dict, token: str) -> str | None:
    """Submit a GDEX subset request for one fire event. Returns request index."""
    control = {
        "dataset": "d312009",
        "date": f"{event['start']}/to/{event['end']}",
        "param": "fire_modisviirs_CO",
        "oformat": "netCDF",
        "nlat": event["nlat"],
        "slat": event["slat"],
        "wlon": event["wlon"],
        "elon": event["elon"],
        "groupindex": 2,  # MODIS+VIIRS group
    }

    url = f"{API_BASE}/submit/?token={token}"
    resp = requests.post(url, json=control)

    if resp.status_code != 200:
        logger.error(f"Submit failed for {event['name']}: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    if data.get("status") != "ok":
        logger.error(f"Submit error for {event['name']}: {data}")
        return None

    req_idx = data["data"].get("request_id") or data["data"].get("request_index")
    logger.info(f"Submitted request {req_idx} for {event['name']}")
    return str(req_idx)


def wait_for_request(req_idx: str, token: str, poll_interval: int = 30, max_wait: int = 1800) -> bool:
    """Poll until request is complete. Returns True if ready."""
    url = f"{API_BASE}/status/{req_idx}?token={token}"
    elapsed = 0
    while elapsed < max_wait:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            rdata = data.get("data", {})
            status = rdata.get("status", "")
            if "completed" in status.lower() or "done" in status.lower():
                logger.info(f"Request {req_idx} completed.")
                return True
            if "error" in status.lower():
                logger.error(f"Request {req_idx} failed: {status}")
                return False
            logger.info(f"Request {req_idx} status: {status} (waiting...)")
        time.sleep(poll_interval)
        elapsed += poll_interval
    logger.error(f"Request {req_idx} timed out after {max_wait}s")
    return False


def download_request(req_idx: str, token: str, output_dir: Path) -> list[Path]:
    """Download completed request files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file list
    url = f"{API_BASE}/get_req_files/{req_idx}?token={token}"
    resp = requests.get(url)
    if resp.status_code != 200:
        logger.error(f"Failed to get file list for request {req_idx}")
        return []

    data = resp.json()
    file_urls = data.get("data", {}).get("web_files", [])
    if not file_urls:
        # Try alternative response structure
        file_urls = data.get("data", [])

    downloaded = []
    for file_url in file_urls:
        if isinstance(file_url, dict):
            file_url = file_url.get("web_path", file_url.get("url", ""))
        if not file_url:
            continue

        # Add token if not present
        if "token=" not in file_url:
            sep = "&" if "?" in file_url else "?"
            file_url = f"{file_url}{sep}token={token}"

        filename = file_url.split("/")[-1].split("?")[0]
        out_path = output_dir / filename

        if out_path.exists():
            logger.info(f"Already exists: {filename}")
            downloaded.append(out_path)
            continue

        logger.info(f"Downloading {filename}...")
        resp = requests.get(file_url, stream=True)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1048576):
                    f.write(chunk)
            downloaded.append(out_path)
            logger.info(f"  Saved: {out_path}")
        else:
            logger.error(f"  Download failed: {resp.status_code}")

    return downloaded


def main():
    token = get_token()
    logger.info(f"Using GDEX token: {token[:8]}...")

    # Verify API access
    resp = requests.get(f"{API_BASE}/status/?token={token}")
    if resp.status_code != 200:
        logger.error(f"API access failed ({resp.status_code}). Check your token.")
        sys.exit(1)
    logger.info("GDEX API access confirmed.\n")

    # Submit all requests
    requests_map = {}
    for event in FIRE_EVENTS:
        req_idx = submit_request(event, token)
        if req_idx:
            requests_map[event["name"]] = req_idx

    if not requests_map:
        logger.error("No requests submitted successfully.")
        sys.exit(1)

    # Wait for all requests to complete
    logger.info(f"\nWaiting for {len(requests_map)} requests to complete...")
    completed = {}
    for name, req_idx in requests_map.items():
        if wait_for_request(req_idx, token):
            completed[name] = req_idx

    # Download completed requests
    logger.info(f"\nDownloading {len(completed)} completed requests...")
    for name, req_idx in completed.items():
        out_dir = FINN_DIR / name
        files = download_request(req_idx, token, out_dir)
        logger.info(f"  {name}: {len(files)} files downloaded to {out_dir}")

    logger.info(f"\nDone. FINN data saved to {FINN_DIR}")


if __name__ == "__main__":
    main()
