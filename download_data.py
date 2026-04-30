"""
Downloads species observation images from iNaturalist API.

Fetches images for a set of coastal/marine species relevant to
Mediterranean and Atlantic biodiversity monitoring.

Usage:
    python download_data.py              # default species, 100 imgs each
    python download_data.py --per-species 200 --quality research
"""

import argparse
import os
import time
import urllib.request
from pathlib import Path

import requests

# Coastal/marine species — chosen for ecological relevance to Mediterranean
# and Atlantic monitoring contexts (aligns with the SDM and EDA projects)
DEFAULT_SPECIES = {
    "Posidonia_oceanica": 118943,      # Mediterranean seagrass — key habitat engineer
    "Cymodocea_nodosa": 130222,        # Coastal seagrass
    "Caulerpa_cylindracea": 128823,    # Invasive green alga — monitoring priority
    "Paracentrotus_lividus": 48032,    # Sea urchin — benthic biodiversity indicator
    "Octopus_vulgaris": 49315,         # Common octopus
}

INAT_API = "https://api.inaturalist.org/v1/observations"


def fetch_observations(taxon_id: int, per_page: int = 100, quality: str = "research") -> list[dict]:
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality,
        "photos": "true",
        "per_page": per_page,
        "order_by": "votes",
    }
    response = requests.get(INAT_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def download_image(url: str, dest_path: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  Failed {dest_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download iNaturalist images for training")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--per-species", type=int, default=100)
    parser.add_argument("--quality", default="research", choices=["research", "needs_id", "casual"])
    args = parser.parse_args()

    base_dir = Path(args.output)

    for species_name, taxon_id in DEFAULT_SPECIES.items():
        species_dir = base_dir / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(species_dir.glob("*.jpg")))
        if existing >= args.per_species:
            print(f"{species_name}: {existing} images already present, skipping")
            continue

        print(f"Fetching {species_name} (taxon {taxon_id})...")
        observations = fetch_observations(taxon_id, per_page=args.per_species, quality=args.quality)

        downloaded = 0
        for obs in observations:
            photos = obs.get("photos", [])
            if not photos:
                continue
            url = photos[0].get("url", "").replace("square", "medium")
            if not url:
                continue
            dest = species_dir / f"{obs['id']}.jpg"
            if dest.exists():
                downloaded += 1
                continue
            if download_image(url, dest):
                downloaded += 1
            time.sleep(0.1)  # polite rate limiting

        print(f"  {downloaded} images saved to {species_dir}")

    print("\nDownload complete. Run: python main.py --feature-extract")


if __name__ == "__main__":
    main()
