"""Download mcif files from MAGNDATA (Bilbao Crystallographic Server).

MAGNDATA: https://www.cryst.ehu.es/magndata/

This script downloads magnetic CIF (mcif) files which contain:
- Crystal structure
- Magnetic space group
- Magnetic moments for each atom
- Propagation vector
"""
import os
import json
import requests
from typing import List, Dict, Optional
import re
import urllib3
from tqdm import tqdm

# Disable SSL warnings (Bilbao server has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# MAGNDATA base URLs
MAGNDATA_BASE = "https://www.cryst.ehu.es/magndata"
MCIF_URL = "https://www.cryst.ehu.es/magndata/mcif"


def download_mcif(magndata_id: str, output_dir: str = "magndata_mcif", save: bool = True) -> Optional[str]:
    """Download a single mcif file from MAGNDATA.
    
    Args:
        magndata_id: MAGNDATA ID (e.g., "0.10", "1.234")
        output_dir: Directory to save mcif files
        save: Whether to save to file
        
    Returns:
        mcif content as string, or None if failed
    """
    url = f"{MCIF_URL}/{magndata_id}.mcif"
    
    try:
        response = requests.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            mcif_content = response.text
            
            if save:
                os.makedirs(output_dir, exist_ok=True)
                # Replace dots with underscores for filename
                filename = f"{magndata_id.replace('.', '_')}.mcif"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(mcif_content)
                print(f"Saved: {filepath}")
            
            return mcif_content
        else:
            print(f"Failed to download {magndata_id}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {magndata_id}: {e}")
        return None


def parse_mcif(mcif_content: str) -> Dict:
    """Parse mcif file content and extract key information."""
    info = {}
    
    lines = mcif_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Basic crystal info
        if line.startswith('_chemical_formula_sum'):
            info["formula"] = line.split(maxsplit=1)[1].strip().strip("'\"") if len(line.split(maxsplit=1)) > 1 else ""
        elif line.startswith('_cell_length_a'):
            try:
                info["a"] = float(line.split()[1])
            except:
                pass
        elif line.startswith('_cell_length_b'):
            try:
                info["b"] = float(line.split()[1])
            except:
                pass
        elif line.startswith('_cell_length_c'):
            try:
                info["c"] = float(line.split()[1])
            except:
                pass
        elif line.startswith('_cell_angle_alpha'):
            try:
                info["alpha"] = float(line.split()[1])
            except:
                pass
        elif line.startswith('_cell_angle_beta'):
            try:
                info["beta"] = float(line.split()[1])
            except:
                pass
        elif line.startswith('_cell_angle_gamma'):
            try:
                info["gamma"] = float(line.split()[1])
            except:
                pass
        
        # Magnetic space group
        elif line.startswith('_space_group_magn_name_BNS'):
            info["magnetic_space_group_BNS"] = line.split(maxsplit=1)[1].strip().strip("'\"") if len(line.split(maxsplit=1)) > 1 else ""
        elif line.startswith('_space_group_magn.name_BNS'):
            info["magnetic_space_group_BNS"] = line.split(maxsplit=1)[1].strip().strip("'\"") if len(line.split(maxsplit=1)) > 1 else ""
        
        # Parent space group
        elif line.startswith('_parent_space_group.name_H-M'):
            info["parent_space_group"] = line.split(maxsplit=1)[1].strip().strip("'\"") if len(line.split(maxsplit=1)) > 1 else ""
        
        # ICSD reference
        elif '_atomic_positions_source_database_code_ICSD' in line:
            try:
                info["icsd_code"] = line.split()[-1]
            except:
                pass
        
        # Propagation vector
        elif line.startswith('_Sr') or 'Sr' in line and 'propagation' in line.lower():
            # Complex parsing needed for propagation vector
            pass
        
        # Magnetic ordering type
        elif '_Sr_Sr_Sr_Sr_Sr' in line or 'antiferromagn' in line.lower():
            info["ordering_hint"] = "AFM"
        elif 'ferromagn' in line.lower() and 'antiferro' not in line.lower():
            info["ordering_hint"] = "FM"
    
    return info


def get_magndata_list() -> List[Dict]:
    """Get list of all MAGNDATA entries by scraping the index page."""
    
    # Try multiple methods to get the list
    entries = []
    
    # Method 1: Try the index page
    try:
        url = f"{MAGNDATA_BASE}/magndata.magneticcompounds"
        response = requests.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            # Parse HTML to extract IDs
            content = response.text
            # Find patterns like "0.10" or "1.234" in href attributes
            ids = re.findall(r'href="mcif/(\d+\.\d+)\.mcif"', content)
            for id in ids:
                entries.append({"magndata_id": id})
            print(f"Found {len(entries)} entries from index page")
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try known ID ranges if method 1 fails
    if not entries:
        print("Trying to enumerate known MAGNDATA ID ranges...")
        # MAGNDATA uses format X.Y where X is category, Y is number
        # Categories: 0.x (commensurate), 1.x (incommensurate), etc.
        
        # Generate potential IDs
        for category in range(4):  # 0, 1, 2, 3
            for num in range(1, 500):  # Up to 500 per category
                magndata_id = f"{category}.{num}"
                entries.append({"magndata_id": magndata_id})
        
        print(f"Generated {len(entries)} potential IDs to try")
    
    return entries


def download_all_mcif(
    output_dir: str = "magndata_mcif",
    max_entries: int = None,
    start_from: str = None,
    categories: List[int] = None
) -> List[str]:
    """Download all mcif files from MAGNDATA.
    
    Args:
        output_dir: Directory to save mcif files
        max_entries: Maximum number of entries to download (None for all)
        start_from: Start from this ID (e.g., "0.50")
        categories: List of categories to download (e.g., [0, 1] for commensurate and incommensurate)
        
    Returns:
        List of successfully downloaded IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = []
    failed = []
    
    # Get list of entries
    entries = get_magndata_list()
    
    # Filter by category if specified
    if categories:
        entries = [e for e in entries if int(e['magndata_id'].split('.')[0]) in categories]
    
    # Start from specific ID if specified
    if start_from:
        start_idx = 0
        for i, e in enumerate(entries):
            if e['magndata_id'] == start_from:
                start_idx = i
                break
        entries = entries[start_idx:]
    
    # Limit entries
    if max_entries:
        entries = entries[:max_entries]
    
    print(f"Downloading {len(entries)} mcif files to {output_dir}...")
    
    for entry in tqdm(entries, desc="Downloading"):
        magndata_id = entry['magndata_id']
        
        # Check if already downloaded
        filename = f"{magndata_id.replace('.', '_')}.mcif"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            downloaded.append(magndata_id)
            continue
        
        # Download
        content = download_mcif(magndata_id, output_dir, save=True)
        if content:
            downloaded.append(magndata_id)
        else:
            failed.append(magndata_id)
    
    print(f"\nDownloaded: {len(downloaded)}")
    print(f"Failed: {len(failed)}")
    
    # Save metadata
    metadata = {
        "total_downloaded": len(downloaded),
        "total_failed": len(failed),
        "downloaded_ids": downloaded,
        "failed_ids": failed[:100],  # First 100 failures
    }
    with open(os.path.join(output_dir, "download_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return downloaded


def download_by_formula(
    formula: str,
    output_dir: str = "magndata_mcif"
) -> List[Dict]:
    """Download all mcif files matching a chemical formula.
    
    Args:
        formula: Chemical formula (e.g., "DyFeO3", "Fe2O3")
        output_dir: Directory to save mcif files
        
    Returns:
        List of downloaded entries with parsed info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    target_elements = set(re.findall(r'[A-Z][a-z]?', formula))
    print(f"Searching for formula with elements: {target_elements}")
    
    downloaded = []
    
    # Try a range of IDs
    for category in range(4):
        for num in range(1, 300):
            magndata_id = f"{category}.{num}"
            
            # Download and check formula
            content = download_mcif(magndata_id, output_dir=None, save=False)
            if content:
                info = parse_mcif(content)
                formula_in_file = info.get('formula', '')
                file_elements = set(re.findall(r'[A-Z][a-z]?', formula_in_file))
                
                if file_elements == target_elements:
                    # Save this file
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{magndata_id.replace('.', '_')}.mcif"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    info['magndata_id'] = magndata_id
                    info['filepath'] = filepath
                    downloaded.append(info)
                    print(f"Found: {magndata_id} - {formula_in_file}")
    
    print(f"\nTotal found for {formula}: {len(downloaded)}")
    return downloaded


def download_specific_ids(
    ids: List[str],
    output_dir: str = "magndata_mcif"
) -> List[Dict]:
    """Download specific mcif files by ID.
    
    Args:
        ids: List of MAGNDATA IDs (e.g., ["0.10", "0.11", "1.5"])
        output_dir: Directory to save mcif files
        
    Returns:
        List of downloaded entries with parsed info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = []
    
    for magndata_id in ids:
        print(f"Downloading {magndata_id}...")
        content = download_mcif(magndata_id, output_dir, save=True)
        if content:
            info = parse_mcif(content)
            info['magndata_id'] = magndata_id
            downloaded.append(info)
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(downloaded, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    return downloaded


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download mcif files from MAGNDATA")
    parser.add_argument("--id", type=str, help="Download specific ID (e.g., 0.10)")
    parser.add_argument("--ids", type=str, nargs='+', help="Download multiple IDs")
    parser.add_argument("--formula", type=str, help="Download by formula (e.g., DyFeO3)")
    parser.add_argument("--all", action="store_true", help="Download all mcif files")
    parser.add_argument("--output", type=str, default="magndata_mcif", help="Output directory")
    parser.add_argument("--max", type=int, default=None, help="Maximum entries to download")
    
    args = parser.parse_args()
    
    if args.id:
        # Download single ID
        content = download_mcif(args.id, args.output)
        if content:
            info = parse_mcif(content)
            print(f"\nParsed info for {args.id}:")
            for k, v in info.items():
                if k != 'mcif_content':
                    print(f"  {k}: {v}")
    
    elif args.ids:
        # Download multiple IDs
        download_specific_ids(args.ids, args.output)
    
    elif args.formula:
        # Download by formula
        download_by_formula(args.formula, args.output)
    
    elif args.all:
        # Download all
        download_all_mcif(args.output, max_entries=args.max)
    
    else:
        # Default: download DyFeO3 examples
        print("Downloading DyFeO3 examples (0.10 and 0.11)...")
        results = download_specific_ids(["0.10", "0.11"], args.output)
        
        print("\n=== Downloaded Files ===")
        for r in results:
            print(f"\n{r.get('magndata_id')}:")
            print(f"  Formula: {r.get('formula')}")
            print(f"  Magnetic Space Group: {r.get('magnetic_space_group_BNS')}")
            print(f"  Parent Space Group: {r.get('parent_space_group')}")
            print(f"  ICSD: {r.get('icsd_code')}")
            if 'a' in r:
                print(f"  Lattice: a={r['a']:.3f}, b={r.get('b', 0):.3f}, c={r.get('c', 0):.3f}")
