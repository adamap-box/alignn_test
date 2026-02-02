"""Match Materials Project data with MAGNDATA (Bilbao Crystallographic Server).

This script attempts to match materials from mp_data with magnetic structures
from MAGNDATA based on formula, space group, and lattice parameters.

MAGNDATA: https://www.cryst.ehu.es/magndata/
"""
import os
import json
import requests
from typing import List, Dict, Optional
import re
import urllib3

# Disable SSL warnings (Bilbao server has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# MAGNDATA base URL
MAGNDATA_URL = "https://www.cryst.ehu.es/magndata"


def download_magndata_index():
    """Download MAGNDATA index (list of all magnetic structures)."""
    # MAGNDATA provides a text file with all entries
    url = "https://www.cryst.ehu.es/magndata/magndata_baselabel.txt"
    
    print("Downloading MAGNDATA index...")
    try:
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        entries = []
        lines = response.text.strip().split('\n')
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                magndata_id = parts[0]  # e.g., "0.10", "1.234"
                formula = parts[1] if len(parts) > 1 else ""
                entries.append({
                    "magndata_id": magndata_id,
                    "formula": formula,
                    "raw_line": line.strip()
                })
        
        print(f"Found {len(entries)} MAGNDATA entries")
        return entries
    except Exception as e:
        print(f"Error downloading MAGNDATA index: {e}")
        return []


def get_magndata_details(magndata_id: str) -> Optional[Dict]:
    """Get detailed information for a MAGNDATA entry."""
    # MAGNDATA provides CIF files and details via their API
    url = f"https://www.cryst.ehu.es/magndata/mcif/{magndata_id}.mcif"
    
    try:
        response = requests.get(url, timeout=30, verify=False)
        if response.status_code == 200:
            mcif_content = response.text
            
            # Parse basic info from mcif
            info = {
                "magndata_id": magndata_id,
                "mcif_content": mcif_content,
            }
            
            # Extract key fields from mcif
            for line in mcif_content.split('\n'):
                if line.startswith('_chemical_formula_sum'):
                    info["formula"] = line.split(maxsplit=1)[1].strip().strip("'\"")
                elif line.startswith('_cell_length_a'):
                    info["a"] = float(line.split()[1])
                elif line.startswith('_cell_length_b'):
                    info["b"] = float(line.split()[1])
                elif line.startswith('_cell_length_c'):
                    info["c"] = float(line.split()[1])
                elif line.startswith('_space_group_magn_name_BNS'):
                    info["magnetic_space_group"] = line.split(maxsplit=1)[1].strip().strip("'\"")
                elif line.startswith('_parent_space_group.name_H-M'):
                    info["parent_space_group"] = line.split(maxsplit=1)[1].strip().strip("'\"")
                elif '_atomic_positions_source_database_code_ICSD' in line:
                    try:
                        info["icsd_code"] = line.split()[-1]
                    except:
                        pass
            
            return info
    except Exception as e:
        print(f"Error fetching MAGNDATA {magndata_id}: {e}")
    
    return None


def normalize_formula(formula: str) -> str:
    """Normalize chemical formula for comparison."""
    # Remove spaces, subscripts, and standardize
    formula = formula.replace(" ", "").replace("_", "")
    # Sort elements alphabetically
    elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    sorted_elements = sorted(elements, key=lambda x: x[0])
    return ''.join([f"{e}{n}" for e, n in sorted_elements])


def match_formula(mp_formula: str, magndata_formula: str, tolerance: float = 0.1) -> bool:
    """Check if two formulas match (allowing for different conventions)."""
    mp_norm = normalize_formula(mp_formula)
    mag_norm = normalize_formula(magndata_formula)
    
    # Exact match
    if mp_norm == mag_norm:
        return True
    
    # Check if elements are the same
    mp_elements = set(re.findall(r'[A-Z][a-z]?', mp_formula))
    mag_elements = set(re.findall(r'[A-Z][a-z]?', magndata_formula))
    
    return mp_elements == mag_elements


def match_lattice(mp_lattice: Dict, magndata_lattice: Dict, tolerance: float = 0.05) -> bool:
    """Check if lattice parameters match within tolerance (5%)."""
    for param in ['a', 'b', 'c']:
        if param in mp_lattice and param in magndata_lattice:
            mp_val = mp_lattice[param]
            mag_val = magndata_lattice[param]
            if abs(mp_val - mag_val) / mp_val > tolerance:
                return False
    return True


def load_mp_data(mp_data_file: str) -> List[Dict]:
    """Load Materials Project data."""
    print(f"Loading Materials Project data from {mp_data_file}...")
    with open(mp_data_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} materials")
    return data


def match_mp_with_magndata(
    mp_data: List[Dict],
    magndata_entries: List[Dict],
    output_file: str = "mp_magndata_matches.json",
    detailed: bool = False
) -> List[Dict]:
    """Match Materials Project materials with MAGNDATA entries."""
    
    matches = []
    
    # Filter MP data to magnetic materials only
    magnetic_mp = [m for m in mp_data if m.get('is_magnetic') or m.get('ordering') not in [None, 'NM']]
    print(f"Filtering to {len(magnetic_mp)} magnetic materials from MP")
    
    # Create formula lookup for MAGNDATA
    magndata_by_formula = {}
    for entry in magndata_entries:
        formula = entry.get('formula', '')
        elements = frozenset(re.findall(r'[A-Z][a-z]?', formula))
        if elements not in magndata_by_formula:
            magndata_by_formula[elements] = []
        magndata_by_formula[elements].append(entry)
    
    print(f"Matching {len(magnetic_mp)} MP materials against {len(magndata_entries)} MAGNDATA entries...")
    
    for i, mp_mat in enumerate(magnetic_mp):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(magnetic_mp)}...")
        
        mp_formula = mp_mat.get('formula', '')
        mp_elements = frozenset(re.findall(r'[A-Z][a-z]?', mp_formula))
        
        # Find MAGNDATA entries with same elements
        potential_matches = magndata_by_formula.get(mp_elements, [])
        
        if potential_matches:
            match_entry = {
                "material_id": mp_mat.get('material_id'),
                "mp_formula": mp_formula,
                "mp_ordering": mp_mat.get('ordering'),
                "mp_magnetization": mp_mat.get('total_magnetization'),
                "magndata_matches": []
            }
            
            for mag_entry in potential_matches:
                mag_match = {
                    "magndata_id": mag_entry.get('magndata_id'),
                    "magndata_formula": mag_entry.get('formula'),
                }
                
                # Get detailed info if requested
                if detailed:
                    details = get_magndata_details(mag_entry.get('magndata_id'))
                    if details:
                        mag_match["magnetic_space_group"] = details.get("magnetic_space_group")
                        mag_match["parent_space_group"] = details.get("parent_space_group")
                        mag_match["icsd_code"] = details.get("icsd_code")
                        
                        # Check lattice match
                        if 'structure' in mp_mat and mp_mat['structure']:
                            mp_lattice = mp_mat['structure'].get('lattice', {})
                            if 'a' in details:
                                mag_match["lattice_match"] = match_lattice(
                                    {"a": mp_lattice.get('a', 0), 
                                     "b": mp_lattice.get('b', 0), 
                                     "c": mp_lattice.get('c', 0)},
                                    {"a": details.get('a', 0),
                                     "b": details.get('b', 0),
                                     "c": details.get('c', 0)}
                                )
                
                match_entry["magndata_matches"].append(mag_match)
            
            matches.append(match_entry)
    
    print(f"Found {len(matches)} MP materials with potential MAGNDATA matches")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=2)
    print(f"Saved matches to {output_file}")
    
    return matches


def search_magndata_for_formula(formula: str) -> List[Dict]:
    """Search MAGNDATA for a specific formula."""
    magndata_entries = download_magndata_index()
    
    target_elements = set(re.findall(r'[A-Z][a-z]?', formula))
    
    matches = []
    for entry in magndata_entries:
        entry_elements = set(re.findall(r'[A-Z][a-z]?', entry.get('formula', '')))
        if entry_elements == target_elements:
            # Get detailed info
            details = get_magndata_details(entry['magndata_id'])
            if details:
                matches.append(details)
            else:
                matches.append(entry)
    
    return matches


def compare_mp_magndata_structure(mp_id: str, magndata_id: str, mp_data_file: str = None):
    """Compare a specific MP material with a MAGNDATA entry."""
    from mp_api.client import MPRester
    
    API_KEY = os.environ.get("MP_API_KEY", "711wCSiRkuGXuBB9MJPb1OZSy1nx7aml")
    
    print(f"\n=== Comparing {mp_id} with MAGNDATA {magndata_id} ===\n")
    
    # Get MP data
    with MPRester(API_KEY) as mpr:
        mp_doc = mpr.materials.summary.search(material_ids=[mp_id])[0]
    
    # Get MAGNDATA data
    magndata = get_magndata_details(magndata_id)
    
    print("--- Materials Project ---")
    print(f"ID: {mp_doc.material_id}")
    print(f"Formula: {mp_doc.formula_pretty}")
    print(f"Ordering: {mp_doc.ordering}")
    print(f"Magnetization: {mp_doc.total_magnetization} μB")
    if mp_doc.structure:
        print(f"Space Group: {mp_doc.structure.get_space_group_info()[0]}")
        print(f"Lattice: a={mp_doc.structure.lattice.a:.3f}, b={mp_doc.structure.lattice.b:.3f}, c={mp_doc.structure.lattice.c:.3f}")
    
    print("\n--- MAGNDATA ---")
    if magndata:
        print(f"ID: {magndata.get('magndata_id')}")
        print(f"Formula: {magndata.get('formula')}")
        print(f"Magnetic Space Group: {magndata.get('magnetic_space_group')}")
        print(f"Parent Space Group: {magndata.get('parent_space_group')}")
        print(f"ICSD Code: {magndata.get('icsd_code')}")
        if 'a' in magndata:
            print(f"Lattice: a={magndata['a']:.3f}, b={magndata.get('b', 0):.3f}, c={magndata.get('c', 0):.3f}")
    else:
        print("Could not fetch MAGNDATA details")
    
    return mp_doc, magndata


if __name__ == "__main__":
    import sys
    
    # Example usage:
    
    # 1. Search MAGNDATA for a specific formula (e.g., DyFeO3)
    print("=== Searching MAGNDATA for DyFeO3 ===")
    dyfe_matches = search_magndata_for_formula("DyFeO3")
    print(f"Found {len(dyfe_matches)} MAGNDATA entries for DyFeO3:")
    for m in dyfe_matches:
        print(f"  {m.get('magndata_id')}: {m.get('formula')} - {m.get('magnetic_space_group', 'N/A')}")
    
    # 2. Compare specific MP material with MAGNDATA entry
    print("\n" + "="*60)
    compare_mp_magndata_structure("mp-22283", "0.10")
    
    # 3. Full matching (uncomment to run - takes time)
    # mp_data = load_mp_data("c:/workspace/mp_data/mp_all_materials.json")
    # magndata_entries = download_magndata_index()
    # matches = match_mp_with_magndata(mp_data, magndata_entries)
