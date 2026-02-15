"""
Script to find additional compounds from MAGNDATA that:
1. Have corresponding info in Materials Project
2. Are NOT already in combined_database.xlsx

Outputs append_database.xlsx with same format as combined_database.xlsx
"""

import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Paths
MP_DATA_PATH = "c:/workspace/mp_data/mp_all_materials.json"
MAGNDATA_DIR = "c:/workspace/magndata_mcif"
COMBINED_DB_PATH = "c:/workspace/alignn_test/combined_database.xlsx"
OUTPUT_PATH = "c:/workspace/alignn_test/append_database.xlsx"
OUTPUT_JSON_PATH = "c:/workspace/alignn_test/append_database.json"


def normalize_formula(formula: str) -> str:
    """Normalize chemical formula for comparison."""
    if not formula or pd.isna(formula):
        return ""
    formula = str(formula).replace(" ", "")
    return formula.lower()


def parse_formula_to_elements(formula: str) -> Dict[str, float]:
    """Parse formula into element:count dictionary."""
    if not formula:
        return {}
    
    formula = str(formula).replace(" ", "")
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    elements = {}
    for element, count in matches:
        if element:
            count = float(count) if count else 1.0
            elements[element] = elements.get(element, 0) + count
    
    return elements


def formulas_match(formula1: str, formula2: str) -> bool:
    """Check if two formulas represent the same compound."""
    norm1 = normalize_formula(formula1)
    norm2 = normalize_formula(formula2)
    
    if norm1 == norm2:
        return True
    
    elem1 = parse_formula_to_elements(formula1)
    elem2 = parse_formula_to_elements(formula2)
    
    if set(elem1.keys()) != set(elem2.keys()):
        return False
    
    if not elem1 or not elem2:
        return False
        
    min1 = min(elem1.values()) if elem1.values() else 1
    min2 = min(elem2.values()) if elem2.values() else 1
    
    norm_elem1 = {k: v/min1 for k, v in elem1.items()}
    norm_elem2 = {k: v/min2 for k, v in elem2.items()}
    
    for k in norm_elem1:
        if abs(norm_elem1[k] - norm_elem2.get(k, 0)) > 0.01:
            return False
    
    return True


def infer_transition_type(magndata_info: Dict) -> str:
    """
    Infer the magnetic transition type from MAGNDATA info.
    
    Returns: 'AFM', 'FM', 'FiM', 'Canted AFM', or 'Unknown'
    """
    # Check magnetic properties details
    props = magndata_info.get('magnetic_properties_details', '').lower()
    msg = magndata_info.get('magnetic_space_group', '')
    has_time_reversal = magndata_info.get('has_time_reversal', False)
    irreps = magndata_info.get('irreps', '')
    
    # Check for explicit mentions in properties
    if 'ferrimagnetic' in props or 'ferrimagnet' in props:
        return 'FiM'
    if 'antiferromagnetic' in props or 'antiferromagnet' in props or 'afm' in props:
        if 'weak ferromagnet' in props or 'canted' in props or 'fm component' in props:
            return 'Canted AFM'
        return 'AFM'
    if 'ferromagnetic' in props or 'ferromagnet' in props:
        return 'FM'
    
    # Infer from magnetic space group
    if msg:
        # Check for primed operations in space group name (indicates AFM or canted)
        primed_count = msg.count("'")
        
        # Check for specific patterns
        # Type I (no primes, same as crystal) - usually FM
        # Type III (some primes) - usually AFM or canted
        # Type IV (with _c, _I, _A, _C, _F, _R) - incommensurate or complex
        
        if '_I' in msg or '_c' in msg or '_C' in msg or '_A' in msg or '_F' in msg:
            # Typically AFM with propagation vector
            return 'AFM'
        
        if primed_count > 0:
            # Has primed operations
            if primed_count >= 2:
                # Multiple primed ops often indicates canted or weak FM
                if 'fm component' in props or 'weak' in props:
                    return 'Canted AFM'
                return 'AFM'
            else:
                # Single prime, often FM with some AFM component
                return 'FM'
        else:
            # No primes - could be FM or type I magnetic structure
            return 'FM'
    
    # Check irreps for hints
    if irreps:
        # GM (Gamma point) irreps often indicate specific orderings
        if 'mGM1' in irreps:
            return 'AFM'  # G-type AFM in perovskites
        if 'mGM4' in irreps:
            return 'Canted AFM'  # Often canted
    
    # Default based on time reversal
    if has_time_reversal:
        return 'AFM'
    
    return 'Unknown'


def parse_mcif_file(filepath: str) -> Optional[Dict]:
    """Parse mcif file and extract key information."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        info = {
            'magndata_id': os.path.basename(filepath).replace('.mcif', '').replace('_', '.'),
            'mcif_file': os.path.basename(filepath)
        }
        
        # Extract formula
        formula_match = re.search(r"_chemical_formula_sum\s+'([^']+)'", content)
        if formula_match:
            info['magndata_formula'] = formula_match.group(1)
        
        # Extract magnetic space group
        msg_match = re.search(r'_space_group_magn\.name_BNS\s+"([^"]+)"', content)
        if msg_match:
            info['magnetic_space_group'] = msg_match.group(1)
        
        # Extract parent space group
        psg_match = re.search(r"_parent_space_group\.name_H-M_alt\s+'([^']+)'", content)
        if psg_match:
            info['parent_space_group'] = psg_match.group(1)
        
        # Extract lattice parameters
        for param in ['a', 'b', 'c']:
            match = re.search(rf'_cell_length_{param}\s+(\d+\.?\d*)', content)
            if match:
                info[f'magndata_{param}'] = float(match.group(1))
        
        for param in ['alpha', 'beta', 'gamma']:
            match = re.search(rf'_cell_angle_{param}\s+(\d+\.?\d*)', content)
            if match:
                info[f'magndata_{param}'] = float(match.group(1))
        
        # Extract transition temperature
        tc_match = re.search(r'_transition_temperature\s+(\S+)', content)
        if tc_match and tc_match.group(1) != '?':
            info['magndata_Tc'] = tc_match.group(1)
        
        # Extract ICSD code
        icsd_match = re.search(r'_atomic_positions_source_database_code_ICSD\s+(\d+)', content)
        if icsd_match:
            info['icsd_code'] = icsd_match.group(1)
        
        # Extract magnetic properties details
        props_match = re.search(r'_exptl_crystal_magnetic_properties_details\s*\n;\n(.*?)\n;', content, re.DOTALL)
        if props_match:
            info['magnetic_properties_details'] = props_match.group(1).strip()
        
        # Extract irreps
        irrep_matches = re.findall(r'^(mGM\S+)\s+\d+\s+\d+', content, re.MULTILINE)
        if irrep_matches:
            info['irreps'] = ', '.join(irrep_matches)
        
        # Check for primed operations (time reversal)
        if re.search(r",-1\s*$", content, re.MULTILINE):
            info['has_time_reversal'] = True
        else:
            info['has_time_reversal'] = False
        
        # Infer transition type
        info['inferred_transition_type'] = infer_transition_type(info)
        
        return info
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def load_mp_data() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load Materials Project data and create index by formula."""
    print("Loading Materials Project data...")
    with open(MP_DATA_PATH, 'r') as f:
        mp_data = json.load(f)
    
    mp_by_formula = defaultdict(list)
    for mat in mp_data:
        formula = mat.get('formula', '')
        norm_formula = normalize_formula(formula)
        mp_by_formula[norm_formula].append(mat)
        
        elements = parse_formula_to_elements(formula)
        elem_key = "_".join(sorted(elements.keys())).lower()
        mp_by_formula[f"elem_{elem_key}"].append(mat)
    
    print(f"Loaded {len(mp_data)} materials from MP")
    return mp_data, mp_by_formula


def load_existing_formulas() -> set:
    """Load formulas already in combined_database.xlsx."""
    print("Loading existing combined database...")
    df = pd.read_excel(COMBINED_DB_PATH)
    existing = set()
    for formula in df['formula']:
        if pd.notna(formula):
            existing.add(normalize_formula(formula))
    print(f"Found {len(existing)} existing formulas")
    return existing


def find_mp_matches(formula: str, mp_by_formula: Dict) -> List[Dict]:
    """Find Materials Project matches for a formula."""
    matches = []
    norm_formula = normalize_formula(formula)
    
    if norm_formula in mp_by_formula:
        matches.extend(mp_by_formula[norm_formula])
    
    elements = parse_formula_to_elements(formula)
    elem_key = f"elem_{'_'.join(sorted(elements.keys())).lower()}"
    if elem_key in mp_by_formula:
        for mat in mp_by_formula[elem_key]:
            if formulas_match(formula, mat.get('formula', '')):
                if mat not in matches:
                    matches.append(mat)
    
    return matches


def scan_and_append():
    """Main function to scan MAGNDATA and create append database."""
    # Load existing data
    existing_formulas = load_existing_formulas()
    mp_data, mp_by_formula = load_mp_data()
    
    # Load all MAGNDATA files
    print("\nScanning MAGNDATA files...")
    mcif_files = list(Path(MAGNDATA_DIR).glob("*.mcif"))
    print(f"Found {len(mcif_files)} mcif files")
    
    # Track unique formulas to avoid duplicates
    processed_formulas = set()
    append_data = []
    
    for mcif_file in mcif_files:
        magndata_info = parse_mcif_file(str(mcif_file))
        if not magndata_info:
            continue
        
        formula = magndata_info.get('magndata_formula', '')
        if not formula:
            continue
        
        norm_formula = normalize_formula(formula)
        
        # Skip if already in combined database
        if norm_formula in existing_formulas:
            continue
        
        # Skip if already processed (avoid duplicates from same formula)
        if norm_formula in processed_formulas:
            # But still add this magndata entry to existing record
            for entry in append_data:
                if normalize_formula(entry.get('formula', '')) == norm_formula:
                    # Append to all_magndata_ids
                    existing_ids = entry.get('all_magndata_ids', '')
                    new_id = magndata_info.get('magndata_id', '')
                    if new_id and new_id not in existing_ids:
                        entry['all_magndata_ids'] = f"{existing_ids}, {new_id}" if existing_ids else new_id
                        entry['magndata_match_count'] = entry.get('magndata_match_count', 0) + 1
                    break
            continue
        
        # Find MP matches
        mp_matches = find_mp_matches(formula, mp_by_formula)
        
        # Only include if found in MP
        if not mp_matches:
            continue
        
        # Mark as processed
        processed_formulas.add(norm_formula)
        
        # Create entry with same format as combined_database
        entry = {
            # Excel-like fields (from MAGNDATA)
            'formula': formula.replace(' ', ''),
            'transition_type': magndata_info.get('inferred_transition_type', 'Unknown'),
            'Tc_exp': magndata_info.get('magndata_Tc'),
            'lowest_T_PM': None,
            'specific_heat_PM': None,
            'space_group_exp': magndata_info.get('parent_space_group', '').replace(' ', ''),
            'notes_1': f"Added from MAGNDATA {magndata_info.get('magndata_id', '')}",
            'notes_2': magndata_info.get('magnetic_properties_details'),
            'mp_predicted_ground_state': None,
        }
        
        # Add MP data
        entry['mp_match_count'] = len(mp_matches)
        
        best_mp = min(mp_matches, key=lambda x: x.get('energy_above_hull') if x.get('energy_above_hull') is not None else float('inf'))
        entry['mp_id'] = best_mp.get('material_id')
        entry['mp_formula'] = best_mp.get('formula')
        entry['mp_energy_per_atom'] = best_mp.get('energy_per_atom')
        entry['mp_formation_energy'] = best_mp.get('formation_energy_per_atom')
        entry['mp_energy_above_hull'] = best_mp.get('energy_above_hull')
        entry['mp_is_stable'] = best_mp.get('is_stable')
        entry['mp_band_gap'] = best_mp.get('band_gap')
        entry['mp_is_magnetic'] = best_mp.get('is_magnetic')
        entry['mp_ordering'] = best_mp.get('ordering')
        entry['mp_total_magnetization'] = best_mp.get('total_magnetization')
        entry['mp_volume'] = best_mp.get('volume')
        entry['mp_density'] = best_mp.get('density')
        entry['mp_nsites'] = best_mp.get('nsites')
        
        symmetry = best_mp.get('symmetry', {})
        if symmetry:
            entry['mp_space_group_symbol'] = symmetry.get('symbol')
            entry['mp_space_group_number'] = symmetry.get('number')
            entry['mp_crystal_system'] = symmetry.get('crystal_system')
        
        entry['all_mp_ids'] = ', '.join([m.get('material_id', '') for m in mp_matches[:10]])
        
        # Add MAGNDATA info
        entry['magndata_match_count'] = 1
        entry['magndata_id'] = magndata_info.get('magndata_id')
        entry['magndata_formula'] = magndata_info.get('magndata_formula')
        entry['magndata_magnetic_space_group'] = magndata_info.get('magnetic_space_group')
        entry['magndata_parent_space_group'] = magndata_info.get('parent_space_group')
        entry['magndata_Tc'] = magndata_info.get('magndata_Tc')
        entry['magndata_icsd'] = magndata_info.get('icsd_code')
        entry['magndata_irreps'] = magndata_info.get('irreps')
        entry['magndata_has_time_reversal'] = magndata_info.get('has_time_reversal')
        entry['magndata_properties'] = magndata_info.get('magnetic_properties_details')
        
        for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            entry[f'magndata_{param}'] = magndata_info.get(f'magndata_{param}')
        
        entry['all_magndata_ids'] = magndata_info.get('magndata_id', '')
        
        append_data.append(entry)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Total MAGNDATA entries scanned: {len(mcif_files)}")
    print(f"Entries already in combined_database: {len(existing_formulas)}")
    print(f"New entries with MP match: {len(append_data)}")
    
    # Count by transition type
    type_counts = defaultdict(int)
    for entry in append_data:
        type_counts[entry.get('transition_type', 'Unknown')] += 1
    
    print("\nBy transition type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    if not append_data:
        print("\nNo new entries to append.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(append_data)
    
    # Reorder columns to match combined_database format
    priority_cols = [
        'formula', 'transition_type', 'Tc_exp',
        'mp_match_count', 'mp_id', 'mp_ordering', 'mp_is_magnetic', 'mp_total_magnetization',
        'magndata_match_count', 'magndata_id', 'magndata_magnetic_space_group'
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]
    
    # Save to Excel
    print(f"\nSaving to Excel: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index=False)
    
    # Save to JSON
    print(f"Saving to JSON: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(append_data, f, indent=2, default=str)
    
    print("\nDone!")
    return df


if __name__ == "__main__":
    df = scan_and_append()
    
    if df is not None and len(df) > 0:
        print("\n=== Sample new entries ===")
        sample = df.head(20)
        print(sample[['formula', 'transition_type', 'mp_id', 'mp_ordering', 
                      'magndata_id', 'magndata_magnetic_space_group']].to_string())
