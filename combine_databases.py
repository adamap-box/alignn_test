"""
Script to combine compound data from:
1. Excel file (U database Tc.xlsx) - experimental data
2. Materials Project (mp_all_materials.json) - computational data
3. MAGNDATA mcif files - magnetic structure data

For each compound in Excel, finds matches in MP and MAGNDATA,
then writes combined info to a new file.
"""

import json
import os
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Paths
EXCEL_PATH = "c:/workspace/alignn_test/excel/U database Tc.xlsx"
MP_DATA_PATH = "c:/workspace/mp_data/mp_all_materials.json"
MAGNDATA_DIR = "c:/workspace/magndata_mcif"
OUTPUT_PATH = "c:/workspace/alignn_test/combined_database.xlsx"
OUTPUT_JSON_PATH = "c:/workspace/alignn_test/combined_database.json"


def normalize_formula(formula: str) -> str:
    """Normalize chemical formula for comparison."""
    if not formula or pd.isna(formula):
        return ""
    # Remove spaces
    formula = str(formula).replace(" ", "")
    # Convert to lowercase for comparison
    return formula.lower()


def parse_formula_to_elements(formula: str) -> Dict[str, float]:
    """Parse formula into element:count dictionary."""
    if not formula:
        return {}
    
    formula = str(formula).replace(" ", "")
    # Pattern to match element and optional count
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
    
    # Compare by elements
    elem1 = parse_formula_to_elements(formula1)
    elem2 = parse_formula_to_elements(formula2)
    
    if set(elem1.keys()) != set(elem2.keys()):
        return False
    
    # Check if ratios match (allow for reduced formulas)
    if not elem1 or not elem2:
        return False
        
    # Normalize by smallest count
    min1 = min(elem1.values()) if elem1.values() else 1
    min2 = min(elem2.values()) if elem2.values() else 1
    
    norm_elem1 = {k: v/min1 for k, v in elem1.items()}
    norm_elem2 = {k: v/min2 for k, v in elem2.items()}
    
    for k in norm_elem1:
        if abs(norm_elem1[k] - norm_elem2.get(k, 0)) > 0.01:
            return False
    
    return True


def load_mp_data() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load Materials Project data and create index by formula."""
    print("Loading Materials Project data...")
    with open(MP_DATA_PATH, 'r') as f:
        mp_data = json.load(f)
    
    # Create index by normalized formula
    mp_by_formula = defaultdict(list)
    for mat in mp_data:
        formula = mat.get('formula', '')
        norm_formula = normalize_formula(formula)
        mp_by_formula[norm_formula].append(mat)
        
        # Also index by elements for flexible matching
        elements = parse_formula_to_elements(formula)
        elem_key = "_".join(sorted(elements.keys())).lower()
        mp_by_formula[f"elem_{elem_key}"].append(mat)
    
    print(f"Loaded {len(mp_data)} materials from MP")
    return mp_data, mp_by_formula


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
        for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            match = re.search(rf'_cell_length_{param}\s+(\d+\.?\d*)|_cell_angle_{param}\s+(\d+\.?\d*)', content)
            if match:
                info[f'magndata_{param}'] = float(match.group(1) or match.group(2))
        
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
        
        # Check for primed operations (indicating canted/weak FM)
        if re.search(r",-1\s*$", content, re.MULTILINE):
            info['has_time_reversal'] = True
        else:
            info['has_time_reversal'] = False
        
        return info
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def load_magndata() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load all MAGNDATA mcif files and create index."""
    print("Loading MAGNDATA files...")
    magndata = []
    magndata_by_formula = defaultdict(list)
    
    mcif_files = list(Path(MAGNDATA_DIR).glob("*.mcif"))
    print(f"Found {len(mcif_files)} mcif files")
    
    for mcif_file in mcif_files:
        info = parse_mcif_file(str(mcif_file))
        if info:
            magndata.append(info)
            formula = info.get('magndata_formula', '')
            norm_formula = normalize_formula(formula)
            magndata_by_formula[norm_formula].append(info)
            
            # Also index by elements
            elements = parse_formula_to_elements(formula)
            elem_key = "_".join(sorted(elements.keys())).lower()
            magndata_by_formula[f"elem_{elem_key}"].append(info)
    
    print(f"Loaded {len(magndata)} MAGNDATA entries")
    return magndata, magndata_by_formula


def find_mp_matches(formula: str, mp_by_formula: Dict) -> List[Dict]:
    """Find Materials Project matches for a formula."""
    matches = []
    norm_formula = normalize_formula(formula)
    
    # Direct match
    if norm_formula in mp_by_formula:
        matches.extend(mp_by_formula[norm_formula])
    
    # Element-based match
    elements = parse_formula_to_elements(formula)
    elem_key = f"elem_{'_'.join(sorted(elements.keys())).lower()}"
    if elem_key in mp_by_formula:
        for mat in mp_by_formula[elem_key]:
            if formulas_match(formula, mat.get('formula', '')):
                if mat not in matches:
                    matches.append(mat)
    
    return matches


def find_magndata_matches(formula: str, magndata_by_formula: Dict) -> List[Dict]:
    """Find MAGNDATA matches for a formula."""
    matches = []
    norm_formula = normalize_formula(formula)
    
    # Direct match
    if norm_formula in magndata_by_formula:
        matches.extend(magndata_by_formula[norm_formula])
    
    # Element-based match
    elements = parse_formula_to_elements(formula)
    elem_key = f"elem_{'_'.join(sorted(elements.keys())).lower()}"
    if elem_key in magndata_by_formula:
        for entry in magndata_by_formula[elem_key]:
            if formulas_match(formula, entry.get('magndata_formula', '')):
                if entry not in matches:
                    matches.append(entry)
    
    return matches


def combine_databases():
    """Main function to combine all databases."""
    # Load Excel data
    print("Loading Excel data...")
    excel_df = pd.read_excel(EXCEL_PATH)
    print(f"Loaded {len(excel_df)} compounds from Excel")
    
    # Load MP and MAGNDATA
    mp_data, mp_by_formula = load_mp_data()
    magndata, magndata_by_formula = load_magndata()
    
    # Process each compound
    combined_data = []
    
    print("\nMatching compounds...")
    for idx, row in excel_df.iterrows():
        formula = row['Formula']
        if pd.isna(formula):
            continue
            
        entry = {
            # Excel data
            'formula': formula,
            'transition_type': row.get('Transition Type'),
            'Tc_exp': row.get('Tc'),
            'lowest_T_PM': row.get('Lowest T for PM'),
            'specific_heat_PM': row.get('specific heat (mJ/mol K^2) for PM'),
            'space_group_exp': row.get('Space group'),
            'notes_1': row.get('additional notes 1'),
            'notes_2': row.get('additional notes 2'),
            'mp_predicted_ground_state': row.get('Material project predicted ground state'),
        }
        
        # Find MP matches
        mp_matches = find_mp_matches(formula, mp_by_formula)
        entry['mp_match_count'] = len(mp_matches)
        
        if mp_matches:
            # Use most stable (lowest energy above hull) match
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
            
            # Get symmetry info
            symmetry = best_mp.get('symmetry', {})
            if symmetry:
                entry['mp_space_group_symbol'] = symmetry.get('symbol')
                entry['mp_space_group_number'] = symmetry.get('number')
                entry['mp_crystal_system'] = symmetry.get('crystal_system')
            
            # Store all MP IDs
            entry['all_mp_ids'] = ', '.join([m.get('material_id', '') for m in mp_matches[:10]])
        
        # Find MAGNDATA matches
        magndata_matches = find_magndata_matches(formula, magndata_by_formula)
        entry['magndata_match_count'] = len(magndata_matches)
        
        if magndata_matches:
            # Store first match details
            best_mag = magndata_matches[0]
            entry['magndata_id'] = best_mag.get('magndata_id')
            entry['magndata_formula'] = best_mag.get('magndata_formula')
            entry['magndata_magnetic_space_group'] = best_mag.get('magnetic_space_group')
            entry['magndata_parent_space_group'] = best_mag.get('parent_space_group')
            entry['magndata_Tc'] = best_mag.get('magndata_Tc')
            entry['magndata_icsd'] = best_mag.get('icsd_code')
            entry['magndata_irreps'] = best_mag.get('irreps')
            entry['magndata_has_time_reversal'] = best_mag.get('has_time_reversal')
            entry['magndata_properties'] = best_mag.get('magnetic_properties_details')
            
            # Lattice parameters
            for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
                entry[f'magndata_{param}'] = best_mag.get(f'magndata_{param}')
            
            # Store all MAGNDATA IDs
            entry['all_magndata_ids'] = ', '.join([m.get('magndata_id', '') for m in magndata_matches])
        
        combined_data.append(entry)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(excel_df)} compounds")
    
    # Create summary statistics
    mp_found = sum(1 for e in combined_data if e.get('mp_match_count', 0) > 0)
    magndata_found = sum(1 for e in combined_data if e.get('magndata_match_count', 0) > 0)
    both_found = sum(1 for e in combined_data if e.get('mp_match_count', 0) > 0 and e.get('magndata_match_count', 0) > 0)
    
    print(f"\n=== Summary ===")
    print(f"Total compounds from Excel: {len(combined_data)}")
    print(f"Found in Materials Project: {mp_found} ({100*mp_found/len(combined_data):.1f}%)")
    print(f"Found in MAGNDATA: {magndata_found} ({100*magndata_found/len(combined_data):.1f}%)")
    print(f"Found in both: {both_found} ({100*both_found/len(combined_data):.1f}%)")
    
    # Save to Excel
    print(f"\nSaving to Excel: {OUTPUT_PATH}")
    df = pd.DataFrame(combined_data)
    
    # Reorder columns for better readability
    priority_cols = [
        'formula', 'transition_type', 'Tc_exp', 
        'mp_match_count', 'mp_id', 'mp_ordering', 'mp_is_magnetic', 'mp_total_magnetization',
        'magndata_match_count', 'magndata_id', 'magndata_magnetic_space_group'
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]
    
    df.to_excel(OUTPUT_PATH, index=False)
    
    # Save to JSON
    print(f"Saving to JSON: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    print("\nDone!")
    return df


if __name__ == "__main__":
    df = combine_databases()
    
    # Show some examples with matches
    print("\n=== Sample entries with matches ===")
    matched = df[df['magndata_match_count'] > 0].head(5)
    if len(matched) > 0:
        print(matched[['formula', 'transition_type', 'mp_id', 'mp_ordering', 
                       'magndata_id', 'magndata_magnetic_space_group']].to_string())
