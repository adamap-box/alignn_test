"""
Create a new combined dataset JSON from:
1. combined_database.xlsx - compounds with mp_id
2. append_database.xlsx - all compounds

For each entry:
- Copy MP data and transition_type
- If transition_type is empty or TBD, use mp_ordering
- Include magndata_id if available
"""

import json
import pandas as pd
from pathlib import Path

# Paths
COMBINED_DB_PATH = "c:/workspace/alignn_test/combined_database.xlsx"
APPEND_DB_PATH = "c:/workspace/alignn_test/append_database.xlsx"
OUTPUT_JSON_PATH = "c:/workspace/alignn_test/magnetic_dataset.json"

# MP data fields to copy
MP_FIELDS = [
    'mp_id',
    'mp_formula',
    'mp_energy_per_atom',
    'mp_formation_energy',
    'mp_energy_above_hull',
    'mp_is_stable',
    'mp_band_gap',
    'mp_is_magnetic',
    'mp_ordering',
    'mp_total_magnetization',
    'mp_volume',
    'mp_density',
    'mp_nsites',
    'mp_space_group_symbol',
    'mp_space_group_number',
    'mp_crystal_system',
]


def normalize_transition_type(transition_type, mp_ordering):
    """
    Normalize transition type:
    - If empty, None, NaN, or 'TBD', use mp_ordering
    - Standardize common values
    """
    # Check if transition_type is empty or TBD
    if pd.isna(transition_type) or transition_type is None:
        use_mp = True
    elif str(transition_type).strip().upper() in ['', 'TBD', 'NAN', 'NONE']:
        use_mp = True
    else:
        use_mp = False
    
    if use_mp:
        # Map MP ordering to standard transition types
        if pd.isna(mp_ordering) or mp_ordering is None:
            return 'Unknown'
        
        mp_ordering = str(mp_ordering).strip().upper()
        ordering_map = {
            'FM': 'FM',
            'AFM': 'AFM',
            'FIM': 'FiM',
            'FRI': 'FiM',
            'NM': 'NM',  # Non-magnetic
            'PM': 'PM',  # Paramagnetic
        }
        return ordering_map.get(mp_ordering, mp_ordering)
    else:
        # Standardize the transition type
        tt = str(transition_type).strip().upper()
        if tt in ['AFM', 'ANTIFERROMAGNETIC']:
            return 'AFM'
        elif tt in ['FM', 'FERROMAGNETIC']:
            return 'FM'
        elif tt in ['FIM', 'FERRIMAGNETIC', 'FRI']:
            return 'FiM'
        elif 'CANTED' in tt or 'WEAK' in tt:
            return 'Canted AFM'
        else:
            return transition_type


def process_combined_database():
    """Process combined_database.xlsx"""
    print("Processing combined_database.xlsx...")
    df = pd.read_excel(COMBINED_DB_PATH)
    
    entries = []
    skipped = 0
    
    for idx, row in df.iterrows():
        # Skip if no mp_id
        mp_id = row.get('mp_id')
        if pd.isna(mp_id) or mp_id is None or str(mp_id).strip() == '':
            skipped += 1
            continue
        
        # Create entry
        entry = {
            'formula': row.get('formula'),
            'source': 'combined_database',
        }
        
        # Copy MP fields
        for field in MP_FIELDS:
            value = row.get(field)
            if pd.notna(value):
                entry[field] = value
            else:
                entry[field] = None
        
        # Get transition type
        transition_type = row.get('transition_type')
        mp_ordering = row.get('mp_ordering')
        entry['transition_type'] = normalize_transition_type(transition_type, mp_ordering)
        entry['original_transition_type'] = transition_type if pd.notna(transition_type) else None
        
        # Get magndata_id if available
        magndata_id = row.get('magndata_id')
        if pd.notna(magndata_id) and magndata_id is not None:
            entry['magndata_id'] = str(magndata_id)
        else:
            entry['magndata_id'] = None
        
        # Get experimental Tc if available
        tc_exp = row.get('Tc_exp')
        if pd.notna(tc_exp):
            entry['Tc_exp'] = tc_exp
        else:
            entry['Tc_exp'] = None
        
        entries.append(entry)
    
    print(f"  Processed: {len(entries)} entries (skipped {skipped} without mp_id)")
    return entries


def process_append_database():
    """Process append_database.xlsx"""
    print("Processing append_database.xlsx...")
    df = pd.read_excel(APPEND_DB_PATH)
    
    entries = []
    
    for idx, row in df.iterrows():
        mp_id = row.get('mp_id')
        if pd.isna(mp_id) or mp_id is None:
            continue
        
        # Create entry
        entry = {
            'formula': row.get('formula'),
            'source': 'append_database',
        }
        
        # Copy MP fields
        for field in MP_FIELDS:
            value = row.get(field)
            if pd.notna(value):
                entry[field] = value
            else:
                entry[field] = None
        
        # Get transition type (already inferred in append script)
        transition_type = row.get('transition_type')
        mp_ordering = row.get('mp_ordering')
        entry['transition_type'] = normalize_transition_type(transition_type, mp_ordering)
        entry['original_transition_type'] = transition_type if pd.notna(transition_type) else None
        
        # Get magndata_id (should always be available from append)
        magndata_id = row.get('magndata_id')
        if pd.notna(magndata_id) and magndata_id is not None:
            entry['magndata_id'] = str(magndata_id)
        else:
            entry['magndata_id'] = None
        
        # Get experimental Tc if available
        tc_exp = row.get('Tc_exp')
        if pd.notna(tc_exp):
            entry['Tc_exp'] = tc_exp
        else:
            entry['Tc_exp'] = None
        
        entries.append(entry)
    
    print(f"  Processed: {len(entries)} entries")
    return entries


def create_dataset():
    """Main function to create the dataset"""
    # Process both databases
    combined_entries = process_combined_database()
    append_entries = process_append_database()
    
    # Merge entries
    all_entries = combined_entries + append_entries
    
    # Remove duplicates by mp_id (keep first occurrence)
    seen_mp_ids = set()
    unique_entries = []
    duplicates = 0
    
    for entry in all_entries:
        mp_id = entry.get('mp_id')
        if mp_id and mp_id not in seen_mp_ids:
            seen_mp_ids.add(mp_id)
            unique_entries.append(entry)
        elif mp_id:
            duplicates += 1
    
    # Create summary statistics
    transition_counts = {}
    with_magndata = 0
    with_tc = 0
    
    for entry in unique_entries:
        tt = entry.get('transition_type', 'Unknown')
        transition_counts[tt] = transition_counts.get(tt, 0) + 1
        
        if entry.get('magndata_id'):
            with_magndata += 1
        if entry.get('Tc_exp'):
            with_tc += 1
    
    # Create final dataset structure
    dataset = {
        'metadata': {
            'description': 'Combined magnetic materials dataset',
            'sources': ['combined_database.xlsx', 'append_database.xlsx'],
            'total_entries': len(unique_entries),
            'with_magndata': with_magndata,
            'with_Tc': with_tc,
            'duplicates_removed': duplicates,
            'transition_type_counts': transition_counts,
        },
        'entries': unique_entries
    }
    
    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Total entries: {len(unique_entries)}")
    print(f"With MAGNDATA: {with_magndata}")
    print(f"With Tc: {with_tc}")
    print(f"Duplicates removed: {duplicates}")
    print(f"\nTransition types:")
    for tt, count in sorted(transition_counts.items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")
    
    # Save to JSON
    print(f"\nSaving to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    print("Done!")
    return dataset


if __name__ == "__main__":
    dataset = create_dataset()
    
    # Show sample entries
    print("\n=== Sample Entries ===")
    for entry in dataset['entries'][:5]:
        print(f"\n{entry['formula']} ({entry['mp_id']}):")
        print(f"  Transition: {entry['transition_type']}")
        print(f"  MAGNDATA: {entry['magndata_id']}")
        print(f"  Tc: {entry['Tc_exp']}")
        print(f"  MP ordering: {entry['mp_ordering']}")
        print(f"  Is magnetic: {entry['mp_is_magnetic']}")
