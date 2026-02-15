"""
Add crystal structure info (lattice and atom positions) from MP data
to each compound in magnetic_dataset.json
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
MP_DATA_PATH = "c:/workspace/mp_data/mp_all_materials.json"
INPUT_DATASET_PATH = "c:/workspace/alignn_test/magnetic_dataset.json"
OUTPUT_DATASET_PATH = "c:/workspace/alignn_test/magnetic_dataset_with_structure.json"


def load_mp_structures():
    """Load MP data and create index by material_id"""
    print("Loading Materials Project data...")
    with open(MP_DATA_PATH, 'r') as f:
        mp_data = json.load(f)
    
    mp_by_id = {}
    for mat in mp_data:
        mp_id = mat.get('material_id')
        if mp_id:
            mp_by_id[mp_id] = mat
    
    print(f"Loaded {len(mp_by_id)} materials with structures")
    return mp_by_id


def extract_structure_info(mp_entry):
    """Extract crystal structure info from MP entry"""
    structure = mp_entry.get('structure', {})
    if not structure:
        return None
    
    lattice = structure.get('lattice', {})
    sites = structure.get('sites', [])
    
    # Extract lattice info
    lattice_info = {
        'matrix': lattice.get('matrix'),  # 3x3 lattice vectors
        'a': lattice.get('a'),
        'b': lattice.get('b'),
        'c': lattice.get('c'),
        'alpha': lattice.get('alpha'),
        'beta': lattice.get('beta'),
        'gamma': lattice.get('gamma'),
        'volume': lattice.get('volume'),
    }
    
    # Extract atom positions
    atoms = []
    for site in sites:
        atom = {
            'species': site.get('species', [{}])[0].get('element') if site.get('species') else None,
            'abc': site.get('abc'),  # Fractional coordinates
            'xyz': site.get('xyz'),  # Cartesian coordinates
            'label': site.get('label'),
        }
        
        # Handle species with oxidation state
        if site.get('species'):
            species_list = site['species']
            if len(species_list) > 0:
                atom['species'] = species_list[0].get('element')
                atom['oxidation_state'] = species_list[0].get('oxidation_state')
                atom['occu'] = species_list[0].get('occu', 1.0)
        
        atoms.append(atom)
    
    return {
        'lattice': lattice_info,
        'atoms': atoms,
        'nsites': len(atoms),
    }


def add_structures_to_dataset():
    """Add structure info to each compound in dataset"""
    # Load MP data
    mp_by_id = load_mp_structures()
    
    # Load existing dataset
    print(f"\nLoading dataset from {INPUT_DATASET_PATH}...")
    with open(INPUT_DATASET_PATH, 'r') as f:
        dataset = json.load(f)
    
    entries = dataset.get('entries', [])
    print(f"Processing {len(entries)} entries...")
    
    # Add structure to each entry
    found = 0
    not_found = 0
    
    for entry in entries:
        mp_id = entry.get('mp_id')
        if not mp_id:
            not_found += 1
            entry['structure'] = None
            continue
        
        mp_entry = mp_by_id.get(mp_id)
        if not mp_entry:
            not_found += 1
            entry['structure'] = None
            continue
        
        structure_info = extract_structure_info(mp_entry)
        if structure_info:
            entry['structure'] = structure_info
            found += 1
        else:
            entry['structure'] = None
            not_found += 1
    
    # Update metadata
    dataset['metadata']['with_structure'] = found
    dataset['metadata']['description'] = 'Combined magnetic materials dataset with crystal structures'
    
    print(f"\nStructures added: {found}")
    print(f"Without structure: {not_found}")
    
    # Save updated dataset
    print(f"\nSaving to {OUTPUT_DATASET_PATH}...")
    with open(OUTPUT_DATASET_PATH, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Also save a compact version without indentation (smaller file)
    compact_path = OUTPUT_DATASET_PATH.replace('.json', '_compact.json')
    print(f"Saving compact version to {compact_path}...")
    with open(compact_path, 'w') as f:
        json.dump(dataset, f)
    
    print("\nDone!")
    return dataset


if __name__ == "__main__":
    dataset = add_structures_to_dataset()
    
    # Show sample entry with structure
    print("\n=== Sample Entry with Structure ===")
    for entry in dataset['entries']:
        if entry.get('structure'):
            print(f"\n{entry['formula']} ({entry['mp_id']}):")
            print(f"  Transition: {entry['transition_type']}")
            struct = entry['structure']
            lattice = struct['lattice']
            print(f"  Lattice: a={lattice['a']:.3f}, b={lattice['b']:.3f}, c={lattice['c']:.3f}")
            print(f"  Angles: alpha={lattice['alpha']:.1f}, beta={lattice['beta']:.1f}, gamma={lattice['gamma']:.1f}")
            print(f"  Volume: {lattice['volume']:.2f} A^3")
            print(f"  Atoms ({struct['nsites']}):")
            for atom in struct['atoms'][:5]:
                abc = atom['abc']
                print(f"    {atom['species']}: ({abc[0]:.4f}, {abc[1]:.4f}, {abc[2]:.4f})")
            if struct['nsites'] > 5:
                print(f"    ... and {struct['nsites'] - 5} more atoms")
            break
