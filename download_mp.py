"""Download all materials from Materials Project using mp-api.

Requirements:
    pip install mp-api

Usage:
    1. Get your API key from https://materialsproject.org/api
    2. Set your API key below or as environment variable MP_API_KEY
    3. Run: py -3.11 download_mp.py
"""
import os
import json
from datetime import datetime

# Set your Materials Project API key here or use environment variable
API_KEY = os.environ.get("MP_API_KEY", "711wCSiRkuGXuBB9MJPb1OZSy1nx7aml")

def download_all_materials(output_dir="mp_data", batch_size=1000):
    """Download all materials from Materials Project."""
    from mp_api.client import MPRester
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to Materials Project API...")
    print(f"Output directory: {output_dir}")
    
    with MPRester(API_KEY) as mpr:
        # Get summary of all materials
        print("Fetching material IDs...")
        
        # Query all materials with basic properties
        # Fields to download - adjust as needed
        fields = [
            "material_id",
            "formula_pretty", 
            "structure",
            "energy_per_atom",
            "formation_energy_per_atom",
            "energy_above_hull",
            "is_stable",
            "band_gap",
            "is_magnetic",
            "ordering",
            "total_magnetization",
            "symmetry",
            "volume",
            "density",
            "nsites",
            "elements",
        ]
        
        print(f"Downloading materials with fields: {fields}")
        
        # Get all materials (this may take a while)
        docs = mpr.materials.summary.search(
            fields=fields,
            num_chunks=None,  # Get all
        )
        
        print(f"Downloaded {len(docs)} materials")
        
        # Convert to serializable format
        materials = []
        for i, doc in enumerate(docs):
            if i % 10000 == 0:
                print(f"Processing {i}/{len(docs)}...")
            
            mat = {
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "energy_per_atom": doc.energy_per_atom,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "is_stable": doc.is_stable,
                "band_gap": doc.band_gap,
                "is_magnetic": doc.is_magnetic,
                "ordering": str(doc.ordering) if doc.ordering else None,
                "total_magnetization": doc.total_magnetization,
                "volume": doc.volume,
                "density": doc.density,
                "nsites": doc.nsites,
                "elements": [str(e) for e in doc.elements] if doc.elements else [],
            }
            
            # Convert structure to dict if available
            if doc.structure:
                mat["structure"] = doc.structure.as_dict()
            
            # Add symmetry info
            if doc.symmetry:
                mat["symmetry"] = {
                    "crystal_system": str(doc.symmetry.crystal_system) if doc.symmetry.crystal_system else None,
                    "symbol": doc.symmetry.symbol,
                    "number": doc.symmetry.number,
                }
            
            materials.append(mat)
        
        # Save to JSON
        output_file = os.path.join(output_dir, "mp_all_materials.json")
        print(f"Saving to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(materials, f)
        
        print(f"Done! Saved {len(materials)} materials")
        
        # Save metadata
        meta_file = os.path.join(output_dir, "metadata.json")
        metadata = {
            "download_date": datetime.now().isoformat(),
            "total_materials": len(materials),
            "fields": fields,
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return materials


def download_magnetic_materials(output_dir="mp_data"):
    """Download only magnetic materials."""
    from mp_api.client import MPRester
    
    os.makedirs(output_dir, exist_ok=True)
    
    with MPRester(API_KEY) as mpr:
        print("Fetching magnetic materials...")
        
        # Query magnetic materials
        docs = mpr.materials.summary.search(
            is_magnetic=True,
            fields=[
                "material_id",
                "formula_pretty",
                "structure", 
                "formation_energy_per_atom",
                "ordering",
                "total_magnetization",
                "symmetry",
            ],
        )
        
        print(f"Found {len(docs)} magnetic materials")
        
        materials = []
        for doc in docs:
            mat = {
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "ordering": str(doc.ordering) if doc.ordering else None,
                "total_magnetization": doc.total_magnetization,
            }
            if doc.structure:
                mat["structure"] = doc.structure.as_dict()
            materials.append(mat)
        
        output_file = os.path.join(output_dir, "mp_magnetic_materials.json")
        with open(output_file, "w") as f:
            json.dump(materials, f)
        
        print(f"Saved {len(materials)} magnetic materials to {output_file}")
        return materials


def search_material(material_id):
    """Search for a specific material by ID with lattice and atomic positions."""
    from mp_api.client import MPRester
    
    with MPRester(API_KEY) as mpr:
        doc = mpr.materials.summary.get_data_by_id(material_id)
        print(f"Material: {doc.material_id}")
        print(f"Formula: {doc.formula_pretty}")
        print(f"Formation energy: {doc.formation_energy_per_atom} eV/atom")
        
        # Magnetic properties
        print(f"\n--- Magnetic Properties ---")
        print(f"Is magnetic: {doc.is_magnetic}")
        print(f"Magnetic Ordering: {doc.ordering}")
        print(f"Total magnetization: {doc.total_magnetization} μB/f.u.")
        if hasattr(doc, 'total_magnetization_normalized_vol') and doc.total_magnetization_normalized_vol:
            print(f"Magnetization (normalized): {doc.total_magnetization_normalized_vol} μB/Å³")
        
        # Print lattice and atomic positions
        if doc.structure:
            struct = doc.structure
            print(f"\n--- Lattice Parameters ---")
            print(f"a = {struct.lattice.a:.4f} Å")
            print(f"b = {struct.lattice.b:.4f} Å")
            print(f"c = {struct.lattice.c:.4f} Å")
            print(f"alpha = {struct.lattice.alpha:.2f}°")
            print(f"beta  = {struct.lattice.beta:.2f}°")
            print(f"gamma = {struct.lattice.gamma:.2f}°")
            print(f"Volume = {struct.lattice.volume:.2f} Å³")
            
            print(f"\n--- Lattice Matrix ---")
            for i, row in enumerate(struct.lattice.matrix):
                print(f"  a{i+1} = [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")
            
            print(f"\n--- Atomic Positions ({len(struct)} atoms) ---")
            print(f"{'Site':<5} {'Element':<8} {'Frac x':<12} {'Frac y':<12} {'Frac z':<12}")
            print("-" * 55)
            for i, site in enumerate(struct):
                fc = site.frac_coords
                print(f"{i:<5} {str(site.specie):<8} {fc[0]:12.6f} {fc[1]:12.6f} {fc[2]:12.6f}")
            
            print(f"\n--- Cartesian Coordinates ---")
            print(f"{'Site':<5} {'Element':<8} {'x (Å)':<12} {'y (Å)':<12} {'z (Å)':<12}")
            print("-" * 55)
            for i, site in enumerate(struct):
                cc = site.coords
                print(f"{i:<5} {str(site.specie):<8} {cc[0]:12.6f} {cc[1]:12.6f} {cc[2]:12.6f}")
        
        return doc


def get_magnetic_data(material_id, output_file=None):
    """Get detailed magnetic properties for a material."""
    from mp_api.client import MPRester
    
    with MPRester(API_KEY) as mpr:
        # Get summary data
        doc = mpr.materials.summary.get_data_by_id(material_id)
        
        data = {
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "is_magnetic": doc.is_magnetic,
            "ordering": str(doc.ordering) if doc.ordering else None,
            "total_magnetization": doc.total_magnetization,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
        }
        
        # Try to get magnetism-specific data
        try:
            mag_docs = mpr.magnetism.search(material_ids=[material_id])
            if mag_docs:
                mag = mag_docs[0]
                data["magnetism"] = {
                    "ordering": str(mag.ordering) if hasattr(mag, 'ordering') and mag.ordering else None,
                    "total_magnetization": mag.total_magnetization if hasattr(mag, 'total_magnetization') else None,
                    "total_magnetization_normalized_vol": mag.total_magnetization_normalized_vol if hasattr(mag, 'total_magnetization_normalized_vol') else None,
                    "total_magnetization_normalized_formula_units": mag.total_magnetization_normalized_formula_units if hasattr(mag, 'total_magnetization_normalized_formula_units') else None,
                }
                # Get magnetic moments per site if available
                if hasattr(mag, 'magmoms') and mag.magmoms:
                    data["magnetism"]["magmoms"] = mag.magmoms
                print(f"Magnetic ordering: {data['magnetism']['ordering']}")
                print(f"Total magnetization: {data['magnetism']['total_magnetization']} μB")
        except Exception as e:
            print(f"Could not fetch magnetism data: {e}")
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved magnetic data to {output_file}")
        
        return data


def get_structure_data(material_id, output_file=None):
    """Get lattice and atomic positions for a material and optionally save to file."""
    from mp_api.client import MPRester
    
    with MPRester(API_KEY) as mpr:
        doc = mpr.materials.summary.get_data_by_id(material_id)
        
        if not doc.structure:
            print(f"No structure data for {material_id}")
            return None
        
        struct = doc.structure
        
        data = {
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "lattice": {
                "a": struct.lattice.a,
                "b": struct.lattice.b,
                "c": struct.lattice.c,
                "alpha": struct.lattice.alpha,
                "beta": struct.lattice.beta,
                "gamma": struct.lattice.gamma,
                "volume": struct.lattice.volume,
                "matrix": struct.lattice.matrix.tolist(),
            },
            "sites": [
                {
                    "element": str(site.specie),
                    "frac_coords": site.frac_coords.tolist(),
                    "cart_coords": site.coords.tolist(),
                }
                for site in struct
            ],
            "nsites": len(struct),
            "is_magnetic": doc.is_magnetic,
            "ordering": str(doc.ordering) if doc.ordering else None,
            "total_magnetization": doc.total_magnetization,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
        }
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved structure data to {output_file}")
        
        return data


if __name__ == "__main__":
    import sys
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Materials Project API key!")
        print("Get one at: https://materialsproject.org/api")
        print("\nOptions:")
        print("  1. Edit this script and set API_KEY")
        print("  2. Set environment variable: set MP_API_KEY=your_key")
        sys.exit(1)
    
    # Choose what to download:
    
    # Option 1: Download ALL materials (~150k, may take 30+ minutes)
    download_all_materials()
    
    # Option 2: Download only magnetic materials (~15k)
    # download_magnetic_materials()
    
    # Option 3: Search specific material
    # search_material("mp-22283")
    
    print("Uncomment one of the download functions above to run")
    print("\nExample: Search for mp-22283:")
    search_material("mp-22283")
