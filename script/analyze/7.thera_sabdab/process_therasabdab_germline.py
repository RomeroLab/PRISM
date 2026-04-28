#!/usr/bin/env python
# coding: utf-8
"""
Process TheraSAbDab antibody dataset to extract germline information and mutations.

This script:
1. Reads therasabdab.csv with therapeutic antibody sequences
2. Uses ANARCI to assign V/J germline genes and get IMGT numbering
3. Retrieves germline sequences and identifies non-germline mutations
4. Formats mutations as mutation codes (e.g., F28S, I51S, T56S)
5. Generates region masks for FR/CDR annotation
6. Saves results to therasabdab_germline.csv

Usage:
    conda run -n devant python process_therasabdab_germline.py

Output columns added:
    - v_gene_heavy, j_gene_heavy: V/J gene assignments for heavy chain
    - v_gene_light, j_gene_light: V/J gene assignments for light chain
    - germline_heavy, germline_light: Full germline sequences
    - mutations_heavy, mutations_light: Comma-separated mutation codes
    - mutation_count_heavy, mutation_count_light: Number of mutations
    - region_mask_heavy, region_mask_light: IMGT region masks
      (FR1=0, CDR1=1, FR2=2, CDR2=3, FR3=4, CDR3=5, FR4=6)
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Try importing anarci
try:
    from anarci import run_anarci
    from anarci.germlines import all_germlines
    ANARCI_AVAILABLE = True
except ImportError:
    ANARCI_AVAILABLE = False
    print("WARNING: anarci not available. Install with: pip install anarci")


# IMGT region definitions for reference
IMGT_REGIONS = {
    'FR1': (1, 26),
    'CDR1': (27, 38),
    'FR2': (39, 55),
    'CDR2': (56, 65),
    'FR3': (66, 104),
    'CDR3': (105, 117),
    'FR4': (118, 128)
}

# Region ID mapping for region mask (same as add_gene_info.py)
REGION_TO_ID = {
    'FR1': '0', 'CDR1': '1', 'FR2': '2', 'CDR2': '3',
    'FR3': '4', 'CDR3': '5', 'FR4': '6'
}


def get_region_id(imgt_pos):
    """
    Convert IMGT position number to region ID character.

    Args:
        imgt_pos: IMGT position number (1-128)

    Returns:
        Region ID character ('0'-'6') or None if not in any region
    """
    for region, (start, end) in IMGT_REGIONS.items():
        if start <= imgt_pos <= end:
            return REGION_TO_ID[region]
    return None


def get_germline_sequence_from_anarci(gene_name, chain_type='H', species='human'):
    """
    Retrieve germline sequence from ANARCI's internal database.

    ANARCI germline structure: all_germlines[gene_type][chain_type][species][gene_name]
    Example: all_germlines['V']['H']['human']['IGHV1-2*01']

    Args:
        gene_name: Gene name (e.g., 'IGHV1-2*01' or 'IGHV1-2')
        chain_type: 'H' for heavy, 'K' or 'L' for light
        species: Species name (default: 'human')

    Returns:
        Germline sequence string or None if not found
    """
    if not ANARCI_AVAILABLE or not gene_name:
        return None

    # Determine chain key from gene name
    if gene_name.startswith('IGH'):
        chain_key = 'H'
    elif gene_name.startswith('IGK'):
        chain_key = 'K'
    elif gene_name.startswith('IGL'):
        chain_key = 'L'
    else:
        # Try the provided chain_type as fallback
        chain_key = chain_type

    # Determine if V or J gene
    if 'V' in gene_name:
        gene_type = 'V'
    elif 'J' in gene_name:
        gene_type = 'J'
    else:
        return None

    # Access ANARCI germline database
    # Structure: all_germlines[gene_type][chain_type][species][gene_name]
    try:
        germline_db = all_germlines[gene_type][chain_key][species]

        # Try exact match first
        if gene_name in germline_db:
            return germline_db[gene_name]

        # Try without allele
        base_gene = gene_name.split('*')[0]
        for key in germline_db:
            if key.startswith(base_gene):
                return germline_db[key]

        return None
    except (KeyError, TypeError):
        return None


def extract_germline_info(seq, chain_type='H', allowed_species=['human']):
    """
    Run ANARCI on a single sequence and extract germline information.

    Args:
        seq: Amino acid sequence string
        chain_type: 'H' for heavy, 'L' for light
        allowed_species: List of allowed species

    Returns:
        Dictionary with numbering, v_gene, j_gene, and germline_seq
    """
    if not ANARCI_AVAILABLE or not seq or seq == 'na':
        return None

    try:
        results = run_anarci(
            seq=[('seq', seq)],
            scheme='imgt',
            output=False,
            assign_germline=True,
            allowed_species=allowed_species
        )

        _, numberings, alignments, _ = results

        if not alignments or not alignments[0]:
            return None

        alignment = alignments[0][0]
        numbering = numberings[0][0][0]  # List of ((pos, insertion), aa) tuples

        # Extract gene assignments
        v_gene_info = alignment['germlines']['v_gene']
        j_gene_info = alignment['germlines']['j_gene']

        v_gene = v_gene_info[0][1] if v_gene_info[0] else None
        j_gene = j_gene_info[0][1] if j_gene_info[0] else None

        # Get chain type detected
        detected_chain = alignment.get('chain_type', chain_type)

        return {
            'numbering': numbering,
            'v_gene': v_gene,
            'j_gene': j_gene,
            'v_identity': v_gene_info[1] if len(v_gene_info) > 1 else None,
            'j_identity': j_gene_info[1] if len(j_gene_info) > 1 else None,
            'chain_type': detected_chain,
            'species': alignment.get('species', 'unknown')
        }
    except Exception:
        return None


def build_germline_from_numbering(numbering, v_gene, j_gene, chain_type='H', species='human'):
    """
    Build a full germline sequence aligned to the query sequence positions.

    ANARCI germline sequences are 128 characters long, corresponding to IMGT positions 1-128.
    Gaps ('-') indicate positions not present in that germline gene.
    V genes cover positions 1-104 (FR1-FR3), J genes cover positions 105-128 (CDR3-FR4).

    Args:
        numbering: List of ((pos, insertion), aa) tuples from ANARCI
        v_gene: V gene name
        j_gene: J gene name
        chain_type: 'H', 'K', or 'L'
        species: Species name

    Returns:
        Tuple of (germline_sequence, position_mapping)
    """
    if not ANARCI_AVAILABLE:
        return None, None

    # Get germline V and J sequences from ANARCI database
    # These are already IMGT-aligned (128 positions with gaps)
    v_germline = get_germline_sequence_from_anarci(v_gene, chain_type, species)
    j_germline = get_germline_sequence_from_anarci(j_gene, chain_type, species)

    if not v_germline or not j_germline:
        return None, None

    # Build position -> germline AA mapping
    # IMGT positions 1-128 map to string indices 0-127
    germline_pos_map = {}

    # V gene: typically covers positions 1-104 (FR1 through FR3)
    for i, aa in enumerate(v_germline):
        imgt_pos = i + 1  # IMGT positions are 1-indexed
        if aa != '-':
            germline_pos_map[(imgt_pos, ' ')] = aa

    # J gene: typically covers positions 105-128 (CDR3 through FR4)
    for i, aa in enumerate(j_germline):
        imgt_pos = i + 1
        if aa != '-':
            # J gene overrides any V gene data at overlapping positions
            germline_pos_map[(imgt_pos, ' ')] = aa

    # Build aligned germl ine sequence and position mapping
    germline_seq = []
    position_mapping = []

    for (pos, ins), query_aa in numbering:
        if query_aa == '-':
            continue

        # Look up germline AA at this position
        # Note: insertion codes (like 111A) are for CDR3 insertions not in germline
        germ_aa = germline_pos_map.get((pos, ' '), '-')

        # For insertion positions (CDR3 insertions), germline is typically '-'
        if ins.strip():
            germ_aa = '-'

        germline_seq.append(germ_aa)
        position_mapping.append({
            'imgt_pos': pos,
            'insertion': ins,
            'query_aa': query_aa,
            'germline_aa': germ_aa
        })

    return ''.join(germline_seq), position_mapping


def build_region_mask_from_numbering(numbering):
    """
    Build a region mask string from ANARCI numbering.

    Each character in the mask represents the IMGT region of the corresponding
    amino acid position: FR1=0, CDR1=1, FR2=2, CDR2=3, FR3=4, CDR3=5, FR4=6.

    Args:
        numbering: List of ((pos, insertion), aa) tuples from ANARCI

    Returns:
        Region mask string (e.g., "000000011111222223333344444445555556666")
    """
    if not numbering:
        return None

    mask_chars = []
    for (pos, ins), aa in numbering:
        # Skip gaps
        if aa == '-':
            continue

        # Get region ID for this IMGT position
        region_id = get_region_id(pos)
        if region_id is not None:
            mask_chars.append(region_id)

    return ''.join(mask_chars) if mask_chars else None


def find_mutations(position_mapping):
    """
    Find non-germline mutations by comparing query to germline.

    Args:
        position_mapping: List of position dictionaries from build_germline_from_numbering

    Returns:
        List of mutation strings in format "G28S" (germline_aa + position + query_aa)
    """
    if not position_mapping:
        return []

    mutations = []

    for pos_info in position_mapping:
        query_aa = pos_info['query_aa']
        germ_aa = pos_info['germline_aa']
        imgt_pos = pos_info['imgt_pos']
        ins = pos_info['insertion']

        # Skip if no germline info (CDR3 region)
        if germ_aa == '-':
            continue

        # Check for mutation
        if query_aa != germ_aa:
            # Format: {germline_aa}{IMGT_position}{query_aa}
            # Include insertion code if present
            if ins.strip():
                pos_str = f"{imgt_pos}{ins.strip()}"
            else:
                pos_str = str(imgt_pos)

            mutation = f"{germ_aa}{pos_str}{query_aa}"
            mutations.append(mutation)

    return mutations


def process_sequence(seq, chain_type='H', allowed_species=['human']):
    """
    Full pipeline to process a single sequence.

    Args:
        seq: Amino acid sequence
        chain_type: 'H' for heavy, 'L' for light
        allowed_species: List of allowed species

    Returns:
        Dictionary with all germline, mutation, and region information
    """
    result = {
        'v_gene': None,
        'j_gene': None,
        'germline_seq': None,
        'mutations': None,
        'mutation_count': 0,
        'region_mask': None,
        'species': None
    }

    if not seq or seq == 'na' or not ANARCI_AVAILABLE:
        return result

    # Run ANARCI
    info = extract_germline_info(seq, chain_type, allowed_species)
    if not info:
        return result

    result['v_gene'] = info['v_gene']
    result['j_gene'] = info['j_gene']
    result['species'] = info['species']

    # Build region mask from numbering (always available if ANARCI succeeds)
    result['region_mask'] = build_region_mask_from_numbering(info['numbering'])

    # Build germline and find mutations
    detected_chain = info['chain_type']
    germline_seq, pos_mapping = build_germline_from_numbering(
        info['numbering'],
        info['v_gene'],
        info['j_gene'],
        chain_type=detected_chain,
        species=info['species']
    )

    if germline_seq:
        result['germline_seq'] = germline_seq
        mutations = find_mutations(pos_mapping)
        result['mutations'] = ','.join(mutations) if mutations else ''
        result['mutation_count'] = len(mutations)

    return result


def process_therasabdab(input_path, output_path, allowed_species=['human']):
    """
    Process the full TheraSAbDab dataset.

    Args:
        input_path: Path to therasabdab.csv
        output_path: Path for output CSV
        allowed_species: Species to allow for germline assignment
    """
    print("=" * 80)
    print("TheraSAbDab Germline Analysis Pipeline")
    print("=" * 80)

    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)[:10]}...")

    # Filter valid sequences
    df_valid = df[
        (df['HeavySequence'] != 'na') &
        (df['LightSequence'] != 'na') &
        (df['HeavySequence'].notna()) &
        (df['LightSequence'].notna())
    ].copy()
    print(f"  Valid paired sequences: {len(df_valid)}")

    # Initialize new columns
    new_cols = [
        'v_gene_heavy', 'j_gene_heavy', 'germline_heavy', 'mutations_heavy', 'mutation_count_heavy',
        'region_mask_heavy',
        'v_gene_light', 'j_gene_light', 'germline_light', 'mutations_light', 'mutation_count_light',
        'region_mask_light',
        'species_heavy', 'species_light'
    ]
    for col in new_cols:
        df_valid[col] = None

    # Process heavy chains
    print("\nProcessing heavy chains...")
    for idx in tqdm(df_valid.index, desc="  Heavy chains"):
        seq = df_valid.loc[idx, 'HeavySequence']
        result = process_sequence(seq, chain_type='H', allowed_species=allowed_species)

        df_valid.loc[idx, 'v_gene_heavy'] = result['v_gene']
        df_valid.loc[idx, 'j_gene_heavy'] = result['j_gene']
        df_valid.loc[idx, 'germline_heavy'] = result['germline_seq']
        df_valid.loc[idx, 'mutations_heavy'] = result['mutations']
        df_valid.loc[idx, 'mutation_count_heavy'] = result['mutation_count']
        df_valid.loc[idx, 'region_mask_heavy'] = result['region_mask']
        df_valid.loc[idx, 'species_heavy'] = result['species']

    # Process light chains
    print("\nProcessing light chains...")
    for idx in tqdm(df_valid.index, desc="  Light chains"):
        seq = df_valid.loc[idx, 'LightSequence']
        result = process_sequence(seq, chain_type='L', allowed_species=allowed_species)

        df_valid.loc[idx, 'v_gene_light'] = result['v_gene']
        df_valid.loc[idx, 'j_gene_light'] = result['j_gene']
        df_valid.loc[idx, 'germline_light'] = result['germline_seq']
        df_valid.loc[idx, 'mutations_light'] = result['mutations']
        df_valid.loc[idx, 'mutation_count_light'] = result['mutation_count']
        df_valid.loc[idx, 'region_mask_light'] = result['region_mask']
        df_valid.loc[idx, 'species_light'] = result['species']

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nHeavy chain annotations:")
    print(f"  V gene assigned: {df_valid['v_gene_heavy'].notna().sum()}/{len(df_valid)}")
    print(f"  J gene assigned: {df_valid['j_gene_heavy'].notna().sum()}/{len(df_valid)}")
    print(f"  Region mask: {df_valid['region_mask_heavy'].notna().sum()}/{len(df_valid)}")
    print(f"  Germline found: {df_valid['germline_heavy'].notna().sum()}/{len(df_valid)}")
    print(f"  With mutations: {(df_valid['mutation_count_heavy'] > 0).sum()}/{len(df_valid)}")

    avg_mut_heavy = df_valid['mutation_count_heavy'].mean()
    max_mut_heavy = df_valid['mutation_count_heavy'].max()
    print(f"  Avg mutations: {avg_mut_heavy:.1f}, Max: {max_mut_heavy}")

    print("\nLight chain annotations:")
    print(f"  V gene assigned: {df_valid['v_gene_light'].notna().sum()}/{len(df_valid)}")
    print(f"  J gene assigned: {df_valid['j_gene_light'].notna().sum()}/{len(df_valid)}")
    print(f"  Region mask: {df_valid['region_mask_light'].notna().sum()}/{len(df_valid)}")
    print(f"  Germline found: {df_valid['germline_light'].notna().sum()}/{len(df_valid)}")
    print(f"  With mutations: {(df_valid['mutation_count_light'] > 0).sum()}/{len(df_valid)}")

    avg_mut_light = df_valid['mutation_count_light'].mean()
    max_mut_light = df_valid['mutation_count_light'].max()
    print(f"  Avg mutations: {avg_mut_light:.1f}, Max: {max_mut_light}")

    # Show examples
    print("\nSample annotations (first 3 with mutations):")
    sample = df_valid[df_valid['mutations_heavy'].notna() & (df_valid['mutation_count_heavy'] > 0)].head(3)
    for idx, row in sample.iterrows():
        print(f"\n  {row['Therapeutic']}:")
        print(f"    Heavy V/J: {row['v_gene_heavy']} / {row['j_gene_heavy']}")
        print(f"    Heavy mutations ({row['mutation_count_heavy']}): {row['mutations_heavy'][:80]}...")
        if pd.notna(row['region_mask_heavy']):
            print(f"    Heavy region mask: {row['region_mask_heavy'][:40]}...")
        print(f"    Light V/J: {row['v_gene_light']} / {row['j_gene_light']}")
        if row['mutations_light']:
            print(f"    Light mutations ({row['mutation_count_light']}): {row['mutations_light'][:80]}...")
        if pd.notna(row['region_mask_light']):
            print(f"    Light region mask: {row['region_mask_light'][:40]}...")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved to: {output_path}")
    print(f"Total rows saved: {len(df_valid)}")
    print("=" * 80)

    return df_valid


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Process TheraSAbDab to extract germline info and mutations'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../../data/therasabdab.csv',
        help='Path to input therasabdab.csv'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../../data/therasabdab_germline.csv',
        help='Path for output CSV'
    )
    parser.add_argument(
        '--species',
        type=str,
        nargs='+',
        default=None,
        help='Allowed species for germline assignment (default: all species)'
    )

    args = parser.parse_args()

    # Check ANARCI availability
    if not ANARCI_AVAILABLE:
        print("ERROR: ANARCI is not installed.")
        print("Install with: pip install anarci")
        print("Or run: conda run -n devant pip install anarci")
        sys.exit(1)

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Make paths absolute if relative
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    # Check input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Process
    process_therasabdab(input_path, output_path, allowed_species=args.species)


if __name__ == "__main__":
    main()
