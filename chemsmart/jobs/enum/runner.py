import glob
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from contextlib import suppress
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolEnumerator, rdFMCS, rdDepictor

from chemsmart.io.molecules.structure import Molecule
from chemsmart.jobs.runner import JobRunner
from chemsmart.settings.executable import GaussianExecutable
from chemsmart.utils.periodictable import PeriodicTable
from chemsmart.utils.utils import (
    get_prepend_string_list_from_modred_free_format,
    quote_path,
    run_command,
)

pt = PeriodicTable()

logger = logging.getLogger(__name__)


class EnumJobRunner(JobRunner):
    """JobRunner for molecular enumeration using RDKit."""
    
    JOBTYPES = ["enum"]
    PROGRAM = "Enum"
    FAKE = False
    SCRATCH = False
    
    def __init__(self, server, scratch=None, fake=False, scratch_dir=None, **kwargs):
        # Use default SCRATCH if scratch is not explicitly set
        if scratch is None:
            scratch = self.SCRATCH  # default False

        # If user forces scratch, give a warning
        if scratch and not scratch_dir:
            logger.warning(
                "Scratch enabled for EnumJobRunner but no scratch_dir provided. "
                "Enumeration typically doesn't benefit from scratch usage."
            )
        
        super().__init__(
            server=server,
            scratch=scratch,
            scratch_dir=scratch_dir,
            fake=fake,
            **kwargs,
        )
        logger.debug(f"Jobrunner server: {self.server}")
        logger.debug(f"Jobrunner num cores: {self.num_cores}")
        logger.debug(f"Jobrunner num hours: {self.num_hours}")
        logger.debug(f"Jobrunner num gpus: {self.num_gpus}")
        logger.debug(f"Jobrunner mem gb: {self.mem_gb}")
        logger.debug(f"Jobrunner num threads: {self.num_threads}")
        logger.debug(f"Jobrunner scratch: {self.scratch}")

    @property
    def executable(self):
        """EnumJobRunner doesn't need external executable, returns None."""
        return None

    def _prerun(self, job):
        """Prepare for enumeration job execution."""
        self._assign_variables(job)

    def _assign_variables(self, job):
        """Set up file paths and directories for enumeration output."""
        # Set up file paths and directories
        self.running_directory = job.folder
        logger.debug(f"Running directory: {self.running_directory}")
        self.job_basename = job.label
        self.job_inputfile = os.path.abspath(job.inputfile)
        self.job_outputfile = os.path.abspath(job.outputfile)

    def _write_input(self, job):
        """Prepare input for enumeration - convert molecule to RDKit format and 
        modify the MOLBlock information according to linknode_specs and position_variation_specs."""
        
        # obtain the molecule from the job
        mol = job.molecule
        if not mol:
            raise ValueError("No molecule provided for enumeration job.")
        
        # obtain linknode and position_variation parameters
        linknode_specs = getattr(job, 'linknode_specs', [])
        position_variation_specs = getattr(job, 'position_variation_specs', [])
        
        logger.info(f"Converting molecule to RDKit format for enumeration job: {job.label}")
        logger.debug(f"LINKNODE specs: {linknode_specs}")
        logger.debug(f"Position Variation specs: {position_variation_specs}")
        
        # Check if file content is in V3000 format
        if (not linknode_specs and not position_variation_specs and 
            self.job_inputfile.lower().endswith('.mol')):
            try:
                with open(self.job_inputfile, 'r') as f:
                    file_content = f.read()
                if 'V3000' in file_content:
                    self.molblock_v3k = file_content
                    self.has_modifications = False
                    logger.info("Directly loaded MOL file content as V3K format")
                    return  # Directly return, skip further conversion
            except Exception as e:
                logger.warning(f"Failed to directly read MOL file, falling back to normal conversion: {e}")
        
        # Convert to an RDKit Mol object using the to_rdkit() method of the Molecule object
        try:
            self.rdkit_mol = mol.to_rdkit(
                add_bonds=True,
                bond_cutoff_buffer=0.05,
                adjust_H=True
            )
                
        except Exception as e:
            raise ValueError(f"Failed to convert molecule to RDKit format: {e}")
        
        if self.rdkit_mol is None:
            raise ValueError("Molecule.to_rdkit() returned None. Check molecule validity.")
        
        # Try to generate MOLBlock V3000 format
        try:
            self.molblock_v3k = Chem.MolToV3KMolBlock(self.rdkit_mol)
            logger.info("Successfully generated MOLBlock V3000")
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Failed to generate MOLBlock V3000: {e}")

            # Check if this is an aromatic bond issue
            if "aromatic" in error_msg:
                logger.info("Detected aromatic bond issue, applying fix...")
                # Fix aromatic bond marking issue
                fixed_mol = self._fix_aromatic_bonds(self.rdkit_mol)
                self.molblock_v3k = Chem.MolToV3KMolBlock(fixed_mol)
                self.rdkit_mol = fixed_mol  
                logger.info("Successfully fixed aromatic bonds and generated MOLBlock V3000")
            else:
                # Other errors are raised directly
                raise ValueError(f"Failed to generate MOLBlock V3000: {e}")

        # Check if modification is needed
        self.has_modifications = bool(linknode_specs or position_variation_specs)

        # If no modification needed, directly return
        if not self.has_modifications:
            logger.info("No modifications needed, enumeration completed")
            return

        # Distinguish between the two formats of position_variation_specs
        pv_format1 = self._split_position_variation_format1(position_variation_specs)
        pv_format2 = self._split_position_variation_format2(position_variation_specs)

        self.linknode_specs = linknode_specs
        self.position_variation_specs = position_variation_specs
        self.position_variation_format1 = pv_format1
        self.position_variation_format2 = pv_format2

        # Create MolBlockV3K object for subsequent operations
        molblock_v3k_obj = MolBlockV3K(self.molblock_v3k)
        self.molblock_v3k = self._apply_change_to_molblock(
            molblock_v3k_obj,
            position_variation_format1=pv_format1,
            position_variation_format2=pv_format2,
            linknode_specs=linknode_specs,
        ).get_molblock()

        logger.info(f"Successfully converted molecule to RDKit format")

    def _get_command(self, job):
        """Get command for execution - not needed for direct RDKit execution."""
        # EnumJobRunner does not need external command, uses Python/RDKit API directly
        return None

    def _create_process(self, job, command, env):
        """Create process - not needed for direct RDKit execution."""
        # EnumJobRunner does not need to create external process
        return None

    def run(self, job, **kwargs):
        """Execute the enumeration job."""
        self._prerun(job)
        self._write_input(job)
        
        # Directly execute enumeration logic, not via external process
        self._execute_enumeration(job, **kwargs)
        
        self._postrun(job)

    def _execute_enumeration(self, job, **kwargs):
        """Execute the actual enumeration using RDKit."""
        logger.info(f"Starting enumeration for job: {job.label}")
        
        # Check if MOLBlock V3000 is available
        if not hasattr(self, 'molblock_v3k') or not self.molblock_v3k:
            raise ValueError("MOLBlock V3000 not generated. Call _write_input() first.")

        # Check if there are modifications (LINKNODE or Position Variation)
        has_modifications = getattr(self, 'has_modifications', False)

        # The actual enumeration logic will be implemented here
        # Use self.molblock_v3k for enumeration
        logger.info("Enumeration logic will be implemented here")
        logger.info(f"Modifications applied: {has_modifications}")

        # Create output file containing MOLBlock information
        # print("=="*10)
        # print(type(self.molblock_v3k))
        # print(self.molblock_v3k)
        # exit()
        bundle = self.enumerate_from_molblock_v3k(Chem.MolFromMolBlock(self.molblock_v3k))
        self.align_bundle_coords(bundle)

        for index, mol in enumerate(bundle):
            logger.info(f"Enumerated molecule {index}: {mol}")
            m = Chem.Mol(mol)
            xyz_path = os.path.join(job.folder, f"{job.label}_enum_{index}.xyz")
            try:
                Chem.MolToXYZFile(m, xyz_path)
                logger.info(f"Wrote enumerated molecule {index} to {xyz_path}")
            except Exception as e:
                logger.error(f"Failed to write enumerated molecule {index} to XYZ file: {e}")

    def _postrun(self, job):
        """Post-processing after enumeration completion."""
        if self.scratch:
            # If scratch is used, copy result file to job directory
            logger.info(f"Copying output from {self.running_directory} to {job.folder}")

            # Copy output file
            import shutil
            target_outputfile = os.path.join(job.folder, os.path.basename(self.job_outputfile))
            shutil.copy2(self.job_outputfile, target_outputfile)

            # Clean up scratch directory
            logger.info(f"Cleaning up scratch directory: {self.running_directory}")
            shutil.rmtree(self.running_directory)

        logger.info(f"Enumeration job {job.label} completed")

    def _fix_aromatic_bonds(self, rdkit_mol):
        """Fix incorrectly marked aromatic bonds in non-carbon atoms."""
        try:
            # Create an editable copy of the molecule
            editable_mol = Chem.EditableMol(rdkit_mol)

            # Collect bonds that need to be fixed
            bonds_to_fix = []
            for bond in rdkit_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    atom1 = bond.GetBeginAtom()
                    atom2 = bond.GetEndAtom()
                    # If not a C-C bond but marked as aromatic, mark for fixing
                    if not (atom1.GetSymbol() == "C" and atom2.GetSymbol() == "C"):
                        bonds_to_fix.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

            # Since we can't modify bond types directly, use a workaround
            # Create a new molecule and rebuild bonds
            new_mol = Chem.RWMol()

            # Copy all atoms
            for atom in rdkit_mol.GetAtoms():
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_mol.AddAtom(new_atom)

            # Copy conformer information
            if rdkit_mol.GetNumConformers() > 0:
                conf = rdkit_mol.GetConformer(0)
                new_conf = Chem.Conformer(new_mol.GetNumAtoms())
                for i in range(new_mol.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                new_mol.AddConformer(new_conf)

            # Re-add bonds, fixing aromaticity issues
            for bond in rdkit_mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()

                if (begin_idx, end_idx) in bonds_to_fix or (end_idx, begin_idx) in bonds_to_fix:
                    # Change incorrect aromatic bond to single bond
                    new_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                    logger.debug(f"Fixed aromatic bond: {begin_idx}-{end_idx} -> SINGLE")
                else:
                    # Keep original bond type
                    new_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

            # Update property cache
            new_mol.UpdatePropertyCache(strict=False)

            logger.info(f"Fixed {len(bonds_to_fix)} incorrectly marked aromatic bonds")
            return new_mol.GetMol()

        except Exception as e:
            logger.warning(f"Failed to fix aromatic bonds: {e}, returning original molecule")
            return rdkit_mol
    
    def _split_position_variation_format1(self, position_variation_specs):
        """
        Extract position variation specs of format1: bond_type:first_atom:endpt_count,...:attach_type
        Returns a list of valid format1 strings, raises ValueError if any format is invalid.
        Checks that the first number after the comma matches the number of following digits, and that digits do not overlap.
        """
        pv_format1 = []
        for pv in position_variation_specs:
            parts = pv.split(":")
            if len(parts) == 4 and "," in parts[2]:
                # Check comma-separated section
                comma_parts = parts[2].split(",")
                try:
                    count = int(comma_parts[0])
                except Exception:
                    raise ValueError(f"Invalid count in position variation format1: {pv}")
                digits = comma_parts[1:]
                if len(digits) != count:
                    raise ValueError(f"Count does not match number of digits in position variation format1: {pv}")
                if len(set(digits)) != len(digits):
                    raise ValueError(f"Digits overlap in position variation format1: {pv}")
                pv_format1.append(pv)
        return pv_format1

    def _split_position_variation_format2(self, position_variation_specs):
        """
        Extract position variation specs of format2: bond_type:virtual_atom:group_first_atom:endpt_count,...:attach_type
        Returns a list of valid format2 strings, raises ValueError if any format is invalid.
        Checks that the first number after the comma matches the number of following digits, and that digits do not overlap.
        """
        pv_format2 = []
        for pv in position_variation_specs:
            parts = pv.split(":")
            if len(parts) == 5 and "," in parts[3]:
                comma_parts = parts[3].split(",")
                try:
                    count = int(comma_parts[0])
                except Exception:
                    raise ValueError(f"Invalid count in position variation format2: {pv}")
                digits = comma_parts[1:]
                if len(digits) != count:
                    raise ValueError(f"Count does not match number of digits in position variation format2: {pv}")
                if len(set(digits)) != len(digits):
                    raise ValueError(f"Digits overlap in position variation format2: {pv}")
                pv_format2.append(pv)
        return pv_format2

    def _apply_change_to_molblock(self, molblock_v3k_obj, position_variation_format1=None, position_variation_format2=None, linknode_specs=None):
        """
        Apply position variation information to the molblock object.
        This method will modify molblock_v3k_obj according to the command line position variation specs.
        Args:
            molblock_v3k_obj: MolBlockV3K instance to be modified
            position_variation_format1: list of format1 position variation strings
            position_variation_format2: list of format2 position variation strings
            linknode_specs: list of LINKNODE specifications
        """
        if linknode_specs:
            for linknode in linknode_specs:
                # Add LINKNODE to molblock_v3k_obj
                molblock_v3k_obj.add_linknode(linknode)

        if position_variation_format2:
            # For each position variation spec in format2, check if the specified virtual atom exists
            virtual_atoms = molblock_v3k_obj.get_virtual_atoms()
            # Build a set of virtual atom indices for fast lookup
            virtual_atom_indices = set(atom['idx'] for atom in virtual_atoms)
            for pv in position_variation_format2:
                parts = pv.split(":")
                if len(parts) == 5:
                    # parts[1] is the virtual atom index (should be int)
                    try:
                        virtual_idx = int(parts[1])
                    except Exception:
                        raise ValueError(f"Invalid virtual atom index in position variation spec: {pv}")
                    if virtual_idx not in virtual_atom_indices:
                        raise ValueError(f"Virtual atom index {virtual_idx} specified in position variation spec '{pv}' does not exist in molblock.")

                    # Parse bond info
                    bond_type = int(parts[0])
                    bond_atom1 = int(parts[2])
                    bond_atom2 = virtual_idx
                    comma_parts = parts[3].split(",")
                    endpts = " ".join(comma_parts[1:])  # skip the first number
                    extra_info = f"ENDPTS=({endpts}) ATTACH={parts[4]}"

                    # Check for bond overlap (ignore order)
                    bond_found = False
                    for bond in molblock_v3k_obj.bonds:
                        atoms = {bond['atom1'], bond['atom2']}
                        if atoms == {bond_atom1, bond_atom2}:
                            # Overlap found, append extra info
                            bond['extra'] = extra_info
                            bond_found = True
                            break
                    if not bond_found:
                        # No overlap, create new bond
                        molblock_v3k_obj.add_bond(
                            type_=bond_type,
                            atom1=bond_atom1,
                            atom2=bond_atom2,
                            extra=extra_info
                        )
        if position_variation_format1:
            for pv in position_variation_format1:
                parts = pv.split(":")
                if len(parts) == 4:
                    bond_type = int(parts[0])
                    group_first_atom = int(parts[1])
                    comma_parts = parts[2].split(",")
                    count = int(comma_parts[0])
                    endpts = [int(x) for x in comma_parts[1:]]
                    attach_type = parts[3]

                    # Find all atoms (neighbors) connected to group_first_atom
                    connected_atoms = set()
                    for bond in molblock_v3k_obj.bonds:
                        if bond['atom1'] == group_first_atom:
                            connected_atoms.add(bond['atom2'])
                        elif bond['atom2'] == group_first_atom:
                            connected_atoms.add(bond['atom1'])

                    # Requirement: overlap between ENDPTS and group_first_atom's neighbors must be exactly 1
                    endpts_set = set(endpts)
                    overlap = sorted(a for a in connected_atoms if a in endpts_set)
                    # There must be exactly one atom in ENDPTS range connected to group_first_atom
                    if len(overlap) != 1:
                        raise ValueError(
                            f"Expected exactly one neighbor of atom {group_first_atom} within ENDPTS {endpts} for spec '{pv}', but got {len(overlap)} (overlap={overlap})."
                        )

                    neighbor_idx = overlap[0]
                    # Find atom information
                    src_atom = None
                    for a in molblock_v3k_obj.atoms:
                        if a.get('idx') == neighbor_idx:
                            src_atom = a
                            break
                    if src_atom is None:
                        raise ValueError(
                            f"Internal error: cannot find atom {neighbor_idx} to clone as virtual atom for spec '{pv}'"
                        )
                    # Create a virtual atom with same coordinates/extra attributes (element symbol as '*', index assigned by editor)
                    v_extra = list(src_atom.get('extra', [])) if isinstance(src_atom.get('extra'), list) else src_atom.get('extra')
                    v_idx = molblock_v3k_obj.add_virtual_atom(src_atom['x'], src_atom['y'], src_atom['z'], extra=v_extra)
                    logger.debug(
                        f"Created virtual atom {v_idx} cloned from atom {neighbor_idx} (ENDPTS {endpts}) for spec '{pv}'"
                    )

                    # 1) Remove the original bond between group_first_atom and this neighbor
                    removed = False
                    for b in list(molblock_v3k_obj.bonds):
                        if {b['atom1'], b['atom2']} == {group_first_atom, neighbor_idx}:
                            molblock_v3k_obj.remove_bond(b['idx'])
                            removed = True
                            logger.debug(
                                f"Removed original bond {b['idx']} between {group_first_atom} and {neighbor_idx}"
                            )
                            break
                    if not removed:
                        logger.warning(
                            f"No existing bond found between {group_first_atom} and {neighbor_idx} to remove for spec '{pv}'"
                        )

                    # 2) Create a new bond between group_first_atom and the virtual atom, and attach position variation info
                    endpts_str = ' '.join(str(x) for x in endpts)
                    extra_info = f"ENDPTS=({endpts_str}) ATTACH={attach_type}"
                    new_bond_idx = molblock_v3k_obj.add_bond(
                        type_=bond_type,
                        atom1=v_idx,
                        atom2=group_first_atom,
                        extra=extra_info,
                    )
                    logger.debug(
                        f"Added new bond {new_bond_idx} between {group_first_atom} and virtual atom {v_idx} with '{extra_info}' for spec '{pv}'"
                    )

        return molblock_v3k_obj

    @staticmethod
    def align_bundle_coords(bndl):
        ps = rdFMCS.MCSParameters()
        for m in bndl:
            Chem.SanitizeMol(m)
        mcs = rdFMCS.FindMCS(bndl,completeRingsOnly=True)
        q = Chem.MolFromSmarts(mcs.smartsString)
        rdDepictor.Compute2DCoords(q)
        for m in bndl:
            rdDepictor.GenerateDepictionMatching2DStructure(m,q)
    
    @staticmethod
    def enumerate_from_molblock_v3k(molblock_v3k_obj):
        return rdMolEnumerator.Enumerate(molblock_v3k_obj)

class MolBlockV3K:
    """
    A class for parsing and manipulating V3000 molblock content.
    Supports adding, removing, and modifying atoms and bonds,
    as well as outputting the current molblock string.
    """
    def __init__(self, molblock_str: str):
        """
        Initialize MolBlockV3K object and parse the V3000 molblock string.
        Splits header, atoms, bonds, linknodes, and footer sections.
        """
        self.raw = molblock_str
        self.atoms = []   # List of atom information, each item is a dict
        self.bonds = []   # List of bond information, each item is a dict
        self.linknodes = []  # List of LINKNODE information, each item is a dict or raw string
        self.header = []  # Header information (e.g., file header, COUNTS, etc.)
        self.footer = []  # Footer information (e.g., END CTAB, M END)
        self._parse_molblock(molblock_str)

    def _parse_molblock(self, molblock_str):
        """
        Parse the V3000 molblock string and separate header, atom, bond, linknode, and footer sections.
        """
        lines = molblock_str.splitlines()
        in_atom = False
        in_bond = False
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('M  V30 BEGIN ATOM'):
                in_atom = True
                continue
            if line_stripped.startswith('M  V30 END ATOM'):
                in_atom = False
                continue
            if line_stripped.startswith('M  V30 BEGIN BOND'):
                in_bond = True
                continue
            if line_stripped.startswith('M  V30 END BOND'):
                in_bond = False
                continue
            if line_stripped.startswith('M  V30 LINKNODE'):
                linknode = self._parse_linknode_line(line_stripped)
                if linknode:
                    self.linknodes.append(linknode)
                continue
            if in_atom:
                atom = self._parse_atom_line(line)
                if atom:
                    self.atoms.append(atom)
                continue
            if in_bond:
                bond = self._parse_bond_line(line)
                if bond:
                    self.bonds.append(bond)
                continue
            # Header and footer
            if not in_atom and not in_bond:
                if not self.atoms and not self.bonds:
                    self.header.append(line)
                else:
                    self.footer.append(line)

    def _parse_linknode_line(self, line):
        """
        Parse a LINKNODE line and return a dictionary of linknode properties.
        Example: M  V30 LINKNODE 1 4 2 1 2 1 5
        """
        parts = line.split()
        if len(parts) < 4 or parts[0] != 'M' or parts[1] != 'V30' or parts[2] != 'LINKNODE':
            return None
        # Simple parsing, store all content as list
        return {'raw': line, 'values': parts[3:]}

    def get_linknodes(self):
        """
        Return a list of all LINKNODE information in the molblock.
        Each item is a dict with raw line and parsed values.
        """
        return self.linknodes.copy()

    def add_linknode(self, values):
        """
        Add a LINKNODE entry to the molblock.
        Args:
            values: list of values (e.g., ['1', '4', '2', '1', '2', '1', '5']) or a string
        Returns:
            The raw LINKNODE line added.
        """
        # Support both string and list input
        if isinstance(values, str):
            # Split by regex and join with single space, remove extra spaces
            values = [x.strip() for x in re.split(r'[:,\s]+', values) if x.strip()]
        # Ensure all elements are strings
        values = [str(v) for v in values]
        # Always use exact prefix with two spaces: 'M  V30 LINKNODE'
        line = 'M  V30 LINKNODE ' + ' '.join(values)
        # Ensure only single space between numbers (but keep two spaces after M)
        # Remove any accidental extra spaces after prefix
        line = re.sub(r'^(M)\s{2,}(V30 LINKNODE)', r'M  V30 LINKNODE', line)
        # Remove extra spaces after prefix (if any)
        line = re.sub(r'^(M  V30 LINKNODE)\s+', r'M  V30 LINKNODE ', line)
        # Ensure only single space between numbers (after prefix)
        prefix = 'M  V30 LINKNODE '
        if line.startswith(prefix):
            rest = line[len(prefix):]
            rest = re.sub(r'\s+', ' ', rest)
            line = prefix + rest
        self.linknodes.append({'raw': line, 'values': values})
        return line

    def remove_linknode(self, idx):
        """
        Remove a LINKNODE entry by its index in the linknodes list.
        Args:
            idx: index of the LINKNODE to remove
        """
        if 0 <= idx < len(self.linknodes):
            del self.linknodes[idx]

    def _parse_atom_line(self, line):
        """
        Parse an atom line and return a dictionary of atom properties.
        Example: M  V30 1 C -1.7083 2.415 0 0
        """
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'M' or parts[1] != 'V30':
            return None
        return {
            'idx': int(parts[2]),
            'element': parts[3],
            'x': float(parts[4]),
            'y': float(parts[5]),
            'z': float(parts[6]),
            'extra': parts[7:] if len(parts) > 7 else []
        }

    def _parse_bond_line(self, line):
        """
        Parse a bond line and return a dictionary of bond properties.
        Example: M  V30 1 1 1 2
        """
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'M' or parts[1] != 'V30':
            return None
        # Handle optional attributes (e.g., ENDPTS/ATTACH)
        extra = ' '.join(parts[6:]) if len(parts) > 6 else ''
        return {
            'idx': int(parts[2]),
            'type': int(parts[3]),
            'atom1': int(parts[4]),
            'atom2': int(parts[5]),
            'extra': extra
        }

    # --- Operation interfaces ---
    def add_atom(self, element, x, y, z, extra=None):
        """
        Add a new atom to the molblock.
        Returns the index of the new atom.

        Example:
            idx = molblock_obj.add_atom("C", 0.0, 0.0, 0.0)
        """
        idx = len(self.atoms) + 1
        atom = {
            'idx': idx,
            'element': element,
            'x': x,
            'y': y,
            'z': z,
            'extra': extra or []
        }
        self.atoms.append(atom)
        return idx

    def remove_atom(self, idx):
        """
        Remove an atom by its index and also remove any bonds associated with it.

        Example:
            molblock_obj.remove_atom(idx)
        """
        self.atoms = [a for a in self.atoms if a['idx'] != idx]
        self.bonds = [b for b in self.bonds if b['atom1'] != idx and b['atom2'] != idx]

    def modify_atom(self, idx, **kwargs):
        """
        Modify properties of an atom by its index.

        Example:
            molblock_obj.modify_atom(idx, x=1.0, y=2.0)
        """
        for atom in self.atoms:
            if atom['idx'] == idx:
                atom.update(kwargs)
                break

    def add_bond(self, type_, atom1, atom2, extra=''):
        """
        Add a new bond to the molblock.
        Returns the index of the new bond.

        Example:
            b_idx = molblock_obj.add_bond(1, 1, 2)
        """
        idx = len(self.bonds) + 1
        bond = {
            'idx': idx,
            'type': type_,
            'atom1': atom1,
            'atom2': atom2,
            'extra': extra
        }
        self.bonds.append(bond)
        return idx

    def add_virtual_atom(self, x, y, z, extra=None):
        """
        Add a virtual atom (element='*') to the molblock.
        Returns the index of the new virtual atom.

        Example:
            v_idx = molblock_obj.add_virtual_atom(1.0, 2.0, 0.0)
        """
        return self.add_atom("*", x, y, z, extra)

    def remove_bond(self, idx):
        """
        Remove a bond by its index.

        Example:
            molblock_obj.remove_bond(b_idx)
        """
        self.bonds = [b for b in self.bonds if b['idx'] != idx]

    def modify_bond(self, idx, **kwargs):
        """
        Modify properties of a bond by its index.

        Example:
            molblock_obj.modify_bond(b_idx, type=2)
        """
        for bond in self.bonds:
            if bond['idx'] == idx:
                bond.update(kwargs)
                break

    def get_molblock(self):
        """
        Output the current molblock string with all atoms, bonds, and linknodes.

        Example:
            molblock_str = molblock_obj.get_molblock()
        """
        lines = []
        lines.extend(self.header)
        lines.append('M  V30 BEGIN ATOM')
        for atom in self.atoms:
            atom_line = f"M  V30 {atom['idx']} {atom['element']} {atom['x']} {atom['y']} {atom['z']}"
            if atom['extra']:
                atom_line += ' ' + ' '.join(str(e) for e in atom['extra'])
            lines.append(atom_line)
        lines.append('M  V30 END ATOM')
        lines.append('M  V30 BEGIN BOND')
        for bond in self.bonds:
            bond_line = f"M  V30 {bond['idx']} {bond['type']} {bond['atom1']} {bond['atom2']}"
            if bond['extra']:
                bond_line += f" {bond['extra']}"
            lines.append(bond_line)
        lines.append('M  V30 END BOND')
        for linknode in self.linknodes:
            lines.append(linknode['raw'])
        lines.extend(self.footer)
        return '\n'.join(lines)

    def get_first_atom(self):
        """
        Return the information of the first atom in the molblock.
        Returns None if no atom exists.
        """
        if self.atoms:
            return self.atoms[0]
        return None

    def get_last_atom(self):
        """
        Return the information of the last atom in the molblock.
        Returns None if no atom exists.
        """
        if self.atoms:
            return self.atoms[-1]
        return None

    def get_virtual_atoms(self):
        """
        Return a list of all virtual atoms (element == '*') in the molblock.
        Each item is a dict with atom information.
        """
        return [atom for atom in self.atoms if atom.get('element') == '*']
