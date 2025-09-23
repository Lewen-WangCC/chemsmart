import logging
import os
import re

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS, rdMolEnumerator

from chemsmart.io.molblockv3k.v3k import MolBlockV3K
from chemsmart.jobs.runner import JobRunner
from chemsmart.utils.periodictable import PeriodicTable

pt = PeriodicTable()

logger = logging.getLogger(__name__)


class IterateJobRunner(JobRunner):
    """JobRunner for molecular iteration using RDKit."""

    JOBTYPES = ["iterate"]
    PROGRAM = "Iterate"
    FAKE = False
    SCRATCH = False

    def __init__(
        self, server, scratch=None, fake=False, scratch_dir=None, **kwargs
    ):
        # Use default SCRATCH if scratch is not explicitly set
        if scratch is None:
            scratch = self.SCRATCH  # default False

        # If user forces scratch, give a warning
        if scratch and not scratch_dir:
            logger.warning(
                "Scratch enabled for IterateJobRunner but no scratch_dir provided. "
                "Iteration typically doesn't benefit from scratch usage."
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
        """IterateJobRunner doesn't need external executable, returns None."""
        return None

    def _prerun(self, job):
        """Prepare for iteration job execution."""
        self._assign_variables(job)

    def _assign_variables(self, job):
        """Set up file paths and directories for iteration output."""
        # Set up file paths and directories
        self.running_directory = job.folder
        logger.debug(f"Running directory: {self.running_directory}")
        self.job_basename = job.label
        self.job_inputfile = os.path.abspath(job.inputfile)
        self.job_outputfile = os.path.abspath(job.outputfile)

    def _write_input(self, job):
        """Prepare input for iteration - convert molecule to RDKit format and
        modify the MOLBlock information according to linknode_specs and position_variation_specs.
        """

        # obtain the molecule from the job
        mol = job.molecule
        if not mol:
            raise ValueError("No molecule provided for iteration job.")

        # obtain linknode and position_variation parameters
        linknode_specs = getattr(job, "linknode_specs", [])
        # Parse command-line LINKNODE specs into structured dicts used by MolBlockV3K.add_linknode
        if isinstance(linknode_specs, (list, tuple)):
            parsed_linknodes = self._parse_linknode_specs(linknode_specs)
            linknode_specs = parsed_linknodes
            self.linknode_specs = parsed_linknodes  # optionally for later use
        position_variation_specs = getattr(job, "position_variation_specs", [])

        logger.info(
            f"Converting molecule to RDKit format for iteration job: {job.label}"
        )
        logger.debug(f"LINKNODE specs: {linknode_specs}")
        logger.debug(f"Position Variation specs: {position_variation_specs}")

        # Check if file content is in V3000 format
        if (
            not linknode_specs
            and not position_variation_specs
            and self.job_inputfile.lower().endswith(".mol")
        ):
            try:
                with open(self.job_inputfile, "r") as f:
                    file_content = f.read()
                if "V3000" in file_content:
                    self.molblock_v3k_str = file_content
                    self.has_modifications = False
                    logger.info(
                        "Directly loaded MOL file content as V3K format"
                    )
                    return  # Directly return, skip further conversion
            except Exception as e:
                logger.warning(
                    f"Failed to directly read MOL file, falling back to normal conversion: {e}"
                )

        # Convert to an RDKit Mol object using the to_rdkit() method of the Molecule object
        try:
            self.rdkit_mol = mol.to_rdkit(
                add_bonds=True, bond_detection="rdkit"
            )

        except Exception as e:
            raise ValueError(
                f"Failed to convert molecule to RDKit format: {e}"
            )

        if self.rdkit_mol is None:
            raise ValueError(
                "Molecule.to_rdkit() returned None. Check molecule validity."
            )

        # Try to generate MOLBlock V3000 format
        try:
            self.molblock_v3k_str = Chem.MolToV3KMolBlock(self.rdkit_mol)
            logger.info("Successfully generated MOLBlock V3000")
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Failed to generate MOLBlock V3000: {e}")

            # Check if this is an aromatic bond issue
            if "aromatic" in error_msg:
                logger.info("Detected aromatic bond issue, applying fix...")
                # Fix aromatic bond marking issue
                fixed_mol = self._fix_aromatic_bonds(self.rdkit_mol)
                self.molblock_v3k_str = Chem.MolToV3KMolBlock(fixed_mol)
                self.rdkit_mol = fixed_mol
                logger.info(
                    "Successfully fixed aromatic bonds and generated MOLBlock V3000"
                )
            else:
                # Other errors are raised directly
                raise ValueError(f"Failed to generate MOLBlock V3000: {e}")

        # Check if modification is needed
        self.has_modifications = bool(
            linknode_specs or position_variation_specs
        )

        # If no modification needed, directly return
        if not self.has_modifications:
            logger.info("No modifications needed, iteration completed")
            return

        # Distinguish between the two formats of position_variation_specs
        pv_format1 = self._split_position_variation_format1(
            position_variation_specs
        )
        pv_format2 = self._split_position_variation_format2(
            position_variation_specs
        )

        self.linknode_specs = linknode_specs
        self.position_variation_specs = position_variation_specs
        self.position_variation_format1 = pv_format1
        self.position_variation_format2 = pv_format2

        # Create MolBlockV3K object for subsequent operations
        molblock_v3k_obj = MolBlockV3K(self.molblock_v3k_str)
        self.molblock_v3k_str = self._apply_change_to_molblock(
            molblock_v3k_obj,
            position_variation_format1=pv_format1,
            position_variation_format2=pv_format2,
            linknode_specs=linknode_specs,
        ).get_molblock()

        logger.info("Successfully converted molecule to RDKit format")

    def _get_command(self, job):
        """Get command for execution - not needed for direct RDKit execution."""
        # IterateJobRunner does not need external command, uses Python/RDKit API directly
        return None

    def _create_process(self, job, command, env):
        """Create process - not needed for direct RDKit execution."""
        # IterateJobRunner does not need to create external process
        return None

    def run(self, job, **kwargs):
        """Execute the iteration job."""
        self._prerun(job)
        self._write_input(job)

        # Directly execute iteration logic, not via external process
        self._execute_iteration(job, **kwargs)

        self._postrun(job)

    def _execute_iteration(self, job, **kwargs):
        """Execute the actual iteration using RDKit."""
        logger.info(f"Starting iteration for job: {job.label}")

        # Check if MOLBlock V3000 is available
        if not hasattr(self, "molblock_v3k_str") or not self.molblock_v3k_str:
            raise ValueError(
                "MOLBlock V3000 not generated. Call _write_input() first."
            )

        # Check if there are modifications (LINKNODE or Position Variation)
        has_modifications = getattr(self, "has_modifications", False)

        # The actual iteration logic will be implemented here
        # Use self.molblock_v3k_str for iteration
        logger.info("Iteration logic will be implemented here")
        logger.info(f"Modifications applied: {has_modifications}")

        print("==" * 20)
        print(self.molblock_v3k_str)
        # exit()
        bundle = self.iterate_from_molblock_v3k(
            Chem.MolFromMolBlock(self.molblock_v3k_str, removeHs=False)
        )
        self.align_bundle_coords(bundle)

        for index, mol in enumerate(bundle):
            logger.info(f"Iterate molecule {index}: {mol}")
            m = Chem.Mol(mol)
            xyz_path = os.path.join(
                job.folder, f"{job.label}_iterate_{index}.xyz"
            )
            try:
                Chem.MolToXYZFile(m, xyz_path)
                logger.info(f"Wrote iterated molecule {index} to {xyz_path}")
            except Exception as e:
                logger.error(
                    f"Failed to write iterated molecule {index} to XYZ file: {e}"
                )

    def _postrun(self, job):
        """Post-processing after iteration completion."""
        if self.scratch:
            # If scratch is used, copy result file to job directory
            logger.info(
                f"Copying output from {self.running_directory} to {job.folder}"
            )

            # Copy output file
            import shutil

            target_outputfile = os.path.join(
                job.folder, os.path.basename(self.job_outputfile)
            )
            shutil.copy2(self.job_outputfile, target_outputfile)

            # Clean up scratch directory
            logger.info(
                f"Cleaning up scratch directory: {self.running_directory}"
            )
            shutil.rmtree(self.running_directory)

        logger.info(f"Iteration job {job.label} completed")

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
                    if not (
                        atom1.GetSymbol() == "C" and atom2.GetSymbol() == "C"
                    ):
                        bonds_to_fix.append(
                            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        )

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

                if (begin_idx, end_idx) in bonds_to_fix or (
                    end_idx,
                    begin_idx,
                ) in bonds_to_fix:
                    # Change incorrect aromatic bond to single bond
                    new_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                    logger.debug(
                        f"Fixed aromatic bond: {begin_idx}-{end_idx} -> SINGLE"
                    )
                else:
                    # Keep original bond type
                    new_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

            # Update property cache
            new_mol.UpdatePropertyCache(strict=False)

            logger.info(
                f"Fixed {len(bonds_to_fix)} incorrectly marked aromatic bonds"
            )
            return new_mol.GetMol()

        except Exception as e:
            logger.warning(
                f"Failed to fix aromatic bonds: {e}, returning original molecule"
            )
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
                    raise ValueError(
                        f"Invalid count in position variation format1: {pv}"
                    )
                digits = comma_parts[1:]
                if len(digits) != count:
                    raise ValueError(
                        f"Count does not match number of digits in position variation format1: {pv}"
                    )
                if len(set(digits)) != len(digits):
                    raise ValueError(
                        f"Digits overlap in position variation format1: {pv}"
                    )
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
                    raise ValueError(
                        f"Invalid count in position variation format2: {pv}"
                    )
                digits = comma_parts[1:]
                if len(digits) != count:
                    raise ValueError(
                        f"Count does not match number of digits in position variation format2: {pv}"
                    )
                if len(set(digits)) != len(digits):
                    raise ValueError(
                        f"Digits overlap in position variation format2: {pv}"
                    )
                pv_format2.append(pv)
        return pv_format2

    def _parse_linknode_specs(self, specs):
        """Parse LINKNODE specs from CLI (list/tuple of strings) into structured dicts.
        Input examples:
            ["1:2:2,20,19,20,21", "1:2:2,1,2,1,6"]
        Output example:
            [{"minrep":1, "maxrep":2, "nbonds":2, "atoms":[20,19,20,21]}, ...]
        """
        parsed = []
        if not specs:
            return parsed
        for s in specs:
            # Expect each s is a string like 'minrep:maxrep:nbonds,in1,out1,in2,out2,...'
            s = s.strip()
            if not s:
                continue
            parts = s.split(":", 2)
            if len(parts) != 3:
                continue
            try:
                minrep = int(parts[0].strip())
                maxrep = int(parts[1].strip())
            except Exception:
                continue
            tail = parts[2].strip()
            if not tail:
                continue
            # tail like '2,20,19,20,21' => nbonds + flattened atoms
            tail_parts = [t for t in re.split(r"[\s,]+", tail) if t]
            if not tail_parts:
                continue
            try:
                nbonds = int(tail_parts[0])
            except Exception:
                continue
            atom_tokens = tail_parts[1:]
            try:
                atoms = [int(x) for x in atom_tokens]
            except Exception:
                continue
            parsed.append(
                {
                    "minrep": minrep,
                    "maxrep": maxrep,
                    "nbonds": nbonds,
                    "atoms": atoms,
                }
            )
        return parsed

    def _apply_change_to_molblock(
        self,
        molblock_v3k_obj,
        position_variation_format1=None,
        position_variation_format2=None,
        linknode_specs=None,
    ):
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
            for ln in linknode_specs:
                # ln is a structured dict: {'minrep','maxrep','nbonds','atoms'}
                molblock_v3k_obj.add_linknode(
                    ln["minrep"], ln["maxrep"], ln["nbonds"], ln["atoms"]
                )

        if position_variation_format2:
            # For each position variation spec in format2, check if the specified virtual atom exists
            virtual_atoms = molblock_v3k_obj.get_virtual_atoms()
            # Build a set of virtual atom indices for fast lookup
            virtual_atom_indices = set(atom["idx"] for atom in virtual_atoms)
            for pv in position_variation_format2:
                parts = pv.split(":")
                if len(parts) == 5:
                    # parts[1] is the virtual atom index (should be int)
                    try:
                        virtual_idx = int(parts[1])
                    except Exception:
                        raise ValueError(
                            f"Invalid virtual atom index in position variation spec: {pv}"
                        )
                    if virtual_idx not in virtual_atom_indices:
                        raise ValueError(
                            f"Virtual atom index {virtual_idx} specified in position variation spec '{pv}' does not exist in molblock."
                        )

                    # Parse bond info
                    bond_type = int(parts[0])
                    bond_atom1 = int(parts[2])
                    bond_atom2 = virtual_idx
                    # Structured fields for ENDPTS and ATTACH
                    endpts_list = [int(x) for x in parts[3].split(",")]
                    attach_type = parts[4]

                    # Check for bond overlap (ignore order)
                    bond_found = False
                    for bond in molblock_v3k_obj.bonds:
                        atoms = {bond["atom1"], bond["atom2"]}
                        if atoms == {bond_atom1, bond_atom2}:
                            # Overlap found, set fields directly
                            molblock_v3k_obj.modify_bond(
                                bond["idx"],
                                endpts=endpts_list,
                                attach=attach_type,
                            )
                            bond_found = True
                            break
                    if not bond_found:
                        # No overlap, create new bond with structured fields
                        molblock_v3k_obj.add_bond(
                            type_=bond_type,
                            atom1=bond_atom1,
                            atom2=bond_atom2,
                            endpts=endpts_list,
                            attach=attach_type,
                        )
        if position_variation_format1:
            for pv in position_variation_format1:
                parts = pv.split(":")
                if len(parts) == 4:
                    bond_type = int(parts[0])
                    group_first_atom = int(parts[1])
                    comma_parts = parts[2].split(",")
                    count = int(comma_parts[0])
                    endpts = [int(x) for x in comma_parts]
                    attach_type = parts[3]

                    # Find all atoms (neighbors) connected to group_first_atom
                    connected_atoms = set()
                    for bond in molblock_v3k_obj.bonds:
                        if bond["atom1"] == group_first_atom:
                            connected_atoms.add(bond["atom2"])
                        elif bond["atom2"] == group_first_atom:
                            connected_atoms.add(bond["atom1"])

                    # Requirement: overlap between ENDPTS and group_first_atom's neighbors must be exactly 1
                    endpts_set = set(endpts)
                    overlap = sorted(
                        a for a in connected_atoms if a in endpts_set
                    )
                    # There must be exactly one atom in ENDPTS range connected to group_first_atom
                    if len(overlap) != 1:
                        raise ValueError(
                            f"Expected exactly one neighbor of atom {group_first_atom} within ENDPTS {endpts} for spec '{pv}', but got {len(overlap)} (overlap={overlap})."
                        )

                    neighbor_idx = overlap[0]
                    # Find atom information
                    src_atom = None
                    for a in molblock_v3k_obj.atoms:
                        if a.get("idx") == neighbor_idx:
                            src_atom = a
                            break
                    if src_atom is None:
                        raise ValueError(
                            f"Internal error: cannot find atom {neighbor_idx} to clone as virtual atom for spec '{pv}'"
                        )
                    # Create a virtual atom with same coordinates/extra attributes (element symbol as '*', index assigned by editor)
                    v_extra = (
                        list(src_atom.get("extra", []))
                        if isinstance(src_atom.get("extra"), list)
                        else src_atom.get("extra")
                    )
                    v_idx = molblock_v3k_obj.add_virtual_atom(
                        src_atom["x"],
                        src_atom["y"],
                        src_atom["z"],
                        extra=v_extra,
                    )
                    logger.debug(
                        f"Created virtual atom {v_idx} cloned from atom {neighbor_idx} (ENDPTS {endpts}) for spec '{pv}'"
                    )

                    # 1) Remove the original bond between group_first_atom and this neighbor
                    removed = False
                    for b in list(molblock_v3k_obj.bonds):
                        if {b["atom1"], b["atom2"]} == {
                            group_first_atom,
                            neighbor_idx,
                        }:
                            molblock_v3k_obj.remove_bond(b["idx"])
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
                    new_bond_idx = molblock_v3k_obj.add_bond(
                        type_=bond_type,
                        atom1=v_idx,
                        atom2=group_first_atom,
                        endpts=endpts,
                        attach=attach_type,
                    )
                    logger.debug(
                        f"Added new bond {new_bond_idx} between {group_first_atom} and virtual atom {v_idx} with ENDPTS={endpts} ATTACH={attach_type} for spec '{pv}'"
                    )

        return molblock_v3k_obj

    @staticmethod
    def align_bundle_coords(bndl):
        # ps = rdFMCS.MCSParameters()
        # for m in bndl:
        #     Chem.SanitizeMol(m)
        # mcs = rdFMCS.FindMCS(bndl,completeRingsOnly=True)
        # q = Chem.MolFromSmarts(mcs.smartsString)
        # rdDepictor.Compute2DCoords(q)
        # for m in bndl:
        #     rdDepictor.GenerateDepictionMatching2DStructure(m,q)

        # Try to create 2D coords without strict sanitization to avoid valence errors
        try:
            mcs = rdFMCS.FindMCS(bndl, completeRingsOnly=True)
            q = Chem.MolFromSmarts(mcs.smartsString)
            rdDepictor.Compute2DCoords(q)
        except Exception:
            q = None

        for m in bndl:
            # Avoid full sanitize: some enumerated structures can violate valence temporarily
            try:
                rdDepictor.Compute2DCoords(m)
            except Exception:
                pass
            if q is not None:
                try:
                    rdDepictor.GenerateDepictionMatching2DStructure(m, q)
                except Exception:
                    # Fallback: leave the molecule with its own 2D coords
                    pass

    @staticmethod
    def iterate_from_molblock_v3k(molblock_v3k_obj):
        return rdMolEnumerator.Enumerate(molblock_v3k_obj)
