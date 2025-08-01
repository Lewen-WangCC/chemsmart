import logging
import tempfile
import os
from typing import Optional, Dict, List, Any

from chemsmart.jobs.writer import InputWriter

logger = logging.getLogger(__name__)


class EnumWriter(InputWriter):
    """Writer class that converts molecules to RDKit MOLBlock format for enumeration."""

    def __init__(self, job, **kwargs):
        super().__init__(job)
        # EnumJob 没有 settings，所以重写初始化
        self.settings = None  # EnumJob 不使用 settings
        self.molblock_v3000 = None
        self.rdkit_mol = None
        self.linknode_specs = kwargs.get('linknode', [])
        self.position_variation_specs = kwargs.get('position_variation', [])

    def write(self, **kwargs):
        """Generate or modify MOLBlock in memory instead of writing to file."""
        self._ensure_molblock_available(**kwargs)
        return self.molblock_v3000

    def _ensure_molblock_available(self, **kwargs):
        """Ensure MOLBlock is available, generate if needed, then apply modifications."""
        # Step 1: Generate initial MOLBlock if not already available
        if self.molblock_v3000 is None:
            logger.debug("MOLBlock not available, generating initial MOLBlock")
            self._generate_initial_molblock(**kwargs)
        else:
            logger.debug("MOLBlock already available, proceeding with modifications")
        
        # Step 2: Apply any linknode or position variation modifications
        self._apply_modifications()
        
        logger.info(f"MOLBlock ready for molecule {self.job.label}")
        logger.debug(f"MOLBlock preview:\n{self.molblock_v3000[:200]}...")

    def _generate_initial_molblock(self, **kwargs):
        """Generate the initial MOLBlock from chemsmart Molecule."""
        logger.debug(f"Converting molecule {self.job.molecule} to MOLBlock V3000 format")
        
        # Step 1: Convert chemsmart Molecule to RDKit Mol
        if self.rdkit_mol is None:
            self.rdkit_mol = self._convert_to_rdkit_mol()
        
        # Step 2: Generate initial MOLBlock V3000
        self.molblock_v3000 = self._generate_molblock_v3000()
        
        logger.info(f"Generated initial MOLBlock V3000 for molecule {self.job.label}")

    def _apply_modifications(self):
        """Apply all modifications to the existing MOLBlock."""
        # Apply LINKNODE modifications
        if self.linknode_specs:
            logger.debug(f"Applying LINKNODE modifications: {self.linknode_specs}")
            self.molblock_v3000 = self._apply_linknode_modifications()
        
        # Apply Position Variation modifications  
        if self.position_variation_specs:
            logger.debug(f"Applying Position Variation modifications: {self.position_variation_specs}")
            self.molblock_v3000 = self._apply_position_variation_modifications()
        
        if self.has_modifications():
            logger.info("Applied all modifications to MOLBlock")
        else:
            logger.debug("No modifications to apply")

    def _convert_to_rdkit_mol(self):
        """Convert chemsmart Molecule to RDKit Mol object."""
        from rdkit import Chem
        
        try:
            # Method 1: Try SMILES if available
            if hasattr(self.job.molecule, 'smiles') and self.job.molecule.smiles:
                rdkit_mol = Chem.MolFromSmiles(self.job.molecule.smiles)
                if rdkit_mol is not None:
                    # Add 3D coordinates if available
                    if hasattr(self.job.molecule, 'positions') and self.job.molecule.positions is not None:
                        rdkit_mol = self._add_coordinates_to_mol(rdkit_mol)
                    return rdkit_mol
            
            # Method 2: Use XYZ coordinates to build molecule
            rdkit_mol = self._build_mol_from_coordinates()
            
            if rdkit_mol is None:
                raise ValueError("Failed to create RDKit molecule")
                
            return rdkit_mol
            
        except Exception as e:
            logger.error(f"Failed to convert molecule to RDKit: {e}")
            raise ValueError(f"Could not convert molecule {self.job.molecule} to RDKit format: {e}")

    def _build_mol_from_coordinates(self):
        """Build RDKit Mol from atomic coordinates and symbols."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        # Create a temporary XYZ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_xyz = f.name
            self.job.molecule.write_xyz(temp_xyz)
        
        try:
            # Try to read XYZ file with RDKit
            # Note: RDKit's XYZ reader is limited, so we might need to use other methods
            
            # Alternative: Build molecule manually from symbols and positions
            mol = Chem.RWMol()
            
            # Add atoms
            for symbol in self.job.molecule.chemical_symbols:
                atom = Chem.Atom(symbol)
                mol.AddAtom(atom)
            
            # Try to infer bonds (this is a simplified approach)
            # In practice, you might want to use more sophisticated bond perception
            mol = mol.GetMol()
            mol = Chem.AddHs(mol)
            
            # Add 3D coordinates
            if hasattr(self.job.molecule, 'positions') and self.job.molecule.positions is not None:
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i, pos in enumerate(self.job.molecule.positions):
                    conf.SetAtomPosition(i, pos)
                mol.AddConformer(conf)
            
            return mol
            
        finally:
            # Clean up temporary file
            os.unlink(temp_xyz)

    def _add_coordinates_to_mol(self, mol):
        """Add 3D coordinates to an RDKit molecule."""
        from rdkit import Chem
        
        if not hasattr(self.job.molecule, 'positions') or self.job.molecule.positions is None:
            return mol
        
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, pos in enumerate(self.job.molecule.positions):
            if i < mol.GetNumAtoms():  # Safety check
                conf.SetAtomPosition(i, pos)
        
        mol.AddConformer(conf)
        return mol

    def _generate_molblock_v3000(self):
        """Generate MOLBlock in V3000 format."""
        from rdkit import Chem
        
        if self.rdkit_mol is None:
            raise ValueError("RDKit molecule not available")
        
        try:
            # Generate MOLBlock V3000 format
            molblock = Chem.MolToV3000MolBlock(self.rdkit_mol)
            logger.debug(f"Generated MOLBlock V3000:\n{molblock}")
            return molblock
            
        except Exception as e:
            logger.error(f"Failed to generate MOLBlock V3000: {e}")
            # Fallback to V2000 if V3000 fails
            try:
                molblock = Chem.MolToMolBlock(self.rdkit_mol)
                logger.warning("Using MOLBlock V2000 format as fallback")
                return molblock
            except Exception as e2:
                raise ValueError(f"Failed to generate MOLBlock: {e2}")

    def _apply_linknode_modifications(self):
        """Apply LINKNODE modifications to the MOLBlock."""
        logger.debug(f"Applying LINKNODE modifications: {self.linknode_specs}")
        
        modified_molblock = self.molblock_v3000
        
        for linknode_spec in self.linknode_specs:
            # Parse linknode specification: 'node_id:atom_num:definition_of_link'
            # Example: '1:4:2,1,2,1,5'
            modified_molblock = self._process_single_linknode(modified_molblock, linknode_spec)
        
        return modified_molblock

    def _process_single_linknode(self, molblock: str, linknode_spec: str) -> str:
        """Process a single LINKNODE specification."""
        try:
            parts = linknode_spec.split(':')
            if len(parts) != 3:
                logger.warning(f"Invalid LINKNODE format: {linknode_spec}")
                return molblock
            
            node_id, atom_num, link_def = parts
            logger.debug(f"Processing LINKNODE - Node: {node_id}, Atom: {atom_num}, Link: {link_def}")
            
            # TODO: Implement actual LINKNODE modification logic
            # This would involve parsing the MOLBlock and modifying specific sections
            # For now, we'll add a comment to track the modification
            lines = molblock.split('\n')
            header_line = f"# LINKNODE: {linknode_spec}"
            
            # Insert after the first line (typically the molecule name line)
            if len(lines) > 1:
                lines.insert(1, header_line)
            else:
                lines.append(header_line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Failed to process LINKNODE {linknode_spec}: {e}")
            return molblock

    def _apply_position_variation_modifications(self):
        """Apply Position Variation modifications to the MOLBlock."""
        logger.debug(f"Applying Position Variation modifications: {self.position_variation_specs}")
        
        modified_molblock = self.molblock_v3000
        
        for pv_spec in self.position_variation_specs:
            # Parse position variation: 'bond_type:first_atom:endpt_count,endpt1,endpt2,...:attach_type'
            # Example: '1:8:3,1,5,6:ANY'
            modified_molblock = self._process_single_position_variation(modified_molblock, pv_spec)
        
        return modified_molblock

    def _process_single_position_variation(self, molblock: str, pv_spec: str) -> str:
        """Process a single Position Variation specification."""
        try:
            parts = pv_spec.split(':')
            if len(parts) != 4:
                logger.warning(f"Invalid Position Variation format: {pv_spec}")
                return molblock
            
            bond_type, first_atom, endpoints, attach_type = parts
            logger.debug(f"Processing Position Variation - Bond: {bond_type}, First: {first_atom}, "
                        f"Endpoints: {endpoints}, Attach: {attach_type}")
            
            # TODO: Implement actual Position Variation modification logic
            # For now, we'll add a comment to track the modification
            lines = molblock.split('\n')
            header_line = f"# POSITION_VARIATION: {pv_spec}"
            
            # Insert after any existing LINKNODE comments
            insert_pos = 1
            for i, line in enumerate(lines[1:], 1):
                if line.startswith('# LINKNODE:'):
                    insert_pos = i + 1
                elif line.startswith('#'):
                    continue
                else:
                    break
            
            lines.insert(insert_pos, header_line)
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Failed to process Position Variation {pv_spec}: {e}")
            return molblock

    def get_molblock(self) -> Optional[str]:
        """Get the generated MOLBlock."""
        return self.molblock_v3000

    def get_rdkit_mol(self):
        """Get the RDKit Mol object."""
        return self.rdkit_mol

    def has_modifications(self) -> bool:
        """Check if any modifications (LINKNODE or Position Variation) are applied."""
        return bool(self.linknode_specs or self.position_variation_specs)

    def is_molblock_ready(self) -> bool:
        """Check if MOLBlock has been generated."""
        return self.molblock_v3000 is not None

    def add_linknode_spec(self, linknode_spec: str):
        """Add a new LINKNODE specification and apply it to existing MOLBlock if available."""
        if linknode_spec not in self.linknode_specs:
            self.linknode_specs.append(linknode_spec)
            logger.debug(f"Added LINKNODE spec: {linknode_spec}")
            
            # If MOLBlock already exists, apply the new modification immediately
            if self.molblock_v3000 is not None:
                self.molblock_v3000 = self._process_single_linknode(self.molblock_v3000, linknode_spec)
                logger.info(f"Applied new LINKNODE modification to existing MOLBlock")

    def add_position_variation_spec(self, pv_spec: str):
        """Add a new Position Variation specification and apply it to existing MOLBlock if available."""
        if pv_spec not in self.position_variation_specs:
            self.position_variation_specs.append(pv_spec)
            logger.debug(f"Added Position Variation spec: {pv_spec}")
            
            # If MOLBlock already exists, apply the new modification immediately
            if self.molblock_v3000 is not None:
                self.molblock_v3000 = self._process_single_position_variation(self.molblock_v3000, pv_spec)
                logger.info(f"Applied new Position Variation modification to existing MOLBlock")

    def clear_modifications(self):
        """Clear all modifications and regenerate clean MOLBlock."""
        self.linknode_specs.clear()
        self.position_variation_specs.clear()
        
        # If we have the base MOLBlock, regenerate it clean
        if self.rdkit_mol is not None:
            self.molblock_v3000 = self._generate_molblock_v3000()
            logger.info("Cleared all modifications and regenerated clean MOLBlock")

    def regenerate_molblock(self, **kwargs):
        """Force regeneration of MOLBlock from scratch."""
        self.molblock_v3000 = None
        self.rdkit_mol = None
        self._ensure_molblock_available(**kwargs)
        logger.info("Regenerated MOLBlock from scratch")
