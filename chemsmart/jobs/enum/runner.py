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
from rdkit.Chem import rdMolDescriptors

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
            scratch = self.SCRATCH  # 默认 False
        
        # 如果用户强制启用 scratch，给出提示
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
        
        # 检查文件内容是否为 V3000 格式
        if (not linknode_specs and not position_variation_specs and 
            self.job_inputfile.lower().endswith('.mol')):
            try:
                with open(self.job_inputfile, 'r') as f:
                    file_content = f.read()
                if 'V3000' in file_content:
                    self.molblock_v3k = file_content
                    self.has_modifications = False
                    logger.info("Directly loaded MOL file content as V3K format")
                    return  # 直接返回，跳过后续转换
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
        
        # 尝试生成 MOLBlock V3000 格式
        try:
            self.molblock_v3k = Chem.MolToV3KMolBlock(self.rdkit_mol)
            logger.info("Successfully generated MOLBlock V3000")
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Failed to generate MOLBlock V3000: {e}")
            
            # 检查是否为芳香性键问题
            if "aromatic" in error_msg:
                logger.info("Detected aromatic bond issue, applying fix...")
                # 修复芳香性标记问题
                fixed_mol = self._fix_aromatic_bonds(self.rdkit_mol)
                self.molblock_v3k = Chem.MolToV3KMolBlock(fixed_mol)
                self.rdkit_mol = fixed_mol  
                logger.info("Successfully fixed aromatic bonds and generated MOLBlock V3000")
            else:
                # 其他类型的错误直接抛出
                raise ValueError(f"Failed to generate MOLBlock V3000: {e}")
        
        # 检查是否有修改需求
        self.has_modifications = bool(linknode_specs or position_variation_specs)
        
        # 如果没有任何修改需求，直接返回
        if not self.has_modifications:
            logger.info("No modifications needed, enumeration completed")
            return
        
        # 区分 position_variation_specs 的两种格式
        pv_format1 = self._split_position_variation_format1(position_variation_specs)
        pv_format2 = self._split_position_variation_format2(position_variation_specs)

        self.linknode_specs = linknode_specs
        self.position_variation_specs = position_variation_specs
        self.position_variation_format1 = pv_format1
        self.position_variation_format2 = pv_format2

        # 创建 MolBlockV3K 对象，方便后续操作
        molblock_v3k_obj = MolBlockV3K(self.molblock_v3k)

        logger.info(f"Successfully converted molecule to RDKit format")
        logger.debug(f"RDKit molecule atoms: {self.rdkit_mol.GetNumAtoms()}")
        logger.debug(f"RDKit molecule bonds: {self.rdkit_mol.GetNumBonds()}")
        logger.debug(f"MOLBlock V3000 size: {len(self.molblock_v3k)} characters")
        logger.debug(f"Has modifications: {self.has_modifications}")
        logger.debug(f"Position Variation Format1: {pv_format1}")
        logger.debug(f"Position Variation Format2: {pv_format2}")

    def _get_command(self, job):
        """Get command for execution - not needed for direct RDKit execution."""
        # EnumJobRunner 不需要外部命令，直接使用 Python/RDKit API
        return None

    def _create_process(self, job, command, env):
        """Create process - not needed for direct RDKit execution."""
        # EnumJobRunner 不需要创建外部进程
        return None

    def run(self, job, **kwargs):
        """Execute the enumeration job."""
        self._prerun(job)
        self._write_input(job)
        
        # 直接执行枚举逻辑，而不是通过外部进程
        self._execute_enumeration(job, **kwargs)
        
        self._postrun(job)

    def _execute_enumeration(self, job, **kwargs):
        """Execute the actual enumeration using RDKit."""
        logger.info(f"Starting enumeration for job: {job.label}")
        
        # 检查是否有可用的 MOLBlock 和 RDKit 分子
        if not hasattr(self, 'molblock_v3k') or not self.molblock_v3k:
            raise ValueError("MOLBlock V3000 not generated. Call _write_input() first.")
        
        if not hasattr(self, 'rdkit_mol') or not self.rdkit_mol:
            raise ValueError("RDKit molecule not available. Call _write_input() first.")
        
        logger.info(f"Using MOLBlock V3000 ({len(self.molblock_v3k)} chars) and RDKit molecule")
        logger.debug(f"RDKit molecule: {self.rdkit_mol}")
        
        # 检查是否有修改（LINKNODE 或 Position Variation）
        has_modifications = getattr(self, 'has_modifications', False)
        
        # 这里将实现实际的枚举逻辑
        # 使用 self.molblock_v3k 和 self.rdkit_mol 进行 RDKit 枚举操作
        logger.info("Enumeration logic will be implemented here")
        logger.info(f"Modifications applied: {has_modifications}")
        
        # 创建输出文件，包含 MOLBlock 信息
        with open(self.job_outputfile, 'w') as f:
            f.write("# Enumerated structures will be written here\n")
            f.write(f"# Job: {job.label}\n")
            f.write(f"# Input molecule: {job.molecule.get_chemical_formula()}\n")
            f.write(f"# RDKit molecule atoms: {self.rdkit_mol.GetNumAtoms()}\n")
            f.write(f"# RDKit molecule bonds: {self.rdkit_mol.GetNumBonds()}\n")
            f.write(f"# MOLBlock V3000 size: {len(self.molblock_v3k)} characters\n")
            f.write(f"# Has modifications: {has_modifications}\n")
            f.write(f"# LINKNODE specs: {getattr(self, 'linknode_specs', [])}\n")
            f.write(f"# Position variation specs: {getattr(self, 'position_variation_specs', [])}\n")
            f.write("\n")
            f.write("# Generated MOLBlock V3000:\n")
            # 写入 MOLBlock 的前几行作为示例
            molblock_lines = self.molblock_v3k.split('\n')
            for i, line in enumerate(molblock_lines[:10]):  # 只显示前10行
                f.write(f"# {line}\n")
            if len(molblock_lines) > 10:
                f.write(f"# ... ({len(molblock_lines)-10} more lines)\n")

    def _postrun(self, job):
        """Post-processing after enumeration completion."""
        if self.scratch:
            # 如果使用了 scratch，复制结果文件到作业目录
            logger.info(f"Copying output from {self.running_directory} to {job.folder}")
            
            # 复制输出文件
            import shutil
            target_outputfile = os.path.join(job.folder, os.path.basename(self.job_outputfile))
            shutil.copy2(self.job_outputfile, target_outputfile)
            
            # 清理 scratch 目录
            logger.info(f"Cleaning up scratch directory: {self.running_directory}")
            shutil.rmtree(self.running_directory)
        
        logger.info(f"Enumeration job {job.label} completed")

    def _fix_aromatic_bonds(self, rdkit_mol):
        """Fix incorrectly marked aromatic bonds in non-carbon atoms."""
        try:
            # 创建可编辑的分子副本
            editable_mol = Chem.EditableMol(rdkit_mol)
            
            # 收集需要修复的键
            bonds_to_fix = []
            for bond in rdkit_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    atom1 = bond.GetBeginAtom()
                    atom2 = bond.GetEndAtom()
                    # 如果不是 C-C 键但被标记为芳香键，标记为需要修复
                    if not (atom1.GetSymbol() == "C" and atom2.GetSymbol() == "C"):
                        bonds_to_fix.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
            # 由于我们不能直接修改键类型，使用替代方法
            # 创建一个新的分子并重建键
            new_mol = Chem.RWMol()
            
            # 复制所有原子
            for atom in rdkit_mol.GetAtoms():
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_mol.AddAtom(new_atom)
            
            # 复制构象信息
            if rdkit_mol.GetNumConformers() > 0:
                conf = rdkit_mol.GetConformer(0)
                new_conf = Chem.Conformer(new_mol.GetNumAtoms())
                for i in range(new_mol.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                new_mol.AddConformer(new_conf)
            
            # 重新添加键，修复芳香性问题
            for bond in rdkit_mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                if (begin_idx, end_idx) in bonds_to_fix or (end_idx, begin_idx) in bonds_to_fix:
                    # 将错误的芳香键改为单键
                    new_mol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                    logger.debug(f"Fixed aromatic bond: {begin_idx}-{end_idx} -> SINGLE")
                else:
                    # 保持原有键类型
                    new_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
            
            # 更新属性缓存
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
                # 检查逗号部分
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
    
    def _apply_change_to_molblock(self, molblock_v3k_obj, position_variation_format1, position_variation_format2):
        """
        Apply position variation information to the molblock object.
        This method will modify molblock_v3k_obj according to the command line position variation specs.
        Args:
            molblock_v3k_obj: MolBlockV3K instance to be modified
            position_variation_format1: list of format1 position variation strings
            position_variation_format2: list of format2 position variation strings
        """
        
        pass

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
        # 只做简单解析，全部内容存为list
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
            values: list of values (e.g., ['1', '4', '2', '1', '2', '1', '5'])
        Returns:
            The raw LINKNODE line added.
        """
        line = 'M  V30 LINKNODE ' + ' '.join(str(v) for v in values)
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
        # 输出所有LINKNODE
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


if __name__ == "__main__":
    # 示例 V3000 molblock 字符串
    molblock_str = '''
  Mrv2108 05132113572D          

  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 13 13 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C 1.2124 -2.4845 0 0
M  V30 2 N 2.5461 -3.2545 0 0
M  V30 3 C 2.5461 -4.7945 0 0
M  V30 4 C 1.2124 -5.5645 0 0
M  V30 5 C 1.2124 -7.1045 0 0
M  V30 6 C -0.0335 -8.0097 0 0
M  V30 7 O 0.4424 -9.4744 0 0
M  V30 8 C 1.9824 -9.4744 0 0
M  V30 9 C 2.4583 -8.0097 0 0
M  V30 10 C -0.1212 -4.7945 0 0
M  V30 11 C -0.1212 -3.2545 0 0
M  V30 12 * 0.5456 -2.8695 0 0
M  V30 13 C -0.6094 -0.869 0 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 2 1 2
M  V30 2 1 2 3
M  V30 3 2 3 4
M  V30 4 1 4 5
M  V30 5 1 6 7
M  V30 6 1 7 8
M  V30 7 1 8 9
M  V30 8 1 5 9
M  V30 9 1 4 10
M  V30 10 2 10 11
M  V30 11 1 1 11
M  V30 12 1 12 13 ENDPTS=(2 11 1) ATTACH=ANY
M  V30 13 1 5 6
M  V30 END BOND
M  V30 LINKNODE 1 2 2 6 5 6 7
M  V30 END CTAB
M  END
'''
    # 创建对象
    molblock_obj = MolBlockV3K(molblock_str)

    print("--- Atoms ---")
    for atom in molblock_obj.atoms:
        print(atom)
    print("\n--- Bonds ---")
    for bond in molblock_obj.bonds:
        print(bond)
    print("\n--- LINKNODEs ---")
    for linknode in molblock_obj.get_linknodes():
        print(linknode)

    print("\nFirst atom:", molblock_obj.get_first_atom())
    print("Last atom:", molblock_obj.get_last_atom())
    print("Virtual atoms:", molblock_obj.get_virtual_atoms())

    print("\nAdd atom H:")
    idx_h = molblock_obj.add_atom("H", 1.0, 2.0, 0.0)
    print("Added atom index:", idx_h)
    print("Last atom after add:", molblock_obj.get_last_atom())

    print("\nAdd bond between atom 1 and new H:")
    idx_bond = molblock_obj.add_bond(1, 1, idx_h)
    print("Added bond index:", idx_bond)
    print("Last bond after add:", molblock_obj.bonds[-1])

    print("\nAdd virtual atom X:")
    idx_x = molblock_obj.add_atom("X", 2.0, 2.0, 0.0)
    print("Added atom index:", idx_x)
    print("Virtual atoms after add:", molblock_obj.get_virtual_atoms())

    print("\nAdd LINKNODE entry:")
    new_linknode_line = molblock_obj.add_linknode(["2", "3", "2", "1", "2", "1", "4"])
    print("Added LINKNODE:", new_linknode_line)
    print("All LINKNODEs:", molblock_obj.get_linknodes())

    print("\nRemove LINKNODE entry at index 0:")
    molblock_obj.remove_linknode(0)
    print("All LINKNODEs after remove:", molblock_obj.get_linknodes())

    print("\nCurrent molblock string:")
    print(molblock_obj.get_molblock())