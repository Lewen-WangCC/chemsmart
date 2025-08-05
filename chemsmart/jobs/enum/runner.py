import glob
import logging
import os
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
        
        # 存储枚举参数以供后续使用
        self.linknode_specs = linknode_specs
        self.position_variation_specs = position_variation_specs
        
        logger.info(f"Successfully converted molecule to RDKit format")
        logger.debug(f"RDKit molecule atoms: {self.rdkit_mol.GetNumAtoms()}")
        logger.debug(f"RDKit molecule bonds: {self.rdkit_mol.GetNumBonds()}")
        logger.debug(f"MOLBlock V3000 size: {len(self.molblock_v3k)} characters")
        logger.debug(f"Has modifications: {self.has_modifications}")

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