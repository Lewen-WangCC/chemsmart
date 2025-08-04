import glob
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from contextlib import suppress

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
    
    JOBTYPES = []
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
        # 设置运行目录为作业目录
        self.running_directory = job.folder
        logger.debug(f"Running directory: {self.running_directory}")
        self.job_basename = job.label
        self.job_inputfile = os.path.abspath(job.inputfile)
        self.job_outputfile = os.path.abspath(job.outputfile)

    def _write_input(self, job):
        """Prepare input for enumeration - convert molecule to RDKit format."""
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            raise ImportError("RDKit is required for enumeration jobs. Please install RDKit.")
        
        # 获取分子对象
        mol = job.molecule
        if not mol:
            raise ValueError("No molecule provided for enumeration job.")
        
        # 获取 linknode 和 position_variation 参数
        linknode_specs = getattr(job, 'linknode_specs', [])
        position_variation_specs = getattr(job, 'position_variation_specs', [])
        
        logger.info(f"Converting molecule to RDKit format for enumeration job: {job.label}")
        logger.debug(f"LINKNODE specs: {linknode_specs}")
        logger.debug(f"Position Variation specs: {position_variation_specs}")
        
        # 将 Molecule 对象转换为 RDKit Mol 对象
        # 这里需要根据 Molecule 对象的具体结构来实现转换
        if hasattr(mol, 'to_rdkit'):
            # 如果 Molecule 对象有 to_rdkit 方法
            self.rdkit_mol = mol.to_rdkit()
        else:
            # 否则通过 MOL 格式字符串转换
            mol_string = mol.write(format='mol')  # 或者使用其他合适的格式
            self.rdkit_mol = Chem.MolFromMolBlock(mol_string)
        
        if self.rdkit_mol is None:
            raise ValueError("Failed to convert molecule to RDKit format.")
        
        # 生成 MOLBlock V3000 格式
        self.molblock_v3000 = Chem.MolToV3000MolBlock(self.rdkit_mol)
        
        # 检查是否有修改需求
        self.has_modifications = bool(linknode_specs or position_variation_specs)
        
        logger.info(f"Successfully converted molecule to RDKit format")
        logger.debug(f"RDKit molecule atoms: {self.rdkit_mol.GetNumAtoms()}")
        logger.debug(f"RDKit molecule bonds: {self.rdkit_mol.GetNumBonds()}")
        logger.debug(f"MOLBlock V3000 size: {len(self.molblock_v3000)} characters")
        logger.debug(f"Has modifications: {self.has_modifications}")
        
        # 存储枚举参数以供后续使用
        self.linknode_specs = linknode_specs
        self.position_variation_specs = position_variation_specs

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
        if not hasattr(self, 'molblock_v3000') or not self.molblock_v3000:
            raise ValueError("MOLBlock V3000 not generated. Call _write_input() first.")
        
        if not hasattr(self, 'rdkit_mol') or not self.rdkit_mol:
            raise ValueError("RDKit molecule not available. Call _write_input() first.")
        
        logger.info(f"Using MOLBlock V3000 ({len(self.molblock_v3000)} chars) and RDKit molecule")
        logger.debug(f"RDKit molecule: {self.rdkit_mol}")
        
        # 检查是否有修改（LINKNODE 或 Position Variation）
        has_modifications = getattr(self, 'has_modifications', False)
        
        # 这里将实现实际的枚举逻辑
        # 使用 self.molblock_v3000 和 self.rdkit_mol 进行 RDKit 枚举操作
        logger.info("Enumeration logic will be implemented here")
        logger.info(f"Modifications applied: {has_modifications}")
        
        # 创建输出文件，包含 MOLBlock 信息
        with open(self.job_outputfile, 'w') as f:
            f.write("# Enumerated structures will be written here\n")
            f.write(f"# Job: {job.label}\n")
            f.write(f"# Input molecule: {job.molecule.get_chemical_formula()}\n")
            f.write(f"# RDKit molecule atoms: {self.rdkit_mol.GetNumAtoms()}\n")
            f.write(f"# RDKit molecule bonds: {self.rdkit_mol.GetNumBonds()}\n")
            f.write(f"# MOLBlock V3000 size: {len(self.molblock_v3000)} characters\n")
            f.write(f"# Has modifications: {has_modifications}\n")
            f.write(f"# LINKNODE specs: {getattr(self, 'linknode_specs', [])}\n")
            f.write(f"# Position variation specs: {getattr(self, 'position_variation_specs', [])}\n")
            f.write("\n")
            f.write("# Generated MOLBlock V3000:\n")
            # 写入 MOLBlock 的前几行作为示例
            molblock_lines = self.molblock_v3000.split('\n')
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