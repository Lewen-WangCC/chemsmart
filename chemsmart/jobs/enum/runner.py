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
        logger.debug(f"EnumJobRunner server: {self.server}")
        logger.debug(f"EnumJobRunner scratch: {self.scratch} (typically not needed)")

    @property
    def executable(self):
        """EnumJobRunner doesn't need external executable, returns None."""
        return None

    def _prerun(self, job):
        """Prepare for enumeration job execution."""
        self._assign_variables(job)

    def _assign_variables(self, job):
        """Set up file paths and directories for enumeration output."""
        # 设置输出文件路径 - 始终在作业目录中
        self.job_outputfile = job.outputfile
        
        if self.scratch and self.scratch_dir:
            self._set_up_variables_in_scratch(job)
        else:
            self._set_up_variables_in_job_directory(job)
        
        # 确保输出目录存在
        if not os.path.exists(self.running_directory):
            os.makedirs(self.running_directory, exist_ok=True)
            logger.info(f"Created output directory: {self.running_directory}")

    def _set_up_variables_in_scratch(self, job):
        """Set up file paths when using scratch directory."""
        scratch_job_dir = os.path.join(self.scratch_dir, job.label)
        if not os.path.exists(scratch_job_dir):
            with suppress(FileExistsError):
                os.makedirs(scratch_job_dir)
        self.running_directory = scratch_job_dir
        logger.debug(f"Running directory (scratch): {self.running_directory}")
        
        # 枚举作业的输出文件
        job_outputfile = job.label + ".xyz"
        scratch_job_outputfile = os.path.join(scratch_job_dir, job_outputfile)
        self.job_outputfile = os.path.abspath(scratch_job_outputfile)

    def _set_up_variables_in_job_directory(self, job):
        """Set up file paths when using job directory directly."""
        self.running_directory = job.folder
        logger.debug(f"Running directory: {self.running_directory}")
        
        # 保持与 scratch 模式一致的文件命名逻辑
        job_outputfile = job.label + ".xyz"
        job_outputfile_path = os.path.join(job.folder, job_outputfile)
        self.job_outputfile = os.path.abspath(job_outputfile_path)

    def _write_input(self, job):
        """Prepare input for enumeration - convert molecule to RDKit MOLBlock format."""
        from chemsmart.jobs.enum.writer import EnumWriter
        
        # 获取 linknode 和 position_variation 参数
        linknode_specs = getattr(job, 'linknode_specs', [])
        position_variation_specs = getattr(job, 'position_variation_specs', [])
        
        logger.debug(f"Creating EnumWriter with LINKNODE specs: {linknode_specs}")
        logger.debug(f"Creating EnumWriter with Position Variation specs: {position_variation_specs}")
        
        # 创建 EnumWriter 并生成 MOLBlock
        enum_writer = EnumWriter(
            job=job,
            linknode=linknode_specs,
            position_variation=position_variation_specs
        )
        
        # 生成 MOLBlock V3000 并存储在内存中
        self.molblock_v3000 = enum_writer.write()
        self.rdkit_mol = enum_writer.get_rdkit_mol()
        
        logger.info(f"Generated MOLBlock V3000 for enumeration job: {job.label}")
        logger.debug(f"MOLBlock size: {len(self.molblock_v3000)} characters")
        logger.debug(f"RDKit molecule: {self.rdkit_mol}")
        logger.debug(f"Has modifications: {enum_writer.has_modifications()}")
        
        # DEBUG: Print MOLBlock and exit for testing
        print("=" * 60)
        print("MOLBLOCK V3000 OUTPUT:")
        print("=" * 60)
        print(self.molblock_v3000)
        print("=" * 60)
        print("Conversion successful! Exiting for testing...")
        import sys
        sys.exit(0)
        
        # 存储 writer 实例以便后续使用
        self.enum_writer = enum_writer

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
        has_modifications = self.enum_writer.has_modifications() if hasattr(self, 'enum_writer') else False
        
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