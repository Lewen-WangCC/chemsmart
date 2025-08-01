from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem import rdMolEnumerator
import rdkit

import functools
import logging
import os

import click

from chemsmart.io.molecules.structure import Molecule
from chemsmart.utils.cli import MyCommand


logger = logging.getLogger(__name__)


def click_enum_settings_options(f):
    """Common click options for enumeration settings."""
    @click.option(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Filename containing molecule structure for enumeration (SDF, MOL, XYZ, etc.)",
    )
    @click.option(
        "-l",
        "--label",
        type=str,
        default=None,
        help="Label for enumeration job (without extension). Used for output file naming.",
    )
    @click.option(
        "-a",
        "--append-label",
        type=str,
        default=None,
        help="Name to be appended to base filename for the enumeration job.",
    )
    @click.option(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for enumerated structures. If not specified, uses current working directory.",
    )
    
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options

def click_emum_jobtype_options(f):
    """Common click options for enumeration job type."""
    @click.option(
        "--linknode",
        type=str,
        multiple=True,
        default=None,
        help="LINKNODE specification: 'node_id:atom_num:definition_of_link' (e.g., '1:4:2,1,2,1,5')",
    )
    @click.option(
        "--position-variation",
        type=str,
        multiple=True,
        default=None,
        help="Position Variation specification: 'bond_type:first_atom:endpt_count,endpt1,endpt2,...:attach_type' (e.g., '1:8:3,1,5,6:ANY')",
    )
    @functools.wraps(f)
    def wrapper_enum_jobtype_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_enum_jobtype_options


@click.command(cls=MyCommand)  # 改为 command 而不是 group
@click_enum_settings_options
@click_emum_jobtype_options
@click.option(
    "-P",
    "--pubchem",
    type=str,
    default=None,
    help="Queries structure from PubChem using name, smiles, cid and conformer information.",
)
@click.pass_context
def enum(
    ctx,
    filename,
    label,
    append_label,
    output_dir,
    linknode,
    position_variation,
    pubchem,
    **kwargs,
    ):

    # obtain molecule structure
    if filename is None and pubchem is None:
        raise ValueError(
            "[filename] or [pubchem] has not been specified!\nPlease specify one of them!"
        )
    if filename and pubchem:
        raise ValueError(
            "Both [filename] and [pubchem] have been specified!\nPlease specify only one of them."
        )

    if filename:
        molecules = Molecule.from_filepath(
            filepath=filename, index=":", return_list=True
        )
        assert (
            molecules is not None
        ), f"Could not obtain molecule from {filename}!"
        logger.debug(f"Obtained molecule {molecules} from {filename}")

    if pubchem:
        molecules = Molecule.from_pubchem(identifier=pubchem, return_list=True)
        assert (
            molecules is not None
        ), f"Could not obtain molecule from PubChem {pubchem}!"
        logger.debug(f"Obtained molecule {molecules} from PubChem {pubchem}")

    logger.debug(f"Obtained molecules: {molecules}")

    # update labels - exactly same logic as gaussian module
    if label is not None and append_label is not None:
        raise ValueError(
            "Only give enum input filename or name to be appended, but not both!"
        )
    if append_label is not None:
        label = os.path.splitext(os.path.basename(filename))[0]
        label = f"{label}_{append_label}"
    if label is None and append_label is None:
        if filename:
            label = os.path.splitext(os.path.basename(filename))[0]
        elif pubchem:
            label = pubchem.replace(' ', '_')
        else:
            label = "enum"
        label = f"{label}_enum"
    
    logger.debug(f"Enumeration job label: {label}")

    # handle output directory
    if output_dir is None:
        output_dir = os.getcwd()  # use current working directory
    else:
        output_dir = os.path.abspath(output_dir)  # convert to absolute path
    
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    logger.debug(f"Output directory: {output_dir}")

    # process multiple linknode and position_variation specifications
    if linknode:
        logger.debug(f"LINKNODE specifications: {linknode}")
        for ln in linknode:
            logger.debug(f"Processing LINKNODE: {ln}")
    
    if position_variation:
        logger.debug(f"Position variation specifications: {position_variation}")
        for pv in position_variation:
            logger.debug(f"Processing position variation: {pv}")

    # 直接创建和运行 EnumJob，而不需要子命令
    logger.info("Starting enumeration job execution")
    
    # 获取分子
    molecule = molecules[-1] if isinstance(molecules, list) else molecules
    
    # 转换参数格式
    linknode_specs = list(linknode) if linknode else []
    position_variation_specs = list(position_variation) if position_variation else []
    
    # 创建 EnumJob
    from chemsmart.jobs.enum.job import EnumJob
    
    enum_job = EnumJob(
        molecule=molecule,
        label=label,
        linknode_specs=linknode_specs,
        position_variation_specs=position_variation_specs,
        jobrunner=None,
        **kwargs,
    )
    
    logger.info(f"Created EnumJob: {enum_job}")
    logger.debug(f"LINKNODE specs: {enum_job.linknode_specs}")
    logger.debug(f"Position Variation specs: {enum_job.position_variation_specs}")
    
    # 创建 JobRunner
    from chemsmart.jobs.runner import JobRunner
    from chemsmart.settings.server import Server
    
    server = Server.current()
    jobrunner = JobRunner.from_job(
        job=enum_job,
        server=server,
        scratch=False,  # EnumJobRunner 默认不使用 scratch
        fake=False,
    )
    logger.info(f"Created JobRunner: {jobrunner}")
    
    # 设置 jobrunner
    enum_job.jobrunner = jobrunner
    
    # 运行作业
    logger.info("Executing enumeration job...")
    enum_job._run()
    
    logger.info("Enumeration job completed successfully")
    return enum_job