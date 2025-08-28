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


def click_iterate_settings_options(f):
    """Common click options for iteration settings."""
    @click.option(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Filename containing molecule structure for iteration (SDF, MOL, XYZ, etc.)",
    )
    @click.option(
        "-l",
        "--label",
        type=str,
        default=None,
        help="Label for iteration job (without extension). Used for output file naming.",
    )
    @click.option(
        "-a",
        "--append-label",
        type=str,
        default=None,
        help="Name to be appended to base filename for the iteration job.",
    )
    @click.option(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for iterated structures. If not specified, uses current working directory.",
    )
    
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options

def click_iterate_jobtype_options(f):
    """Common click options for iteration job type."""
    @click.option(
        "--linknode",
        type=str,
        multiple=True,
        default=None,
        help="LINKNODE specification: 'minrep:maxrep:nbonds,inatom1,outatom1,inatom2,outatom2,...' (e.g., 1:2:2,20,19,20,21)",
    )
    @click.option(
        "--position-variation",
        type=str,
        multiple=True,
        default=None,
        help=(
            "Position Variation specification:"
            "\n  1. 'bond_type:group_first_atom:endpt_count,endpt1,endpt2,...:attach_type' (e.g., 1:8:3,1,5,6:ANY)"
            "\n  2. 'bond_type:virtual_atom:group_first_atom:endpt_count,endpt1,endpt2,...:attach_type' (e.g., 1:7:8:3,1,5,6:ANY)"
        ),
    )
    @functools.wraps(f)
    def wrapper_iterate_jobtype_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_iterate_jobtype_options


@click.command(cls=MyCommand)  # 改为 command 而不是 group
@click_iterate_settings_options
@click_iterate_jobtype_options
@click.option(
    "-P",
    "--pubchem",
    type=str,
    default=None,
    help="Queries structure from PubChem using name, smiles, cid and conformer information.",
)
@click.pass_context
def iterate(
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
            "Only give iterate input filename or name to be appended, but not both!"
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
            label = "iterate"
        label = f"{label}_iterate"

    logger.debug(f"Iteration job label: {label}")

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

    # Directly create and run IterateJob without subcommands
    logger.info("Starting iteration job execution")

    # obtain molecule
    molecule = molecules[-1] if isinstance(molecules, list) else molecules

    # convert parameter format
    linknode_specs = list(linknode) if linknode else []
    position_variation_specs = list(position_variation) if position_variation else []
    
    # create IterateJob
    from chemsmart.jobs.iterate.job import IterateJob

    iterate_job = IterateJob(
        molecule=molecule,
        label=label,
        output_dir=output_dir,
        linknode_specs=linknode_specs,
        position_variation_specs=position_variation_specs,
        **kwargs,
    )

    logger.info(f"Created IterateJob: {iterate_job}")
    logger.debug(f"LINKNODE specs: {iterate_job.linknode_specs}")
    logger.debug(f"Position Variation specs: {iterate_job.position_variation_specs}")

    # create JobRunner
    from chemsmart.jobs.runner import JobRunner
    from chemsmart.settings.server import Server
    
    server = Server.current()
    jobrunner = JobRunner.from_job(
        job=iterate_job,
        server=server,
        scratch=False,  # IterateJobRunner don't use scratch by default
        fake=False,
    )
    logger.info(f"Created JobRunner: {jobrunner}")

    # Set jobrunner
    iterate_job.jobrunner = jobrunner

    # Run job
    logger.info("Executing iteration job...")
    iterate_job._run()

    logger.info("Iteration job completed successfully")
    return iterate_job