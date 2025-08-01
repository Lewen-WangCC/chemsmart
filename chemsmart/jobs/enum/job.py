import logging
import os
from typing import Type

from chemsmart.io.molecules.structure import Molecule
from chemsmart.jobs.job import Job
from chemsmart.jobs.runner import JobRunner
from chemsmart.utils.utils import string2index_1based

logger = logging.getLogger(__name__)


class EnumJob(Job):
    PROGRAM = "Enum"
    TYPE = "enum" 

    def __init__(
        self, molecule, label=None, linknode_specs=None, position_variation_specs=None, **kwargs
    ):
        super().__init__(
            molecule=molecule, label=label, jobrunner=None, **kwargs
        )

        if not isinstance(molecule, Molecule):
            raise ValueError(
                f"Molecule must be instance of Molecule for {self}, but is {molecule} instead!"
            )

        self.molecule = molecule.copy() if molecule is not None else None

        if label is None:
            label = molecule.get_chemical_formula(empirical=True)
        self.label = label
        
        # 存储枚举相关参数
        self.linknode_specs = linknode_specs or []
        self.position_variation_specs = position_variation_specs or []
        
        logger.debug(f"EnumJob created with LINKNODE specs: {self.linknode_specs}")
        logger.debug(f"EnumJob created with Position Variation specs: {self.position_variation_specs}")

    @property
    def outputfile(self):
        outputfile = self.label + ".xyz"
        return os.path.join(self.folder, outputfile)

    def _run(self, **kwargs):
        """Run the enumeration using the assigned jobrunner."""
        if self.jobrunner is None:
            raise ValueError(f"No jobrunner assigned to {self}")
        
        logger.info(f"Running EnumJob {self} with jobrunner {self.jobrunner}")
        self.jobrunner.run(self, **kwargs)

    @classmethod
    def from_filename(
        cls,
        filename,
        index="-1",
        label=None,
        linknode_specs=None,
        position_variation_specs=None,
        **kwargs,
    ):
        """Create an EnumJob from a file containing molecule data."""
        logger.info(f"Reading molecules from file: {filename}.")
        molecules = Molecule.from_filepath(
            filepath=filename, index=":", return_list=True
        )
        logger.info(f"Num of molecules read: {len(molecules)}.")
        molecules = molecules[string2index_1based(index)]

        return cls(
            molecule=molecules,
            label=label,
            linknode_specs=linknode_specs,
            position_variation_specs=position_variation_specs,
            **kwargs,
        )

    @classmethod
    def from_pubchem(
        cls, identifier, label=None, linknode_specs=None, position_variation_specs=None, **kwargs
    ):
        """Create an EnumJob from a PubChem identifier."""
        molecules = Molecule.from_pubchem(identifier=identifier)

        return cls(
            molecule=molecules,
            label=label,
            linknode_specs=linknode_specs,
            position_variation_specs=position_variation_specs,
            **kwargs,
        )

