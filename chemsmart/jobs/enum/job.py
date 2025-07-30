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

    def __init__(
        self, molecule, label=None, **kwargs
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

    @property
    def outputfile(self):
        outputfile = self.label + ".xyz"
        return os.path.join(self.folder, outputfile)

    def _run(self, **kwargs):
        """Run the enumeration directly using RDKit."""
        logger.info(f"Running EnumJob {self}")
        # Actual enumeration logic will be implemented here
        # For now, this is a placeholder that satisfies the abstract method requirement
        pass

    @classmethod
    def from_filename(
        cls,
        filename,
        index="-1",
        label=None,
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
            **kwargs,
        )

    @classmethod
    def from_pubchem(
        cls, identifier, label=None, **kwargs
    ):
        """Create an EnumJob from a PubChem identifier."""
        molecules = Molecule.from_pubchem(identifier=identifier)

        return cls(
            molecule=molecules,
            label=label,
            **kwargs,
        )

