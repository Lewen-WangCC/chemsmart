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
        self, 
        molecule, 
        label=None, 
        jobrunner=None,
        linknode_specs=None, 
        position_variation_specs=None, 
        **kwargs
    ):
        super().__init__(
            molecule=molecule, label=label, jobrunner=jobrunner, **kwargs
        )

        if not isinstance(molecule, Molecule):
            raise ValueError(
                f"Molecule must be instance of Molecule for {self}, but is {molecule} instead!"
            )
        
        # 存储枚举相关参数
        self.linknode_specs = linknode_specs or []
        self.position_variation_specs = position_variation_specs or []
        
        logger.debug(f"EnumJob created with LINKNODE specs: {self.linknode_specs}")
        logger.debug(f"EnumJob created with Position Variation specs: {self.position_variation_specs}")

    @property
    def inputfile(self):
        """Input file path - not typically used for enumeration but kept for consistency."""
        inputfile = self.label + ".mol"
        return os.path.join(self.folder, inputfile)

    @property
    def logfile(self):
        """Log file path for enumeration process."""
        logfile = "log." + self.label
        return os.path.join(self.folder, logfile)

    @property
    def outputfile(self):
        """Output file path for enumerated structures."""
        outputfile = self.label + ".xyz"
        return os.path.join(self.folder, outputfile)

    @property
    def errfile(self):
        """Error file path for enumeration process."""
        errfile = self.label + ".err"
        return os.path.join(self.folder, errfile)

    def _output(self):
        """Return the output file path if it exists."""
        if not os.path.exists(self.outputfile):
            return None
        return os.path.abspath(self.outputfile)

    def _job_is_complete(self):
        """Check if enumeration job is complete."""
        return os.path.exists(self.outputfile)

    def _run(self, **kwargs):
        """Run the enumeration using the assigned jobrunner."""
        self.jobrunner.run(self, **kwargs)

    @classmethod
    def from_filename(
        cls,
        filename,
        index="-1",
        label=None,
        jobrunner=None,
        linknode_specs=None,
        position_variation_specs=None,
        **kwargs,
    ):
        # get all molecule in a file and give the result as a list
        logger.info(f"Reading molecules from file: {filename}.")
        molecules = Molecule.from_filepath(
            filepath=filename, index=":", return_list=True, **kwargs
        )
        logger.info(f"Num of molecules read: {len(molecules)}.")

        if label is None:
            # by default, if no label is given and the job is read in
            # from a file, the label is set to the file basename
            label = os.path.basename(filename).split(".")[0]

        logger.info(f"Num of molecules read: {len(molecules)}.")
        molecules = molecules[string2index_1based(index)]
        logger.info(f"Num of molecules to use: {len(molecules)}.")

        # Create jobrunner if not provided
        if jobrunner is None:
            jobrunner = JobRunner.from_job(
                cls(
                    molecule=molecules,
                    label=label,
                    linknode_specs=linknode_specs,
                    position_variation_specs=position_variation_specs,
                    **kwargs,
                ),
                server=kwargs.get("server"),
                scratch=kwargs.get("scratch"),
                fake=kwargs.get("fake", False),
                **kwargs,
            )

        return cls(
            molecule=molecules,
            label=label,
            jobrunner=jobrunner,
            linknode_specs=linknode_specs,
            position_variation_specs=position_variation_specs,
            **kwargs,
        )

    @classmethod
    def from_pubchem(
        cls, identifier, label=None, jobrunner=None, linknode_specs=None, position_variation_specs=None, **kwargs
    ):
        """Create an EnumJob from a PubChem identifier."""
        molecules = Molecule.from_pubchem(identifier=identifier)

        # Create jobrunner if not provided
        if jobrunner is None:
            jobrunner = JobRunner.from_job(
                cls(
                    molecule=molecules,
                    label=label,
                    linknode_specs=linknode_specs,
                    position_variation_specs=position_variation_specs,
                    **kwargs,
                ),
                server=kwargs.get("server"),
                scratch=kwargs.get("scratch"),
                fake=kwargs.get("fake", False),
                **kwargs,
            )

        return cls(
            molecule=molecules,
            label=label,
            jobrunner=jobrunner,
            linknode_specs=linknode_specs,
            position_variation_specs=position_variation_specs,
            **kwargs,
        )

