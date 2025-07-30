import logging
import os
import shlex
import subprocess
from contextlib import suppress
from datetime import datetime
from functools import lru_cache
from glob import glob
from random import random
from shutil import copy, rmtree

from chemsmart.jobs.runner import JobRunner
from chemsmart.utils.periodictable import PeriodicTable

from rdkit import Chem
from rdkit.Chem import rdMolEnumerator, AllChem

pt = PeriodicTable()

logger = logging.getLogger(__name__)


class EnumJobRunner(JobRunner):
    
    PROGRAM = "Enum"
    
    def __init__(self, job):
        super().__init__(job)