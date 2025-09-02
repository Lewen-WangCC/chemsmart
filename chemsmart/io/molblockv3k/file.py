from functools import cached_property

from chemsmart.io.molblockv3k.v3k import MolBlockV3K
from chemsmart.utils.mixins import FileMixin
from chemsmart.utils.utils import string2index_1based


class MolV3KFile(FileMixin):
    def __init__(self, filename):
        self.filename = filename
        # self.molblock = None

    def __repr__(self):
        return f"MolV3000File({self.filename})"

    def __str__(self):
        return f"MolV3000File object with filename: {self.filename}"

    @cached_property
    def num_atoms(self):
        return int(self.contents[0])
    
    @cached_property
    def molecule(self):
        return self.get_molecules(index="-1")
    
    @cached_property
    def comments(self):
        return self.get_comments(index="-1")

    def read(self):
        with open(self.filename, 'r') as f:
            molblock_str = f.read()
            self.molblock = MolBlockV3K(molblock_str)

    def write(self):
        with open(self.file_path, 'w') as f:
            f.write(self.molblock.raw)