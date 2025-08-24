import re

class MolBlockV3K:
    """
    A class for parsing and manipulating V3000 molblock content.
    Supports adding, removing, and modifying atoms and bonds,
    as well as outputting the current molblock string.
    """
    def __init__(self, molblock_str: str):
        """
        Initialize MolBlockV3K object and parse the V3000 molblock string.
        Splits header, atoms, bonds, linknodes, and footer sections.
        """
        self.raw = molblock_str
        self.atoms = []   # List of atom information, each item is a dict
        self.bonds = []   # List of bond information, each item is a dict
        self.linknodes = []  # List of LINKNODE information, each item is a dict or raw string
        self.header = []  # Header information (e.g., file header, COUNTS, etc.)
        self.footer = []  # Footer information (e.g., END CTAB, M END)
        self._parse_molblock(molblock_str)

    def _parse_molblock(self, molblock_str):
        """
        Parse the V3000 molblock string and separate header, atom, bond, linknode, and footer sections.
        """
        lines = molblock_str.splitlines()
        in_atom = False
        in_bond = False
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('M  V30 BEGIN ATOM'):
                in_atom = True
                continue
            if line_stripped.startswith('M  V30 END ATOM'):
                in_atom = False
                continue
            if line_stripped.startswith('M  V30 BEGIN BOND'):
                in_bond = True
                continue
            if line_stripped.startswith('M  V30 END BOND'):
                in_bond = False
                continue
            if line_stripped.startswith('M  V30 LINKNODE'):
                linknode = self._parse_linknode_line(line_stripped)
                if linknode:
                    self.linknodes.append(linknode)
                continue
            if in_atom:
                atom = self._parse_atom_line(line)
                if atom:
                    self.atoms.append(atom)
                continue
            if in_bond:
                bond = self._parse_bond_line(line)
                if bond:
                    self.bonds.append(bond)
                continue
            # Header and footer
            if not in_atom and not in_bond:
                if not self.atoms and not self.bonds:
                    self.header.append(line)
                else:
                    self.footer.append(line)

    def _parse_linknode_line(self, line):
        """
        Parse a LINKNODE line and return a dictionary of linknode properties.
        Example: M  V30 LINKNODE 1 4 2 1 2 1 5
        """
        parts = line.split()
        if len(parts) < 4 or parts[0] != 'M' or parts[1] != 'V30' or parts[2] != 'LINKNODE':
            return None
        # Simple parsing, store all content as list
        return {'raw': line, 'values': parts[3:]}

    def get_linknodes(self):
        """
        Return a list of all LINKNODE information in the molblock.
        Each item is a dict with raw line and parsed values.
        """
        return self.linknodes.copy()

    def add_linknode(self, values):
        """
        Add a LINKNODE entry to the molblock.
        Args:
            values: list of values (e.g., ['1', '4', '2', '1', '2', '1', '5']) or a string
        Returns:
            The raw LINKNODE line added.
        """
        # Support both string and list input
        if isinstance(values, str):
            # Split by regex and join with single space, remove extra spaces
            values = [x.strip() for x in re.split(r'[:,\s]+', values) if x.strip()]
        # Ensure all elements are strings
        values = [str(v) for v in values]
        # Always use exact prefix with two spaces: 'M  V30 LINKNODE'
        line = 'M  V30 LINKNODE ' + ' '.join(values)
        # Ensure only single space between numbers (but keep two spaces after M)
        # Remove any accidental extra spaces after prefix
        line = re.sub(r'^(M)\s{2,}(V30 LINKNODE)', r'M  V30 LINKNODE', line)
        # Remove extra spaces after prefix (if any)
        line = re.sub(r'^(M  V30 LINKNODE)\s+', r'M  V30 LINKNODE ', line)
        # Ensure only single space between numbers (after prefix)
        prefix = 'M  V30 LINKNODE '
        if line.startswith(prefix):
            rest = line[len(prefix):]
            rest = re.sub(r'\s+', ' ', rest)
            line = prefix + rest
        self.linknodes.append({'raw': line, 'values': values})
        return line

    def remove_linknode(self, idx):
        """
        Remove a LINKNODE entry by its index in the linknodes list.
        Args:
            idx: index of the LINKNODE to remove
        """
        if 0 <= idx < len(self.linknodes):
            del self.linknodes[idx]

    def _parse_atom_line(self, line):
        """
        Parse an atom line and return a dictionary of atom properties.
        Example: M  V30 1 C -1.7083 2.415 0 0
        """
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'M' or parts[1] != 'V30':
            return None
        return {
            'idx': int(parts[2]),
            'element': parts[3],
            'x': float(parts[4]),
            'y': float(parts[5]),
            'z': float(parts[6]),
            'extra': parts[7:] if len(parts) > 7 else []
        }

    def _parse_bond_line(self, line):
        """
        Parse a bond line and return a dictionary of bond properties.
        Example: M  V30 1 1 1 2
        """
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'M' or parts[1] != 'V30':
            return None
        # Handle optional attributes (e.g., ENDPTS/ATTACH)
        extra = ' '.join(parts[6:]) if len(parts) > 6 else ''
        return {
            'idx': int(parts[2]),
            'type': int(parts[3]),
            'atom1': int(parts[4]),
            'atom2': int(parts[5]),
            'extra': extra
        }

    # --- Operation interfaces ---
    def add_atom(self, element, x, y, z, extra=None):
        """
        Add a new atom to the molblock.
        Returns the index of the new atom.

        Example:
            idx = molblock_obj.add_atom("C", 0.0, 0.0, 0.0)
        """
        idx = len(self.atoms) + 1
        atom = {
            'idx': idx,
            'element': element,
            'x': x,
            'y': y,
            'z': z,
            'extra': extra or []
        }
        self.atoms.append(atom)
        return idx

    def remove_atom(self, idx):
        """
        Remove an atom by its index and also remove any bonds associated with it.

        Example:
            molblock_obj.remove_atom(idx)
        """
        self.atoms = [a for a in self.atoms if a['idx'] != idx]
        self.bonds = [b for b in self.bonds if b['atom1'] != idx and b['atom2'] != idx]

    def modify_atom(self, idx, **kwargs):
        """
        Modify properties of an atom by its index.

        Example:
            molblock_obj.modify_atom(idx, x=1.0, y=2.0)
        """
        for atom in self.atoms:
            if atom['idx'] == idx:
                atom.update(kwargs)
                break

    def add_bond(self, type_, atom1, atom2, extra=''):
        """
        Add a new bond to the molblock.
        Returns the index of the new bond.

        Example:
            b_idx = molblock_obj.add_bond(1, 1, 2)
        """
        idx = len(self.bonds) + 1
        bond = {
            'idx': idx,
            'type': type_,
            'atom1': atom1,
            'atom2': atom2,
            'extra': extra
        }
        self.bonds.append(bond)
        return idx

    def add_virtual_atom(self, x, y, z, extra=None):
        """
        Add a virtual atom (element='*') to the molblock.
        Returns the index of the new virtual atom.

        Example:
            v_idx = molblock_obj.add_virtual_atom(1.0, 2.0, 0.0)
        """
        return self.add_atom("*", x, y, z, extra)

    def remove_bond(self, idx):
        """
        Remove a bond by its index.

        Example:
            molblock_obj.remove_bond(b_idx)
        """
        self.bonds = [b for b in self.bonds if b['idx'] != idx]

    def modify_bond(self, idx, **kwargs):
        """
        Modify properties of a bond by its index.

        Example:
            molblock_obj.modify_bond(b_idx, type=2)
        """
        for bond in self.bonds:
            if bond['idx'] == idx:
                bond.update(kwargs)
                break

    def get_molblock(self):
        """
        Output the current molblock string with all atoms, bonds, and linknodes.

        Example:
            molblock_str = molblock_obj.get_molblock()
        """
        lines = []
        lines.extend(self.header)
        lines.append('M  V30 BEGIN ATOM')
        for atom in self.atoms:
            atom_line = f"M  V30 {atom['idx']} {atom['element']} {atom['x']} {atom['y']} {atom['z']}"
            if atom['extra']:
                atom_line += ' ' + ' '.join(str(e) for e in atom['extra'])
            lines.append(atom_line)
        lines.append('M  V30 END ATOM')
        lines.append('M  V30 BEGIN BOND')
        for bond in self.bonds:
            bond_line = f"M  V30 {bond['idx']} {bond['type']} {bond['atom1']} {bond['atom2']}"
            if bond['extra']:
                bond_line += f" {bond['extra']}"
            lines.append(bond_line)
        lines.append('M  V30 END BOND')
        for linknode in self.linknodes:
            lines.append(linknode['raw'])
        lines.extend(self.footer)
        return '\n'.join(lines)

    def get_first_atom(self):
        """
        Return the information of the first atom in the molblock.
        Returns None if no atom exists.
        """
        if self.atoms:
            return self.atoms[0]
        return None

    def get_last_atom(self):
        """
        Return the information of the last atom in the molblock.
        Returns None if no atom exists.
        """
        if self.atoms:
            return self.atoms[-1]
        return None

    def get_virtual_atoms(self):
        """
        Return a list of all virtual atoms (element == '*') in the molblock.
        Each item is a dict with atom information.
        """
        return [atom for atom in self.atoms if atom.get('element') == '*']
