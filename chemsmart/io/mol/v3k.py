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
        # COUNTS information: 
        # na=number of atoms, 
        # nb=number of bonds, 
        # nsg=number of S-groups, 
        # n3d=number of 3D objects, 
        # chiral=chiral flag (0/1), 
        # regno=registry number (optional)
        self.count = {
            "na": 0,
            "nb": 0,
            "nsg": 0,
            "n3d": 0,
            "chiral": 0,
            "regno": ""
        }
        self.atoms = []   # List of atom information, each item is a dict
        self.bonds = []   # List of bond information, each item is a dict
        self.linknodes = []  # List of LINKNODE information, each item is a structured dict with keys (idx, minrep, maxrep, nbonds, atoms).
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
            if line_stripped.startswith("M  V30 COUNTS"):
                parts = line_stripped.split()
                # Required numeric fields
                self.count["na"] = int(parts[3])
                self.count["nb"] = int(parts[4])
                self.count["nsg"] = int(parts[5])
                self.count["n3d"] = int(parts[6])
                self.count["chiral"] = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else 0
                # Optional REGNO=regno (may appear after the numeric fields)
                self.count["regno"] = ""  # reset before parsing
                for tok in parts[8:]:
                    up = tok.upper()
                    if up.startswith("REGNO="):
                        self.count["regno"] = tok.split("=", 1)[1].strip()
                        break
                continue
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
        Parse a LINKNODE line and return a structured dictionary:
        {'idx': <int>, 'minrep': int, 'maxrep': int, 'nbonds': int, 'atoms': [ints...]}
        Example: M  V30 LINKNODE 1 4 2 1 2 1 5
        """
        parts = line.split()
        if len(parts) < 9 or parts[0] != 'M' or parts[1] != 'V30' or parts[2] != 'LINKNODE':
            return None
        try:
            # The LINKNODE line is: M  V30 LINKNODE minrep maxrep nbonds atom1 atom2 ...
            minrep = int(parts[3])
            maxrep = int(parts[4])
            nbonds = int(parts[5])
            atoms = [int(v) for v in parts[6:]]
            idx = len(self.linknodes) + 1
            return {'idx': idx, 'minrep': minrep, 'maxrep': maxrep, 'nbonds': nbonds, 'atoms': atoms}
        except Exception:
            return None

    def get_linknodes(self):
        """
        Return a shallow copy of the list of all LINKNODE entries (structured dicts).
        """
        return self.linknodes.copy()

    def add_linknode(self, values):
        """
        Add a LINKNODE entry to the molblock.
        Args:
            values: string or list; parses values into structured dict.
        Returns:
            The structured linknode dict added.
        """
        # Accept string or list; normalize whitespace/commas, coerce to ints.
        if isinstance(values, str):
            values = [x.strip() for x in re.split(r'[:,\s]+', values) if x.strip()]
        # If values is a list of strings, coerce to ints
        values = [int(v) for v in values]
        if len(values) < 4:
            raise ValueError("LINKNODE requires at least 4 values: minrep, maxrep, nbonds, atoms...")
        minrep = int(values[0])
        maxrep = int(values[1])
        nbonds = int(values[2])
        atoms = [int(v) for v in values[3:]]
        idx = len(self.linknodes) + 1
        ln = {'idx': idx, 'minrep': minrep, 'maxrep': maxrep, 'nbonds': nbonds, 'atoms': atoms}
        self.linknodes.append(ln)
        return ln

    def remove_linknode(self, idx):
        """
        Remove a LINKNODE entry by its internal idx field.
        Args:
            idx: internal 'idx' field of the LINKNODE to remove.
        """
        found = False
        for i, ln in enumerate(self.linknodes):
            if ln.get('idx') == idx:
                del self.linknodes[i]
                found = True
                break
        if found:
            # Resequence idx fields
            for j, ln in enumerate(self.linknodes, start=1):
                ln['idx'] = j

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
        Supports optional V3000 extras like ENDPTS=(...) and ATTACH=...
        Example lines:
          - "M  V30 39 1 36 39"
          - "M  V30 40 1 40 36 ENDPTS=(3 16 14 18) ATTACH=ANY"
        """
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'M' or parts[1] != 'V30':
            return None

        bond = {
            'idx': int(parts[2]),
            'type': int(parts[3]),
            'atom1': int(parts[4]),
            'atom2': int(parts[5]),
        }

        # Handle optional attributes (e.g., ENDPTS/ATTACH and any others)
        if len(parts) > 6:
            tokens = parts[6:]
            endpts = None
            attach = None
            extras = []

            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok.startswith('ENDPTS=('):
                    # Parse values possibly spanning multiple tokens until ')'
                    buf = tok[len('ENDPTS=('):]
                    vals = []
                    while True:
                        ended = buf.endswith(')')
                        if ended:
                            buf = buf[:-1]
                        if buf:
                            # split by whitespace and commas
                            for x in re.split(r'[\s,]+', buf):
                                if x:
                                    try:
                                        vals.append(int(x))
                                    except Exception:
                                        vals.append(x)
                        if ended:
                            break
                        i += 1
                        if i >= len(tokens):
                            break
                        buf = tokens[i]
                    endpts = vals
                elif tok.startswith('ATTACH='):
                    attach = tok.split('=', 1)[1]
                else:
                    extras.append(tok)
                i += 1

            if endpts is not None:
                bond['endpts'] = endpts  # keep the first count element as-is
            if attach is not None:
                bond['attach'] = attach
            if extras:
                bond['extra'] = ' '.join(extras)

        return bond

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
        self.renumber_atoms()
        return idx

    def remove_atom(self, idx):
        """
        Remove an atom by its index and also remove any bonds associated with it.

        Example:
            molblock_obj.remove_atom(idx)
        """
        self.atoms = [a for a in self.atoms if a['idx'] != idx]
        self.bonds = [b for b in self.bonds if b['atom1'] != idx and b['atom2'] != idx]
        self.renumber_all()

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

    def add_bond(self, type_, atom1, atom2, endpts=None, attach=None, extra=None):
        """
        Add a new bond to the molblock.
        Returns the index of the new bond.

        Example:
            b_idx = molblock_obj.add_bond(1, 1, 2)
        """
        idx = len(self.bonds) + 1
        a1 = int(atom1)
        a2 = int(atom2)

        # bond conflict prevention
        for b in self.bonds:
            if {b['atom1'], b['atom2']} == {a1, a2}:
                raise ValueError(f"Bond between atoms {a1}-{a2} already exists.")

        bond = {
            'idx': idx,
            'type': int(type_),
            'atom1': a1,
            'atom2': a2,
        }

        if attach is not None and endpts is None:
            raise ValueError("ATTACH provided but ENDPTS is missing for the bond")
        
        if endpts is not None and isinstance(endpts, list):
            bond['endpts'] = endpts
        if attach is not None and isinstance(attach, str):
            bond['attach'] = attach
        if extra is not None and isinstance(extra, str):
            bond['extra'] = extra
        self.bonds.append(bond)
        self.renumber_bonds()
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
        self.renumber_bonds()

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
        Renders LINKNODE lines from the structured dict (excludes internal idx).
        """
        lines = []
        lines.extend(self.header)
        # Add COUNTS line from self.count
        self.renew_count()  # Ensure counts are up to date
        counts_line = f"M  V30 COUNTS {self.count['na']} {self.count['nb']} {self.count['nsg']} {self.count['n3d']} {self.count['chiral']}"
        if self.count.get('regno'):
            counts_line += f" REGNO={self.count['regno']}"
        lines.append(counts_line)
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
        for ln in self.linknodes:
            # Render as: M  V30 LINKNODE {minrep} {maxrep} {nbonds} {atoms...}
            if isinstance(ln, dict) and 'minrep' in ln and 'maxrep' in ln and 'nbonds' in ln and 'atoms' in ln:
                line = f"M  V30 LINKNODE {ln['minrep']} {ln['maxrep']} {ln['nbonds']} " + ' '.join(str(a) for a in ln['atoms'])
                lines.append(line)
            # else: skip malformed
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

    def renew_count(self, nsg=None, n3d=None, chiral=None, regno=None):
        """
        Renew the atom and bond counts based on the current state.
        """
        self.count['na'] = len(self.atoms)
        self.count['nb'] = len(self.bonds)
        if nsg is not None:
            if not isinstance(nsg, int) or nsg < 0:
                raise ValueError("nsg must be a non-negative integer")
            self.count['nsg'] = nsg
        if n3d is not None:
            if not isinstance(n3d, int) or n3d < 0:
                raise ValueError("n3d must be a non-negative integer")
            self.count['n3d'] = n3d
        if chiral is not None:
            if not isinstance(chiral, int) or chiral not in (0, 1):
                raise ValueError("chiral must be 0 or 1")
            self.count['chiral'] = chiral
        if regno is not None:
            if not isinstance(regno, str):
                raise ValueError("regno must be a string")
            self.count['regno'] = regno
        

        self.count['chiral'] = sum(1 for a in self.atoms if a.get('chiral') is not None)

    def renumber_bonds(self, start: int = 1):
        """
        Renumber all bonds sequentially starting from `start`.
        Updates each bond's 'idx'.
        """
        for i, bond in enumerate(self.bonds, start=start):
            bond['idx'] = i

    def renumber_atoms(self, start: int = 1):
        """
        Renumber all atoms sequentially starting from `start`.
        Updates each atom's 'idx'.
        Also updates any bond references (atom1/atom2) and linknode atoms.
        """
        # Build mapping from old index to new index
        mapping = {atom['idx']: i for i, atom in enumerate(self.atoms, start=start)}
        # Update atoms
        for atom in self.atoms:
            atom['idx'] = mapping[atom['idx']]
        # Update bonds
        for bond in self.bonds:
            if bond['atom1'] in mapping:
                bond['atom1'] = mapping[bond['atom1']]
            if bond['atom2'] in mapping:
                bond['atom2'] = mapping[bond['atom2']]
        # Update linknodes: remap indices inside each linknode's atoms list
        for ln in self.linknodes:
            if isinstance(ln, dict) and 'atoms' in ln:
                ln['atoms'] = [mapping.get(a, a) for a in ln['atoms']]

    def renumber_all(self, atom_start: int = 1, bond_start: int = 1):
        """
        Renumber both atoms and bonds sequentially.
        Calls renumber_atoms() and renumber_bonds() in order.
        """
        self.renumber_atoms(start=atom_start)
        self.renumber_bonds(start=bond_start)
