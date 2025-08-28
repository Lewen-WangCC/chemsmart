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

    # --- Input interfaces ---
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
        self.renumber_all()
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

    def modify_atom(self, idx, element=None, x=None, y=None, z=None, extra=None):
        """
        Modify properties of an atom by its index.
        Only non-None arguments will be updated with minimal type validation.
        Supported fields:
          - element (str)
          - x, y, z (float)
          - extra (list[str])
        Returns the updated atom dict, or None if not found.
        """
        # find target atom
        target = None
        for a in self.atoms:
            if a.get('idx') == idx:
                target = a
                break
        if target is None:
            return None

        if element is not None:
            target['element'] = str(element)

        if x is not None:
            try:
                target['x'] = float(x)
            except Exception:
                raise ValueError("x must be a float")
        if y is not None:
            try:
                target['y'] = float(y)
            except Exception:
                raise ValueError("y must be a float")
        if z is not None:
            try:
                target['z'] = float(z)
            except Exception:
                raise ValueError("z must be a float")

        if extra is not None:
            if not isinstance(extra, list):
                raise ValueError("extra must be a list (e.g., list of tokens/flags)")
            # keep as-is; caller controls the content (e.g., strings)
            target['extra'] = extra

        return target

    def get_atom_indices(self):
        """
        Return a list of all atom indices (the 'idx' field from each atom).
        Example: [1, 2, 3, ...]
        """
        return [a.get('idx') for a in self.atoms]

    def get_atom_by_idx(self, idx):
        """
        Return the atom entry (dict) matching the internal atom 'idx'.
        Args:
            idx (int): atom index to look up
        Returns:
            dict | None: the atom dict if found, else None
        """
        try:
            target = int(idx)
        except Exception:
            return None
        for a in self.atoms:
            if isinstance(a, dict) and a.get('idx') == target:
                return a
        return None

    def add_virtual_atom(self, x, y, z, extra=None):
        """
        Add a virtual atom (element='*') to the molblock.
        Returns the index of the new virtual atom.

        Example:
            v_idx = molblock_obj.add_virtual_atom(1.0, 2.0, 0.0)
        """
        return self.add_atom("*", x, y, z, extra)

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

    def get_virtual_atom_indices(self):
        """
        Return a list of indices (idx) for all virtual atoms (element == '*').
        Example: [7, 12]
        """
        return [atom.get('idx') for atom in self.atoms if atom.get('element') == '*']
    
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

    def remove_bond(self, idx):
        """
        Remove a bond by its index.

        Example:
            molblock_obj.remove_bond(b_idx)
        """
        self.bonds = [b for b in self.bonds if b['idx'] != idx]
        self.renumber_bonds()

    def modify_bond(self, idx, type=None, atom1=None, atom2=None, endpts=None, attach=None, extra=None):
        """
        Modify properties of a bond by its index.
        Only non-None arguments will be updated with minimal type validation.
        Supported fields: type(int), atom1(int), atom2(int), endpts(list[int]), attach(str), extra(str).
        """
        # find target bond
        target = None
        for b in self.bonds:
            if b.get('idx') == idx:
                target = b
                break
        if target is None:
            return None

        # minimal validation & updates
        if type is not None:
            if not isinstance(type, int):
                try:
                    type = int(type)
                except Exception:
                    raise ValueError("type must be an integer")
            target['type'] = type

        if atom1 is not None:
            if not isinstance(atom1, int):
                try:
                    atom1 = int(atom1)
                except Exception:
                    raise ValueError("atom1 must be an integer")
            target['atom1'] = atom1

        if atom2 is not None:
            if not isinstance(atom2, int):
                try:
                    atom2 = int(atom2)
                except Exception:
                    raise ValueError("atom2 must be an integer")
            target['atom2'] = atom2

        if endpts is not None:
            if not isinstance(endpts, list) or not all(isinstance(e, int) for e in endpts):
                # allow coercion to list[int] where possible
                try:
                    if not isinstance(endpts, list):
                        endpts = list(endpts)
                    endpts = [int(e) for e in endpts]
                except Exception:
                    raise ValueError("endpts must be a list of integers (including the first count element)")
            target['endpts'] = endpts

        # If attach is provided, ensure we have endpts either already present or provided now
        if attach is not None:
            if not isinstance(attach, str):
                attach = str(attach)
            if endpts is None and 'endpts' not in target:
                raise ValueError("ATTACH provided but ENDPTS is missing for this bond")
            target['attach'] = attach

        if extra is not None:
            if not isinstance(extra, str):
                extra = str(extra)
            target['extra'] = extra

        return target

    def get_bond_indices(self):
        """
        Return a list of all bond indices (the 'idx' field from each bond).
        Example: [1, 2, 3, ...]
        """
        return [b.get('idx') for b in self.bonds]

    def get_bond_by_idx(self, idx):
        """
        Return the bond entry (dict) matching the internal bond 'idx'.
        Args:
            idx (int): bond index to look up
        Returns:
            dict | None: the bond dict if found, else None
        """
        try:
            target = int(idx)
        except Exception:
            return None
        for b in self.bonds:
            if isinstance(b, dict) and b.get('idx') == target:
                return b
        return None

    def add_linknode(self, minrep, maxrep, nbonds, atoms):
        """
        Add a LINKNODE entry using structured parameters, consistent with self.linknodes storage.
        Args:
            minrep (int): minimal repetitions
            maxrep (int): maximal repetitions
            nbonds (int): number of bond pairs defined
            atoms (Sequence[int]): flat list of atom indices [in1, out1, in2, out2, ...]
        Returns:
            dict: the structured LINKNODE dict added (with auto 'idx').
        """
        m = int(minrep)
        M = int(maxrep)
        n = int(nbonds)
        # strict validation for atoms
        if not isinstance(atoms, list):
            raise ValueError("atoms must be provided as a list of integers")
        atoms = [int(v) for v in atoms]

        ln = {
            'idx': len(self.linknodes) + 1,
            'minrep': m,
            'maxrep': M,
            'nbonds': n,
            'atoms': atoms,
        }
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

    def modify_linknode(self, idx, minrep=None, maxrep=None, nbonds=None, atoms=None):
        """
        Modify properties of a LINKNODE entry by its internal idx.
        Only non-None arguments will be updated.
        Args:
            idx (int): internal 'idx' of the LINKNODE to modify
            minrep (int|None): new minrep if provided
            maxrep (int|None): new maxrep if provided
            nbonds (int|None): new nbonds if provided
            atoms (list[int]|None): new atoms list if provided
        Returns:
            dict | None: the updated LINKNODE dict, or None if not found
        """
        target = self.get_linknode_by_idx(idx)
        if not target:
            return None

        if minrep is not None:
            if not isinstance(minrep, int):
                raise ValueError("minrep must be an integer")
            target['minrep'] = minrep
        if maxrep is not None:
            if not isinstance(maxrep, int):
                raise ValueError("maxrep must be an integer")
            target['maxrep'] = maxrep
        if nbonds is not None:
            if not isinstance(nbonds, int):
                raise ValueError("nbonds must be an integer")
            target['nbonds'] = nbonds
        if atoms is not None:
            if not isinstance(atoms, list) or not all(isinstance(a, int) for a in atoms):
                raise ValueError("atoms must be a list of integers")
            target['atoms'] = atoms

        return target

    def get_linknodes_list(self):
        """
        Return a shallow copy of the list of all LINKNODE entries (structured dicts).
        """
        return self.linknodes.copy()

    def get_linknode_count(self):
        """
        Return the number of LINKNODE entries currently stored.
        This is equivalent to the number of linknode entries parsed or added.
        """
        return len(self.linknodes)

    def get_linknode_by_idx(self, idx):
        """
        Return the LINKNODE entry (structured dict) matching the internal idx.
        Args:
            idx (int): internal 'idx' of the LINKNODE
        Returns:
            dict | None: the LINKNODE dict if found, else None
        """
        try:
            target = int(idx)
        except Exception:
            return None
        for ln in self.linknodes:
            if isinstance(ln, dict) and ln.get('idx') == target:
                return ln
        return None

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
        Updates each atom's 'idx', and also remaps any references to the old
        indices inside bonds (atom1/atom2 and ENDPTS) and LINKNODE atoms.
        """
        # Build mapping from old index -> new index **before** mutating atoms
        old_order = [atom['idx'] for atom in self.atoms]
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(old_order, start=start)}

        # Update atoms themselves
        for atom in self.atoms:
            old_idx = atom['idx']
            atom['idx'] = mapping.get(old_idx, old_idx)

        # Update bonds using helpers; also remap ENDPTS if present
        for bidx in self.get_bond_indices():
            b = self.get_bond_by_idx(bidx)
            if not b:
                continue
            new_a1 = mapping.get(b.get('atom1'), b.get('atom1'))
            new_a2 = mapping.get(b.get('atom2'), b.get('atom2'))

            kwargs = {}
            if new_a1 != b.get('atom1'):
                kwargs['atom1'] = new_a1
            if new_a2 != b.get('atom2'):
                kwargs['atom2'] = new_a2

            # Remap ENDPTS list (keep first element as the count, map the rest)
            if isinstance(b.get('endpts'), list) and b['endpts']:
                ep = b['endpts']
                if isinstance(ep[0], int):
                    new_ep = [ep[0]] + [mapping.get(v, v) for v in ep[1:]]
                else:
                    # Fallback: map all entries if first is not count-int
                    new_ep = [mapping.get(v, v) for v in ep]
                # Only set if changed
                if new_ep != ep:
                    kwargs['endpts'] = new_ep

            if kwargs:
                self.modify_bond(bidx, **kwargs)

        # Update LINKNODE atoms using helpers
        ln_count = self.get_linknode_count()
        for ln_idx in range(1, ln_count + 1):
            ln = self.get_linknode_by_idx(ln_idx)
            if not ln or 'atoms' not in ln or not isinstance(ln['atoms'], list):
                continue
            new_atoms = [mapping.get(a, a) for a in ln['atoms']]
            if new_atoms != ln['atoms']:
                self.modify_linknode(ln_idx, atoms=new_atoms)

    def renumber_all(self, atom_start: int = 1, bond_start: int = 1):
        """
        Renumber both atoms and bonds sequentially.
        Calls renumber_atoms() and renumber_bonds() in order.
        """
        self.renumber_atoms(start=atom_start)
        self.renumber_bonds(start=bond_start)

    # --- Output interfaces ---
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
            opts = []
            # Serialize ENDPTS if present (keep first element as count)
            if 'endpts' in bond and isinstance(bond['endpts'], list) and bond['endpts']:
                endpts_str = ' '.join(str(x) for x in bond['endpts'])
                opts.append(f"ENDPTS=({endpts_str})")
            # Serialize ATTACH if present
            if 'attach' in bond and isinstance(bond['attach'], str) and bond['attach']:
                opts.append(f"ATTACH={bond['attach']}")
            # Serialize EXTRA if present (already a string)
            if 'extra' in bond and bond['extra']:
                opts.append(str(bond['extra']))
            if opts:
                bond_line += ' ' + ' '.join(opts)
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