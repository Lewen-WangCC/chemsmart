import logging
import multiprocessing
import os
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Optional

from chemsmart.io.molecules.structure import Molecule
from chemsmart.jobs.iterate.iterate import (
    IterateAnalyzer,
    SkeletonPreprocessor,
    SubstituentPreprocessor,
)
from chemsmart.jobs.runner import JobRunner

if TYPE_CHECKING:
    from chemsmart.jobs.iterate.job import IterateJob

logger = logging.getLogger(__name__)

# Default timeout for each combination worker (in seconds)
DEFAULT_WORKER_TIMEOUT = 120  # 2 minutes


@dataclass
class IterateMoleculePool:
    """Shared pool of unique Molecule objects for iterate combinations.

    Avoids duplicating Molecule objects across combinations. Each combination
    references molecules by index; actual copies are made only at worker
    dispatch time.

    Attributes
    ----------
    skeletons : list[Molecule]
        Pool of unique skeleton molecules.
    substituents : list[Molecule]
        Pool of unique substituent molecules.
    """

    skeletons: list[Molecule] = field(default_factory=list)
    substituents: list[Molecule] = field(default_factory=list)


@dataclass
class IterateCombination:
    """A single-site iterate combination: one skeleton + one substituent at one position.

    In simple mode, each combination represents attaching a single substituent
    to a single link position on the skeleton. The runner generates one
    IterateCombination per (skeleton, skeleton_link_index, substituent) triple.

    Molecules are stored in a shared IterateMoleculePool and referenced by index.
    Actual Molecule copies are created only at worker dispatch time.

    Attributes
    ----------
    skeleton_idx : int
        Index into IterateMoleculePool.skeletons.
    skeleton_label : str
        Human-readable label for the skeleton (e.g. "benzene").
    skeleton_link_index : int
        1-based atom index on the skeleton where the substituent will be
        attached. The old substituent group at this position is removed
        during preprocessing.
    skeleton_indices : list[int] or None
        1-based atom indices that define which atoms belong to the skeleton
        core. Used by SkeletonPreprocessor to determine what to keep.
        None means all atoms are considered skeleton atoms.
    substituent_idx : int
        Index into IterateMoleculePool.substituents.
    substituent_label : str
        Human-readable label for the substituent (e.g. "methyl").
    substituent_link_index : int
        1-based atom index on the substituent that bonds to the skeleton.
    method : str
        Optimization method for orientation search.
        "lagrange_multipliers" (default) or "etkdg".
    sphere_direction_samples_num : int
        Number of sphere direction samples for orientation search (default 96).
    axial_rotations_sample_num : int
        Number of axial rotation samples per direction (default 6).
        Total candidates = sphere_direction_samples_num × axial_rotations_sample_num.
    """

    skeleton_idx: int
    skeleton_label: str
    skeleton_link_index: int  # 1-based
    skeleton_indices: Optional[list[int]]  # 1-based, or None
    substituent_idx: int
    substituent_label: str
    substituent_link_index: int  # 1-based
    method: str = "lagrange_multipliers"
    sphere_direction_samples_num: int = 96
    axial_rotations_sample_num: int = 6

    @property
    def label(self) -> str:
        """
        Generate a unique label for this combination.

        Format: {skeleton_label}_{skeleton_link_index}
        _{substituent_label}_{substituent_link_index}
        Example: benzene_5_methyl_1
        """
        return f"{self.skeleton_label}_{self.skeleton_link_index}_{self.substituent_label}_{self.substituent_link_index}"


def _run_combination_task(
    combination: IterateCombination,
    pool: IterateMoleculePool,
) -> tuple[str, Optional[Molecule]]:
    """
    Core logic to run a single combination.

    Parameters
    ----------
    combination : IterateCombination
        The combination to process (lightweight, references pool by index)
    pool : IterateMoleculePool
        Shared molecule pool; molecules are copied here at execution time

    Returns
    -------
    tuple[str, Molecule | None]
        (label, result_molecule) or (label, None) if failed
    """
    label = combination.label

    try:
        # Copy molecules from pool at execution time
        skeleton = pool.skeletons[combination.skeleton_idx].copy()
        substituent = pool.substituents[combination.substituent_idx].copy()

        # Preprocess skeleton if needed
        skeleton_preprocessor = SkeletonPreprocessor(
            molecule=skeleton,
            link_index=combination.skeleton_link_index,
            skeleton_indices=combination.skeleton_indices,
        )
        processed_skeleton = skeleton_preprocessor.run()
        new_skeleton_link_index = skeleton_preprocessor.get_new_link_index()

        # Preprocess substituent if needed
        substituent_preprocessor = SubstituentPreprocessor(
            molecule=substituent,
            link_index=combination.substituent_link_index,
        )
        processed_substituent = substituent_preprocessor.run()
        new_substituent_link_index = (
            substituent_preprocessor.get_new_link_index()
        )

        # Run analysis to generate combined molecule
        analyzer = IterateAnalyzer(
            skeleton=processed_skeleton,
            substituent=processed_substituent,
            skeleton_link_index=new_skeleton_link_index,
            substituent_link_index=new_substituent_link_index,
            method=combination.method,
            sphere_direction_samples_num=combination.sphere_direction_samples_num,
            axial_rotations_sample_num=combination.axial_rotations_sample_num,
        )

        result = analyzer.run()

        if result is not None:
            logger.info(f"Generated molecule for {label}")
        else:
            logger.warning(f"Failed to generate molecule for {label}")

        return (label, result)

    except Exception as e:
        logger.error(f"Error in task for {label}: {e}")
        return (label, None)


def _run_combination_worker(
    combination: IterateCombination,
    pool: IterateMoleculePool,
    result_queue: "multiprocessing.Queue",
) -> None:
    """
    Worker function for multiprocessing.Process.

    Parameters
    ----------
    combination : IterateCombination
        The combination to process
    pool : IterateMoleculePool
        Shared molecule pool
    result_queue : multiprocessing.Queue
        Queue to put the result tuple
    """
    try:
        result_pair = _run_combination_task(combination, pool)
        result_queue.put(result_pair)
    except Exception as e:
        logger.error(f"Worker process panic for {combination.label}: {e}")
        result_queue.put((combination.label, None))


@dataclass
class MultiSiteAssignment:
    """One slot's resolved assignment in a multi-site combination.

    Represents which substituent is attached at which skeleton position for
    a single slot. A MultiSiteCombination contains multiple MultiSiteAssignment
    instances — one per slot — to fully describe a multi-site substitution.

    Attributes
    ----------
    substituent : Molecule
        The substituent molecule to attach at this slot.
    substituent_label : str
        Human-readable label for the substituent (e.g. "methyl").
    substituent_link_index : int
        1-based atom index on the substituent that bonds to the skeleton.
    skeleton_link_index : int
        1-based atom index on the **original** skeleton where this
        substituent will be attached. Mapped to the preprocessed skeleton
        index via index_map during execution.
    """

    substituent: Molecule
    substituent_label: str
    substituent_link_index: int  # 1-based
    skeleton_link_index: int  # 1-based, on original skeleton


@dataclass
class MultiSiteCombination:
    """A multi-site iterate combination: one skeleton with simultaneous substitutions.

    In multi-site mode, multiple substituents are attached to different
    positions on the same skeleton in one pass. This dataclass holds the
    skeleton together with all slot assignments. During execution:

    1. Batch preprocessing removes old groups at ALL assignment positions.
    2. Substituents are sequentially inserted (order: descending link_index).
    3. Each insertion uses IterateAnalyzer; the result becomes the new
       skeleton for the next insertion.

    Attributes
    ----------
    skeleton : Molecule
        The skeleton molecule (a deep copy for this combination).
    skeleton_label : str
        Human-readable label for the skeleton.
    skeleton_indices : list[int] or None
        1-based atom indices defining the skeleton core. None means all
        atoms are skeleton atoms.
    assignments : list[MultiSiteAssignment]
        One assignment per slot, each specifying a substituent and the
        skeleton position it occupies. No two assignments may share the
        same skeleton_link_index (enforced during combination generation).
    method : str
        Optimization method for orientation search.
        "lagrange_multipliers" (default) or "etkdg".
    sphere_direction_samples_num : int
        Number of sphere direction samples for orientation search (default 96).
    axial_rotations_sample_num : int
        Number of axial rotation samples per direction (default 6).
        Total candidates = sphere_direction_samples_num × axial_rotations_sample_num.
    """

    skeleton: Molecule
    skeleton_label: str
    skeleton_indices: Optional[list[int]]  # 1-based, or None
    assignments: list[MultiSiteAssignment] = field(default_factory=list)
    method: str = "lagrange_multipliers"
    sphere_direction_samples_num: int = 96
    axial_rotations_sample_num: int = 6

    @property
    def label(self) -> str:
        parts = [self.skeleton_label]
        for a in sorted(self.assignments, key=lambda x: x.skeleton_link_index):
            parts.append(f"{a.skeleton_link_index}{a.substituent_label}")
        return "_".join(parts)


def _batch_preprocess_skeleton(
    skeleton: Molecule,
    link_indices: list[int],
    skeleton_indices: Optional[list[int]],
) -> tuple[Molecule, dict[int, int]]:
    """
    Remove old substituent groups at multiple link positions in one pass.

    Parameters
    ----------
    skeleton : Molecule
        Original skeleton molecule
    link_indices : list[int]
        All 1-based link indices to preprocess
    skeleton_indices : list[int] or None
        1-based skeleton atom indices (for SkeletonPreprocessor)

    Returns
    -------
    tuple[Molecule, dict[int, int]]
        (processed skeleton, mapping from original 1-based index to new 1-based index)
    """
    import numpy as np

    all_remove_indices = set()  # 0-based

    for link_idx in link_indices:
        prep = SkeletonPreprocessor(skeleton, link_idx, skeleton_indices)
        if not prep._has_available_bonding_position():
            removed = prep.detect_substituent()  # 0-based indices
            all_remove_indices.update(removed)

    keep_indices = sorted(set(range(len(skeleton))) - all_remove_indices)

    if not keep_indices:
        raise ValueError("All skeleton atoms would be removed during preprocessing.")

    # Verify all link atoms are kept
    for link_idx in link_indices:
        if (link_idx - 1) not in keep_indices:
            raise ValueError(
                f"Link atom at index {link_idx} (1-based) was removed during "
                f"batch preprocessing."
            )

    # Build processed skeleton
    symbols = [skeleton.chemical_symbols[i] for i in keep_indices]
    positions = skeleton.positions[keep_indices]
    frozen = None
    if skeleton.frozen_atoms is not None:
        frozen = [skeleton.frozen_atoms[i] for i in keep_indices]

    processed = Molecule(
        symbols=symbols,
        positions=positions,
        charge=skeleton.charge,
        multiplicity=skeleton.multiplicity,
        frozen_atoms=frozen,
    )

    # Build index map: original 1-based -> new 1-based
    index_map = {}
    for new_0, orig_0 in enumerate(keep_indices):
        index_map[orig_0 + 1] = new_0 + 1

    return processed, index_map


def _run_multi_site_combination_task(
    combination: MultiSiteCombination,
) -> tuple[str, Optional[Molecule]]:
    """
    Execute a multi-site combination: batch-preprocess skeleton,
    then sequentially insert substituents.

    Parameters
    ----------
    combination : MultiSiteCombination
        The multi-site combination to process

    Returns
    -------
    tuple[str, Molecule | None]
        (label, result_molecule) or (label, None) if failed
    """
    label = combination.label

    try:
        # Collect all link indices used in this combination
        all_link_indices = [a.skeleton_link_index for a in combination.assignments]

        # Step 1: Batch preprocess — remove old groups at all link positions
        processed_skeleton, index_map = _batch_preprocess_skeleton(
            combination.skeleton,
            all_link_indices,
            combination.skeleton_indices,
        )

        # Step 2: Sequential insertion
        # Order doesn't affect correctness since _combine_molecules appends
        # substituent after skeleton, preserving skeleton indices.
        # Sort by link_index descending for determinism.
        sorted_assignments = sorted(
            combination.assignments,
            key=lambda a: a.skeleton_link_index,
            reverse=True,
        )

        current_skeleton = processed_skeleton

        for assignment in sorted_assignments:
            # Map original link index to current skeleton index
            new_skel_link = index_map[assignment.skeleton_link_index]

            # Preprocess substituent
            sub_prep = SubstituentPreprocessor(
                molecule=assignment.substituent,
                link_index=assignment.substituent_link_index,
            )
            processed_sub = sub_prep.run()
            new_sub_link = sub_prep.get_new_link_index()

            # Run IterateAnalyzer
            analyzer = IterateAnalyzer(
                skeleton=current_skeleton,
                substituent=processed_sub,
                skeleton_link_index=new_skel_link,
                substituent_link_index=new_sub_link,
                method=combination.method,
                sphere_direction_samples_num=combination.sphere_direction_samples_num,
                axial_rotations_sample_num=combination.axial_rotations_sample_num,
            )
            result = analyzer.run()

            if result is None:
                logger.warning(
                    f"Failed at assignment {assignment.substituent_label}"
                    f"@{assignment.skeleton_link_index} in {label}"
                )
                return (label, None)

            # Use result as new skeleton for next insertion.
            # Skeleton atoms are at positions 0..n_skel-1 in the combined result,
            # so index_map values remain valid for remaining assignments.
            current_skeleton = result

        logger.info(f"Generated multi-site molecule for {label}")
        return (label, current_skeleton)

    except Exception as e:
        logger.error(f"Error in multi-site task for {label}: {e}")
        return (label, None)


def _run_multi_site_combination_worker(
    combination: MultiSiteCombination,
    result_queue: "multiprocessing.Queue",
) -> None:
    """Worker function for multiprocessing a multi-site combination."""
    try:
        result_pair = _run_multi_site_combination_task(combination)
        result_queue.put(result_pair)
    except Exception as e:
        logger.error(f"Worker process panic for {combination.label}: {e}")
        result_queue.put((combination.label, None))


class IterateJobRunner(JobRunner):
    """
    Job runner for Iterate jobs.

    Iterate jobs are special in that they don't call external programs.
    They run purely in Python to generate molecular structures.

    This runner handles the execution of IterateCombination tasks,
    including multiprocessing support via run_combinations().
    """

    JOBTYPES = ["iterate"]
    PROGRAM = "Iterate"
    FAKE = False
    SCRATCH = False

    def __init__(
        self, server=None, scratch=None, fake=False, scratch_dir=None, **kwargs
    ):
        if scratch is None:
            scratch = self.SCRATCH
        super().__init__(
            server=server,
            scratch=scratch,
            scratch_dir=scratch_dir,
            fake=fake,
            **kwargs,
        )
        logger.debug("IterateJobRunner initialized")
        logger.debug(f"Jobrunner server: {self.server}")
        logger.debug(f"Jobrunner scratch: {self.scratch}")
        logger.debug(f"Jobrunner fake mode: {self.fake}")

    @property
    def executable(self):
        """Iterate jobs don't use an external executable."""
        return None

    def _get_command(self, job):
        """Iterate jobs don't need a command - they run in Python."""
        return None

    def run_single(
        self, combination: IterateCombination, pool: IterateMoleculePool
    ) -> tuple[str, Optional[Molecule]]:
        """
        Run a single combination to generate a combined molecule.

        Parameters
        ----------
        combination : IterateCombination
            The combination to process
        pool : IterateMoleculePool
            Shared molecule pool

        Returns
        -------
        tuple[str, Molecule | None]
            (label, result_molecule) or (label, None) if failed
        """
        if self.fake:
            logger.info(
                f"Fake mode enabled, not actually running {combination.label}."
            )
            return (combination.label, None)

        return _run_combination_task(combination, pool)

    def run_combinations(
        self,
        pool: IterateMoleculePool,
        combinations: list[IterateCombination],
        nprocs: int = 1,
        timeout: float = DEFAULT_WORKER_TIMEOUT,
    ) -> list[tuple[str, Optional[Molecule]]]:
        """
        Run multiple combinations using multiprocessing.Process with watchdog.

        This implementation manually manages processes to ensure they can be
        forcefully killed (terminate/kill) if they exceed the timeout.
        ProcessPoolExecutor does not support killing individual tasks/processes
        mid-execution easily.

        Parameters
        ----------
        pool : IterateMoleculePool
            Shared molecule pool; workers copy molecules at execution time
        combinations : list[IterateCombination]
            List of combinations to process
        nprocs : int
            Number of processes for parallel execution. Default 1.
        timeout : float
            Timeout in seconds for each worker. Default 120 (2 minutes).

        Returns
        -------
        list[tuple[str, Molecule | None]]
            List of (label, result_molecule) tuples
        """
        if self.fake:
            logger.info(
                "Fake mode enabled, not actually running combinations."
            )
            return [(c.label, None) for c in combinations]

        if not combinations:
            logger.warning("No combinations to process.")
            return []

        logger.info(
            f"Running {len(combinations)} combination(s) with {nprocs} process(es) (manual management), timeout={timeout}s"
        )

        results_dict: dict[str, Optional[Molecule]] = {}
        failed_labels: list[str] = []
        timed_out_labels: list[str] = []

        max_workers = 1 if nprocs == 1 else nprocs

        # Use Manager Queue for IPC
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        try:
            # State tracking
            pending_combinations = deque(combinations)
            # process_id -> (process, combination, start_time)
            # Use id(p) instead of p.pid as key
            # to avoid potential None pid issues
            active_processes: dict[
                int, tuple[multiprocessing.Process, IterateCombination, float]
            ] = {}

            while pending_combinations or active_processes:
                # 1. Fill up empty slots
                while (
                    pending_combinations
                    and len(active_processes) < max_workers
                ):
                    comb = pending_combinations.popleft()
                    p = multiprocessing.Process(
                        target=_run_combination_worker,
                        args=(comb, pool, result_queue),
                        daemon=True,
                    )
                    p.start()
                    active_processes[id(p)] = (p, comb, time.time())

                # 2. Check for results in queue
                while True:
                    try:
                        # Grab all available data
                        lbl, mol = result_queue.get_nowait()
                        results_dict[lbl] = mol
                    except queue.Empty:
                        break
                    except Exception:
                        break

                # 3. Monitor running processes
                current_time = time.time()
                pids_to_remove = []

                for proc_id, (p, comb, start_time) in active_processes.items():
                    if not p.is_alive():
                        # Process finished naturally (or crashed).
                        # Result should be in queue (handled
                        # above or next loop before exit)
                        p.join()
                        pids_to_remove.append(proc_id)
                    else:
                        # Check timeout
                        if (current_time - start_time) > timeout:
                            logger.warning(
                                f"Timeout ({timeout}s) for {comb.label} (pid {p.pid}). Terminating..."
                            )
                            p.terminate()
                            # Give it a tiny bit to terminate gracefully-ish
                            p.join(timeout=0.5)
                            if p.is_alive():
                                logger.warning(
                                    f"Process {p.pid} stuck, killing..."
                                )
                                p.kill()  # SIGKILL
                                # Must join after kill to reap zombie process
                                p.join(timeout=1.0)

                            timed_out_labels.append(comb.label)
                            results_dict[comb.label] = (
                                None  # Explicitly mark as failed/timeout
                            )
                            pids_to_remove.append(proc_id)

                # 4. Cleanup removed processes
                for proc_id in pids_to_remove:
                    del active_processes[proc_id]

                # 5. Sleep briefly to yield CPU
                if active_processes:
                    time.sleep(0.1)

            # Double check queue one last time just in case
            while True:
                try:
                    lbl, mol = result_queue.get_nowait()
                    results_dict[lbl] = mol
                except queue.Empty:
                    break
                except Exception:
                    break

        finally:
            # Properly shutdown the Manager to avoid resource leaks
            manager.shutdown()

        # Check for missing results (crashes that didn't write to queue)
        for comb in combinations:
            if comb.label not in results_dict:
                logger.error(
                    f"No result found for {comb.label} - assuming worker crash."
                )
                failed_labels.append(comb.label)
                results_dict[comb.label] = None
            elif (
                results_dict[comb.label] is None
                and comb.label not in timed_out_labels
            ):
                # It might have returned None explicitly (handled in worker)
                if comb.label not in failed_labels:
                    failed_labels.append(comb.label)

        # Build results list in original order
        results = [
            (comb.label, results_dict.get(comb.label)) for comb in combinations
        ]

        # Log results summary
        successful_labels = [
            label for label, mol in results if mol is not None
        ]
        logger.info(
            f"Completed: {len(successful_labels)}/{len(results)} molecules generated successfully"
        )

        # Print summary of results
        if successful_labels or timed_out_labels or failed_labels:
            logger.info("=" * 40)
            logger.info("       SUMMARY OF RESULTS")
            logger.info("=" * 40)

            if successful_labels:
                logger.info(f"Successful ({len(successful_labels)}):")
                for label in successful_labels:
                    logger.info(f"  - {label}")

            if timed_out_labels:
                logger.warning(f"Timed out ({len(timed_out_labels)}):")
                for label in timed_out_labels:
                    logger.warning(f"  - {label}")

            if failed_labels:
                logger.warning(
                    f"Failed to find solution ({len(failed_labels)}):"
                )
                for label in failed_labels:
                    logger.warning(f"  - {label}")

            logger.info("=" * 40)

        return results

    def _load_molecule(
        self, mol_config: dict, mol_type: str, idx: int
    ) -> tuple[Optional[Molecule], str]:
        """
        Load a Molecule from configuration dict.

        Parameters
        ----------
        mol_config : dict
            Configuration containing file_path
        mol_type : str
            "skeleton" or "substituent" for logging
        idx : int
            Index for logging

        Returns
        -------
        tuple[Molecule | None, str]
            (Loaded molecule or None if loading fails, label)
        """
        label = mol_config.get("label") or f"{mol_type}{idx + 1}"
        file_path = mol_config.get("file_path")

        try:
            if file_path:
                molecule = Molecule.from_filepath(file_path)
                logger.debug(
                    f"Loaded {mol_type} '{label}' from file: {file_path}"
                )
            else:
                logger.warning(
                    f"{mol_type.capitalize()} '{label}' has no valid source "
                    f"(file_path), skipping."
                )
                return None, label

            # S2 Check: Validate indices against atom count
            num_atoms = molecule.num_atoms

            # Validate link_index
            link_indices = mol_config.get("link_index")
            # Ensure it is a list if not None (normalized
            # in CLI, but for safety in runner)
            if link_indices is not None and not isinstance(link_indices, list):
                if isinstance(link_indices, int):
                    link_indices = [link_indices]
                # If string or other, it should have been caught by
                # CLI, but we assume it might be raw here if skipped CLI

            if link_indices:
                invalid_links = [i for i in link_indices if i > num_atoms]
                if invalid_links:
                    logger.error(
                        f"{mol_type.capitalize()} '{label}': link_index {invalid_links} "
                        f"out of bounds. Molecule has {num_atoms} atoms."
                    )
                    return None, label

            # Validate skeleton_indices (only for skeletons)
            if mol_type == "skeleton":
                skel_indices = mol_config.get("skeleton_indices")
                # Ensure list
                if skel_indices is not None and not isinstance(
                    skel_indices, list
                ):
                    if isinstance(skel_indices, int):
                        skel_indices = [skel_indices]

                if skel_indices:
                    invalid_skels = [i for i in skel_indices if i > num_atoms]
                    if invalid_skels:
                        logger.error(
                            f"{mol_type.capitalize()} '{label}': skeleton_indices {invalid_skels} "
                            f"out of bounds. Molecule has {num_atoms} atoms."
                        )
                        return None, label

            return molecule, label
        except Exception as e:
            logger.error(f"Failed to load {mol_type} '{label}': {e}")
            return None, label

    def _generate_combinations(
        self, job: "IterateJob"
    ) -> tuple[IterateMoleculePool, list[IterateCombination]]:
        """
        Generate all combinations of (skeleton,
        skeleton_link_index, substituent).

        Molecules are loaded once into a shared IterateMoleculePool. Combinations
        reference molecules by index, avoiding redundant copies.

        Parameters
        ----------
        job : IterateJob
            The job containing settings with skeleton_list and substituent_list

        Returns
        -------
        tuple[IterateMoleculePool, list[IterateCombination]]
            Shared molecule pool and list of all valid combinations
        """
        pool = IterateMoleculePool()
        combinations = []

        skeleton_list = job.settings.skeleton_list or []
        substituent_list = job.settings.substituent_list or []
        method = job.settings.method
        sphere_direction_samples_num = (
            job.settings.sphere_direction_samples_num
        )
        axial_rotations_sample_num = job.settings.axial_rotations_sample_num

        valid_skeletons = []
        for skel_idx, skel_config in enumerate(skeleton_list):
            skeleton, skel_label = self._load_molecule(
                skel_config, "skeleton", skel_idx
            )
            if skeleton is None:
                continue

            skel_link_indices = skel_config.get(
                "link_index"
            )  # list[int], 1-based

            if not skel_link_indices:
                logger.warning(
                    f"Skeleton '{skel_label}' has no link_index, skipping."
                )
                continue

            pool_idx = len(pool.skeletons)
            pool.skeletons.append(skeleton)
            valid_skeletons.append((pool_idx, skel_label, skel_config))

        valid_substituents = []
        for sub_idx, sub_config in enumerate(substituent_list):
            substituent, sub_label = self._load_molecule(
                sub_config, "substituent", sub_idx
            )
            if substituent is None:
                continue

            sub_link_index = sub_config.get("link_index")  # list[int], 1-based

            if not sub_link_index:
                logger.warning(
                    f"Substituent '{sub_label}' has no link_index, skipping."
                )
                continue

            pool_idx = len(pool.substituents)
            pool.substituents.append(substituent)
            valid_substituents.append((pool_idx, sub_label, sub_config))

        for skel_pool_idx, skel_label, skel_config in valid_skeletons:
            skel_link_indices = skel_config.get(
                "link_index"
            )  # list[int], 1-based
            skeleton_indices = skel_config.get(
                "skeleton_indices"
            )  # list[int] or None

            for sub_pool_idx, sub_label, sub_config in valid_substituents:
                sub_link_index = sub_config.get(
                    "link_index"
                )  # list[int], 1-based

                # Use the first substituent link index
                sub_link_idx = sub_link_index[0]

                # For each skeleton link position, create a combination
                for skel_link_idx in skel_link_indices:
                    combination = IterateCombination(
                        skeleton_idx=skel_pool_idx,
                        skeleton_label=skel_label,
                        skeleton_link_index=skel_link_idx,
                        skeleton_indices=skeleton_indices,
                        substituent_idx=sub_pool_idx,
                        substituent_label=sub_label,
                        sphere_direction_samples_num=sphere_direction_samples_num,
                        axial_rotations_sample_num=axial_rotations_sample_num,
                        substituent_link_index=sub_link_idx,
                        method=method,
                    )
                    combinations.append(combination)
                    logger.info(f"Created combination: {combination.label}")

        return pool, combinations

    def _write_outputs(
        self, results: list[tuple[str, Optional[Molecule]]], job: "IterateJob"
    ) -> None:
        """
        Write execution results to output file(s).

        Parameters
        ----------
        results : list[tuple[str, Molecule | None]]
            List of (label, molecule) tuples from run_combinations
        job : IterateJob
             The job instance with configuration options
        """
        successful_count = 0

        if job.separate_outputs:
            # Separate files mode
            output_dir = job.output_directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Should have been validated by CLI, but fallback
                output_dir = "."

            logger.info(
                f"Writing separate output files to directory: {output_dir}"
            )

            for label, mol in results:
                if mol is not None:
                    # Construct filename: directory + label + .xyz
                    filename = os.path.join(output_dir, f"{label}.xyz")
                    try:
                        with open(filename, "w") as f:
                            f.write(f"{mol.num_atoms}\n")
                            f.write(f"       {label}\n")
                            for symbol, pos in zip(
                                mol.chemical_symbols, mol.positions
                            ):
                                f.write(
                                    f"{symbol:2s}  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}\n"
                                )
                        successful_count += 1
                        logger.debug(f"Wrote {filename}")
                    except Exception as e:
                        logger.error(f"Failed to write {filename}: {e}")

            logger.info(
                f"Wrote {successful_count} separate molecule files to {output_dir}"
            )

        else:
            # Single merged output file mode
            outputfile = job.outputfile
            # Ensure output directory exists
            output_dir = os.path.dirname(outputfile)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(outputfile, "w") as f:
                for label, mol in results:
                    if mol is not None:
                        # Build xyz string manually
                        # First line: number of atoms
                        f.write(f"{mol.num_atoms}\n")
                        # Second line: label as comment
                        f.write(f"       {label}\n")
                        # Following lines: atom symbol and coordinates
                        for symbol, pos in zip(
                            mol.chemical_symbols, mol.positions
                        ):
                            f.write(
                                f"{symbol:2s}  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}\n"
                            )
                        successful_count += 1

            logger.info(
                f"Wrote {successful_count} molecule(s) to {outputfile}"
            )

    def run(self, job: "IterateJob", **kwargs) -> None:
        """
        Run an IterateJob.

        This method handles all execution logic:
        1. Load molecules from job settings
        2. Generate all combinations (simple or multi-site)
        3. Run combinations with multiprocessing
        4. Write results to output xyz file

        Parameters
        ----------
        job : IterateJob
            The iterate job to run
        **kwargs
            Additional keyword arguments
        """
        logger.info(f"IterateJobRunner.run() called for job: {job}")

        if self.fake:
            logger.info("Fake mode enabled, not actually running job.")
            return

        # Detect mode: multi-site or simple
        # Multi-site if any skeleton has embedded 'slots'
        is_multi_site = any(
            skel.get("slots")
            for skel in (job.settings.skeleton_list or [])
        )

        if is_multi_site:
            combinations = self._generate_multi_site_combinations(job)
            logger.info(
                f"Generated {len(combinations)} multi-site combination(s)"
            )

            if not combinations:
                logger.warning("No valid multi-site combinations to process.")
                return

            results = self.run_multi_site_combinations(
                combinations, nprocs=job.nprocs, timeout=job.timeout
            )
        else:
            pool, combinations = self._generate_combinations(job)
            logger.info(f"Generated {len(combinations)} combination(s)")

            if not combinations:
                logger.warning("No valid combinations to process.")
                return

            results = self.run_combinations(
                pool, combinations, nprocs=job.nprocs, timeout=job.timeout
            )

        # Write output results
        self._write_outputs(results, job)

        logger.info("IterateJobRunner completed job")

    def _generate_multi_site_combinations(
        self, job: "IterateJob"
    ) -> list[MultiSiteCombination]:
        """
        Generate all multi-site combinations from embedded slots and groups.

        Algorithm:
        1. For each skeleton with 'slots', find substituents whose 'groups'
           match each slot's group number
        2. Expand each slot into possible assignments
           (matching substituent × link_index)
        3. Cartesian product across all slots within a skeleton
        4. Filter: no two assignments share the same link_index

        Parameters
        ----------
        job : IterateJob
            The job containing settings

        Returns
        -------
        list[MultiSiteCombination]
            List of all valid multi-site combinations
        """
        skeleton_list = job.settings.skeleton_list or []
        substituent_list = job.settings.substituent_list or []
        method = job.settings.method
        sphere_samples = job.settings.sphere_direction_samples_num
        axial_samples = job.settings.axial_rotations_sample_num

        # Load and index substituents by group number
        # group_number -> list of (mol, label, link_index)
        sub_by_group: dict[int, list[tuple]] = {}

        for sub_idx, sub_config in enumerate(substituent_list):
            groups = sub_config.get("groups")
            if not groups:
                continue  # skip substituents without groups

            mol, label = self._load_molecule(
                sub_config, "substituent", sub_idx
            )
            if mol is None:
                continue

            sub_link_idx = sub_config["link_index"][0]  # 1-based

            for g in groups:
                if g not in sub_by_group:
                    sub_by_group[g] = []
                sub_by_group[g].append((mol, label, sub_link_idx))

        combinations = []

        for skel_idx, skel_config in enumerate(skeleton_list):
            slots = skel_config.get("slots")
            if not slots:
                continue  # skip simple mode skeletons

            skel_mol, skel_label = self._load_molecule(
                skel_config, "skeleton", skel_idx
            )
            if skel_mol is None:
                continue

            skeleton_indices = skel_config.get("skeleton_indices")

            # Build options for each slot
            slot_options = []

            for slot in slots:
                group = slot["group"]
                link_indices = slot["link_indices"]

                subs_for_group = sub_by_group.get(group, [])
                if not subs_for_group:
                    logger.warning(
                        f"Skeleton '{skel_label}': no substituents "
                        f"for group {group}, skipping slot."
                    )
                    continue
                
                # Build options for single slot
                options = []
                for sub_mol, sub_label, sub_link_idx in subs_for_group:
                    for link_idx in link_indices:
                        assignment = MultiSiteAssignment(
                            substituent=sub_mol,
                            substituent_label=sub_label,
                            substituent_link_index=sub_link_idx,
                            skeleton_link_index=link_idx,
                        )
                        options.append(assignment)

                if options:
                    slot_options.append(options)

            if not slot_options:
                logger.warning(
                    f"Skeleton '{skel_label}': no valid slot options."
                )
                continue

            # Reorganize by individual skeleton position.
            # Each position independently chooses: None (skip) or one
            # candidate substituent. This produces the power set of all
            # valid assignments, naturally avoiding duplicate positions.
            position_options: dict[int, list[MultiSiteAssignment]] = {}

            for options_list in slot_options:
                for assignment in options_list:
                    pos = assignment.skeleton_link_index
                    if pos not in position_options:
                        position_options[pos] = []
                    # Deduplicate: same substituent at same position
                    # may appear from overlapping groups
                    is_dup = any(
                        existing.substituent_label == assignment.substituent_label
                        and existing.substituent_link_index == assignment.substituent_link_index
                        for existing in position_options[pos]
                    )
                    if not is_dup:
                        position_options[pos].append(assignment)

            if not position_options:
                logger.warning(
                    f"Skeleton '{skel_label}': no valid position options."
                )
                continue

            # For each position: [None (don't fill), candidate1, candidate2, ...]
            sorted_positions = sorted(position_options.keys())
            per_position_choices = []
            for pos in sorted_positions:
                per_position_choices.append([None] + position_options[pos])

            # Cartesian product across all positions
            for combo in product(*per_position_choices):
                assignments = [a for a in combo if a is not None]
                if not assignments:
                    continue  # skip the all-empty case

                combination = MultiSiteCombination(
                    skeleton=skel_mol.copy(),
                    skeleton_label=skel_label,
                    skeleton_indices=skeleton_indices,
                    assignments=[
                        MultiSiteAssignment(
                            substituent=a.substituent.copy(),
                            substituent_label=a.substituent_label,
                            substituent_link_index=a.substituent_link_index,
                            skeleton_link_index=a.skeleton_link_index,
                        )
                        for a in assignments
                    ],
                    method=method,
                    sphere_direction_samples_num=sphere_samples,
                    axial_rotations_sample_num=axial_samples,
                )
                combinations.append(combination)
                logger.info(
                    f"Created multi-site combination: {combination.label}"
                )

        return combinations

    def run_multi_site_combinations(
        self,
        combinations: list[MultiSiteCombination],
        nprocs: int = 1,
        timeout: float = DEFAULT_WORKER_TIMEOUT,
    ) -> list[tuple[str, Optional[Molecule]]]:
        """
        Run multiple multi-site combinations using multiprocessing.

        Same architecture as run_combinations but for MultiSiteCombination.
        """
        if self.fake:
            logger.info(
                "Fake mode enabled, not actually running combinations."
            )
            return [(c.label, None) for c in combinations]

        if not combinations:
            logger.warning("No combinations to process.")
            return []

        logger.info(
            f"Running {len(combinations)} multi-site combination(s) "
            f"with {nprocs} process(es), timeout={timeout}s"
        )

        results_dict: dict[str, Optional[Molecule]] = {}
        failed_labels: list[str] = []
        timed_out_labels: list[str] = []

        max_workers = 1 if nprocs == 1 else nprocs

        manager = multiprocessing.Manager()
        result_queue = manager.Queue()

        try:
            pending_combinations = deque(combinations)
            active_processes: dict[
                int,
                tuple[multiprocessing.Process, MultiSiteCombination, float],
            ] = {}

            while pending_combinations or active_processes:
                while (
                    pending_combinations
                    and len(active_processes) < max_workers
                ):
                    comb = pending_combinations.popleft()
                    p = multiprocessing.Process(
                        target=_run_multi_site_combination_worker,
                        args=(comb, result_queue),
                        daemon=True,
                    )
                    p.start()
                    active_processes[id(p)] = (p, comb, time.time())

                while True:
                    try:
                        lbl, mol = result_queue.get_nowait()
                        results_dict[lbl] = mol
                    except queue.Empty:
                        break
                    except Exception:
                        break

                current_time = time.time()
                pids_to_remove = []

                for proc_id, (
                    p,
                    comb,
                    start_time,
                ) in active_processes.items():
                    if not p.is_alive():
                        p.join()
                        pids_to_remove.append(proc_id)
                    else:
                        if (current_time - start_time) > timeout:
                            logger.warning(
                                f"Timeout ({timeout}s) for {comb.label} "
                                f"(pid {p.pid}). Terminating..."
                            )
                            p.terminate()
                            p.join(timeout=0.5)
                            if p.is_alive():
                                logger.warning(
                                    f"Process {p.pid} stuck, killing..."
                                )
                                p.kill()
                                p.join(timeout=1.0)
                            timed_out_labels.append(comb.label)
                            results_dict[comb.label] = None
                            pids_to_remove.append(proc_id)

                for proc_id in pids_to_remove:
                    del active_processes[proc_id]

                if active_processes:
                    time.sleep(0.1)

            while True:
                try:
                    lbl, mol = result_queue.get_nowait()
                    results_dict[lbl] = mol
                except queue.Empty:
                    break
                except Exception:
                    break

        finally:
            manager.shutdown()

        for comb in combinations:
            if comb.label not in results_dict:
                logger.error(
                    f"No result found for {comb.label} - assuming worker crash."
                )
                failed_labels.append(comb.label)
                results_dict[comb.label] = None
            elif (
                results_dict[comb.label] is None
                and comb.label not in timed_out_labels
            ):
                if comb.label not in failed_labels:
                    failed_labels.append(comb.label)

        results = [
            (comb.label, results_dict.get(comb.label)) for comb in combinations
        ]

        successful_labels = [
            label for label, mol in results if mol is not None
        ]
        logger.info(
            f"Completed: {len(successful_labels)}/{len(results)} "
            f"multi-site molecules generated successfully"
        )

        if successful_labels or timed_out_labels or failed_labels:
            logger.info("=" * 40)
            logger.info("  MULTI-SITE SUMMARY OF RESULTS")
            logger.info("=" * 40)

            if successful_labels:
                logger.info(f"Successful ({len(successful_labels)}):")
                for label in successful_labels:
                    logger.info(f"  - {label}")

            if timed_out_labels:
                logger.warning(f"Timed out ({len(timed_out_labels)}):")
                for label in timed_out_labels:
                    logger.warning(f"  - {label}")

            if failed_labels:
                logger.warning(
                    f"Failed to find solution ({len(failed_labels)}):"
                )
                for label in failed_labels:
                    logger.warning(f"  - {label}")

            logger.info("=" * 40)

        return results
