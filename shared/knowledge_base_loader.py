# shared/knowledge_base_loader.py

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from shared.knowledge_base_models import (
    CommonMistakesRegistry,
    CompatibilityMatrixSchema,
    ForceFieldSchema,
    IonParameterSchema,
    PhaseSchema,
    SimulationPhase,
    WaterModelSchema,
)

logger = logging.getLogger(__name__)


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base cannot be loaded
    or validated."""
    pass


class KnowledgeBase:
    """
    Central access point for all knowledge base data.

    Loads all YAML files at instantiation, validates them
    against Pydantic schemas, and exposes typed accessors
    for each section.

    All data is immutable after loading — the knowledge base
    is read-only at runtime.

    Usage:
        kb = KnowledgeBase(Path("knowledge_base"))
        schema = kb.get_phase_schema(
            SimulationPhase.ENERGY_MINIMIZATION
        )
        ff = kb.get_force_field("amber99sb-ildn")
        mistakes = kb.get_common_mistakes()

    Or via the singleton factory:
        kb = get_knowledge_base("knowledge_base")
    """

    def __init__(self, knowledge_base_dir: Path) -> None:
        self._kb_dir = Path(knowledge_base_dir)
        self._validate_directory_structure()

        logger.info(
            f"Loading knowledge base from {self._kb_dir}"
        )

        self._phases: dict[
            SimulationPhase, PhaseSchema
        ] = {}
        self._force_fields: dict[
            str, ForceFieldSchema
        ] = {}
        self._water_models: dict[
            str, WaterModelSchema
        ] = {}
        self._ion_parameters: dict[
            str, IonParameterSchema
        ] = {}
        self._compatibility_matrix: (
            CompatibilityMatrixSchema
        )
        self._common_mistakes: CommonMistakesRegistry

        self._load_all()
        logger.info(
            "Knowledge base loaded and validated "
            "successfully"
        )

    # ─────────────────────────────────────────
    # Public Accessors
    # ─────────────────────────────────────────

    def get_phase_schema(
        self, phase: SimulationPhase
    ) -> PhaseSchema:
        """
        Return the MDP parameter schema for a simulation
        phase.

        Args:
            phase: One of the SimulationPhase enum values.

        Raises:
            KnowledgeBaseError: If the phase is not found.
        """
        if phase not in self._phases:
            raise KnowledgeBaseError(
                f"No schema found for phase: "
                f"{phase.value}"
            )
        return self._phases[phase]

    def get_force_field(
        self, ff_name: str
    ) -> ForceFieldSchema:
        """
        Return the force field definition by name.
        Lookup is case-insensitive.

        Args:
            ff_name: Force field name, e.g.
                     'amber99sb-ildn', 'charmm36m'.

        Raises:
            KnowledgeBaseError: If the force field is
                not found.
        """
        ff_name_lower = ff_name.lower()
        if ff_name_lower not in self._force_fields:
            raise KnowledgeBaseError(
                f"Unknown force field: {ff_name}. "
                f"Available: "
                f"{list(self._force_fields.keys())}"
            )
        return self._force_fields[ff_name_lower]

    def get_water_model(
        self, water_model_name: str
    ) -> WaterModelSchema:
        """
        Return the water model definition by name.
        Lookup is case-insensitive.

        Args:
            water_model_name: Water model name, e.g.
                              'TIP3P', 'SPCE'.

        Raises:
            KnowledgeBaseError: If the water model is
                not found.
        """
        name_lower = water_model_name.lower()
        if name_lower not in self._water_models:
            raise KnowledgeBaseError(
                f"Unknown water model: "
                f"{water_model_name}. "
                f"Available: "
                f"{list(self._water_models.keys())}"
            )
        return self._water_models[name_lower]

    def get_ion_parameters(
        self, ion_param_name: str
    ) -> IonParameterSchema:
        """
        Return ion parameter set definition by name.
        Lookup is case-insensitive.

        Args:
            ion_param_name: Ion parameter set name, e.g.
                            'Joung-Cheatham', 'Aqvist'.

        Raises:
            KnowledgeBaseError: If the ion parameter set
                is not found.
        """
        name_lower = ion_param_name.lower()
        if name_lower not in self._ion_parameters:
            raise KnowledgeBaseError(
                f"Unknown ion parameter set: "
                f"{ion_param_name}. "
                f"Available: "
                f"{list(self._ion_parameters.keys())}"
            )
        return self._ion_parameters[name_lower]

    def get_compatibility_matrix(
        self,
    ) -> CompatibilityMatrixSchema:
        """Return the full force field compatibility
        matrix."""
        return self._compatibility_matrix

    def get_common_mistakes(
        self,
    ) -> CommonMistakesRegistry:
        """Return the common mistakes registry."""
        return self._common_mistakes

    def list_force_fields(self) -> list[str]:
        """Return names of all known force fields."""
        return list(self._force_fields.keys())

    def list_water_models(self) -> list[str]:
        """Return names of all known water models."""
        return list(self._water_models.keys())

    def list_ion_parameter_sets(self) -> list[str]:
        """Return names of all known ion parameter
        sets."""
        return list(self._ion_parameters.keys())

    def is_combination_forbidden(
        self,
        force_field: str,
        water_model: Optional[str] = None,
        disp_corr: Optional[str] = None,
        vdw_modifier: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a parameter combination is explicitly
        forbidden by the compatibility matrix.

        A combination matches a forbidden entry only if
        ALL specified fields in the forbidden entry match
        the provided arguments. Fields not present in the
        forbidden entry are ignored.

        Args:
            force_field:  Force field name to check.
            water_model:  Water model name (optional).
            disp_corr:    DispCorr value (optional).
            vdw_modifier: VdW modifier value (optional).

        Returns:
            (is_forbidden, reason_message) tuple.
            reason_message is None if not forbidden.
        """
        for forbidden in (
            self._compatibility_matrix
            .forbidden_combinations
        ):
            combo = forbidden.combination
            match = True

            if "force_field" in combo:
                if (
                    combo["force_field"].lower()
                    != force_field.lower()
                ):
                    match = False

            if water_model and "water" in combo:
                if (
                    combo["water"].lower()
                    != water_model.lower()
                ):
                    match = False

            if disp_corr and "DispCorr" in combo:
                if (
                    combo["DispCorr"].lower()
                    != disp_corr.lower()
                ):
                    match = False

            if vdw_modifier and "vdw-modifier" in combo:
                if (
                    combo["vdw-modifier"].lower()
                    != vdw_modifier.lower()
                ):
                    match = False

            if match:
                return True, forbidden.reason

        return False, None

    # ─────────────────────────────────────────
    # Private — Directory Validation
    # ─────────────────────────────────────────

    def _validate_directory_structure(self) -> None:
        """
        Verify all required knowledge base files exist
        before attempting to load any of them.

        Raises:
            KnowledgeBaseError: If any required file is
                missing, listing all missing files at once.
        """
        if not self._kb_dir.exists():
            raise KnowledgeBaseError(
                f"Knowledge base directory does not "
                f"exist: {self._kb_dir}"
            )

        required_files = [
            "phases/energy_minimization.yaml",
            "phases/nvt_equilibration.yaml",
            "phases/npt_equilibration.yaml",
            "phases/production_md.yaml",
            "forcefield_compatibility/force_fields.yaml",
            "forcefield_compatibility/water_models.yaml",
            "forcefield_compatibility/"
            "ion_parameters.yaml",
            "forcefield_compatibility/"
            "compatibility_matrix.yaml",
            "box_solvation_rules/box_geometry.yaml",
            "box_solvation_rules/solvation.yaml",
            "box_solvation_rules/ionization.yaml",
            "box_solvation_rules/validation_checks.yaml",
            "common_mistakes.yaml",
        ]
        missing = [
            f for f in required_files
            if not (self._kb_dir / f).exists()
        ]
        if missing:
            raise KnowledgeBaseError(
                f"Missing knowledge base files: {missing}"
            )

    # ─────────────────────────────────────────
    # Private — YAML Loading
    # ─────────────────────────────────────────

    def _load_yaml(
        self, relative_path: str
    ) -> dict[str, Any]:
        """
        Load and parse a single YAML file.

        Args:
            relative_path: Path relative to kb_dir.

        Returns:
            Parsed YAML content as a dict.

        Raises:
            KnowledgeBaseError: On parse error, empty
                file, or OS error.
        """
        full_path = self._kb_dir / relative_path
        try:
            with open(
                full_path, "r", encoding="utf-8"
            ) as f:
                data = yaml.safe_load(f)
            if data is None:
                raise KnowledgeBaseError(
                    f"Empty YAML file: {full_path}"
                )
            if not isinstance(data, dict):
                raise KnowledgeBaseError(
                    f"YAML file does not contain a "
                    f"mapping at top level: {full_path}"
                )
            return data
        except yaml.YAMLError as e:
            raise KnowledgeBaseError(
                f"YAML parse error in {full_path}: {e}"
            ) from e
        except OSError as e:
            raise KnowledgeBaseError(
                f"Cannot read file {full_path}: {e}"
            ) from e

    # ─────────────────────────────────────────
    # Private — Section Loaders
    # ─────────────────────────────────────────

    def _load_all(self) -> None:
        """
        Load and validate all knowledge base sections
        in dependency order.
        """
        self._load_phases()
        self._load_force_fields()
        self._load_water_models()
        self._load_ion_parameters()
        self._load_compatibility_matrix()
        self._load_common_mistakes()

    def _load_phases(self) -> None:
        """
        Load all four simulation phase schemas.

        Each phase YAML file has a single top-level key
        matching the phase name (e.g. 'energy_minimization')
        that wraps the actual schema content. This key is
        unwrapped before passing to PhaseSchema.

        File structure:
            energy_minimization:    ← root key (unwrapped)
              description: ...
              run_control:
                parameters: ...
              ...
        """
        phase_files: dict[
            SimulationPhase, tuple[str, str]
        ] = {
            SimulationPhase.ENERGY_MINIMIZATION: (
                "phases/energy_minimization.yaml",
                "energy_minimization",
            ),
            SimulationPhase.NVT_EQUILIBRATION: (
                "phases/nvt_equilibration.yaml",
                "nvt_equilibration",
            ),
            SimulationPhase.NPT_EQUILIBRATION: (
                "phases/npt_equilibration.yaml",
                "npt_equilibration",
            ),
            SimulationPhase.PRODUCTION_MD: (
                "phases/production_md.yaml",
                "production_md",
            ),
        }
        for phase, (filepath, root_key) in (
            phase_files.items()
        ):
            raw = self._load_yaml(filepath)
            if root_key not in raw:
                raise KnowledgeBaseError(
                    f"Expected root key '{root_key}' "
                    f"not found in {filepath}. "
                    f"Found keys: {list(raw.keys())}"
                )
            data = raw[root_key]
            try:
                self._phases[phase] = (
                    PhaseSchema(**data)
                )
                logger.debug(
                    f"Loaded phase schema: {phase.value}"
                )
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid schema in {filepath}: {e}"
                ) from e

    def _load_force_fields(self) -> None:
        """
        Load all force field definitions.

        The file groups force fields by family under keys
        such as 'amber_family', 'charmm_family', etc.
        Each family dict contains individual force field
        entries keyed by their short name. All families
        are flattened into a single dict.

        File structure:
            amber_family:           ← family key
              description: ...      ← skipped
              amber99sb-ildn:       ← FF entry
                full_name: ...
              amber14sb:            ← FF entry
                full_name: ...
            charmm_family:          ← family key
              charmm36m:            ← FF entry
                full_name: ...
        """
        raw = self._load_yaml(
            "forcefield_compatibility/force_fields.yaml"
        )
        family_keys = [
            "amber_family",
            "charmm_family",
            "gromos_family",
            "opls_family",
        ]
        # Keys within a family dict that are not FF entries
        non_ff_keys = {"description"}

        for family_key in family_keys:
            if family_key not in raw:
                logger.warning(
                    f"Force field family '{family_key}' "
                    f"not found in force_fields.yaml — "
                    f"skipping."
                )
                continue
            family_data = raw[family_key]
            if not isinstance(family_data, dict):
                continue
            for ff_name, ff_data in (
                family_data.items()
            ):
                if ff_name in non_ff_keys:
                    continue
                if not isinstance(ff_data, dict):
                    continue
                try:
                    self._force_fields[
                        ff_name.lower()
                    ] = ForceFieldSchema(**ff_data)
                    logger.debug(
                        f"Loaded force field: {ff_name}"
                    )
                except ValidationError as e:
                    raise KnowledgeBaseError(
                        f"Invalid force field schema "
                        f"for {ff_name}: {e}"
                    ) from e

    def _load_water_models(self) -> None:
        """
        Load all water model definitions.

        The file groups water models by number of sites
        under keys such as 'three_site_models',
        'four_site_models', 'five_site_models'. Each
        group dict contains individual water model entries.
        All groups are flattened into a single dict.

        File structure:
            three_site_models:      ← group key
              description: ...      ← skipped
              TIP3P:                ← water model entry
                sites: 3
              SPCE:                 ← water model entry
                sites: 3
            four_site_models:       ← group key
              TIP4P-EW:             ← water model entry
                sites: 4
        """
        raw = self._load_yaml(
            "forcefield_compatibility/water_models.yaml"
        )
        model_group_keys = [
            "three_site_models",
            "four_site_models",
            "five_site_models",
        ]
        # Keys within a group dict that are not model
        # entries
        non_model_keys = {
            "water_model_selection_guide",
            "water_structure_files",
            "description",
        }
        for group_key in model_group_keys:
            if group_key not in raw:
                logger.warning(
                    f"Water model group '{group_key}' "
                    f"not found in water_models.yaml — "
                    f"skipping."
                )
                continue
            group_data = raw[group_key]
            if not isinstance(group_data, dict):
                continue
            for model_name, model_data in (
                group_data.items()
            ):
                if model_name in non_model_keys:
                    continue
                if not isinstance(model_data, dict):
                    continue
                try:
                    self._water_models[
                        model_name.lower()
                    ] = WaterModelSchema(**model_data)
                    logger.debug(
                        f"Loaded water model: "
                        f"{model_name}"
                    )
                except ValidationError as e:
                    raise KnowledgeBaseError(
                        f"Invalid water model schema "
                        f"for {model_name}: {e}"
                    ) from e

    def _load_ion_parameters(self) -> None:
        """
        Load all ion parameter set definitions.

        The file wraps all parameter set entries under a
        'parameter_sets' key. Additional top-level keys
        (ion_naming_conventions, ion_concentration_rules,
        etc.) contain reference information used by tools
        but are not loaded as Pydantic models here.

        File structure:
            description: ...        ← skipped
            parameter_sets:         ← wrapper key
              Joung-Cheatham:       ← ion param entry
                reference: ...
              Aqvist:               ← ion param entry
                reference: ...
            ion_naming_conventions: ← skipped
              ...
            ion_concentration_rules: ← skipped
              ...
        """
        raw = self._load_yaml(
            "forcefield_compatibility/ion_parameters.yaml"
        )
        # Top-level keys that are not parameter set entries
        non_param_keys = {
            "description",
            "parameter_sets",
            "ion_naming_conventions",
            "ion_concentration_rules",
            "ion_placement_rules",
            "genion_protocol",
        }

        # Prefer the explicit parameter_sets block
        if "parameter_sets" in raw:
            param_data_source = raw["parameter_sets"]
            logger.debug(
                "Loading ion parameters from "
                "'parameter_sets' block"
            )
        else:
            # Fallback: treat all non-reserved top-level
            # keys as parameter set entries
            logger.warning(
                "No 'parameter_sets' key found in "
                "ion_parameters.yaml — falling back to "
                "top-level key scan"
            )
            param_data_source = {
                k: v for k, v in raw.items()
                if k not in non_param_keys
            }

        if not isinstance(param_data_source, dict):
            raise KnowledgeBaseError(
                "ion_parameters.yaml: 'parameter_sets' "
                "section is not a mapping"
            )

        for param_name, param_data in (
            param_data_source.items()
        ):
            if not isinstance(param_data, dict):
                continue
            try:
                self._ion_parameters[
                    param_name.lower()
                ] = IonParameterSchema(**param_data)
                logger.debug(
                    f"Loaded ion parameters: {param_name}"
                )
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid ion parameter schema for "
                    f"{param_name}: {e}"
                ) from e

    def _load_compatibility_matrix(self) -> None:
        """
        Load the force field compatibility matrix.

        The file contains the matrix sections directly at
        the top level alongside non-matrix reference
        sections. Non-matrix keys are filtered out before
        passing to CompatibilityMatrixSchema.

        File structure:
            description: ...              ← filtered out
            protein_simulations:          ← matrix data
              standard_folded_protein:
                description: ...
                top_choices:
                  - force_field: ...
            forbidden_combinations:       ← matrix data
              - id: FC001
                combination: ...
            mdp_requirements_by_force_field: ← filtered
              ...
            force_field_selection_guide:  ← filtered
              ...
        """
        raw = self._load_yaml(
            "forcefield_compatibility/"
            "compatibility_matrix.yaml"
        )
        non_matrix_keys = {
            "description",
            "mdp_requirements_by_force_field",
            "force_field_selection_guide",
        }
        matrix_data = {
            k: v for k, v in raw.items()
            if k not in non_matrix_keys
        }
        try:
            self._compatibility_matrix = (
                CompatibilityMatrixSchema(**matrix_data)
            )
            n_forbidden = len(
                self._compatibility_matrix
                .forbidden_combinations
            )
            logger.debug(
                f"Loaded compatibility matrix with "
                f"{n_forbidden} forbidden combinations"
            )
        except ValidationError as e:
            raise KnowledgeBaseError(
                f"Invalid compatibility matrix "
                f"schema: {e}"
            ) from e

    def _load_common_mistakes(self) -> None:
        """
        Load the common mistakes registry.

        Supports two YAML structures for critical_errors
        and warnings sections:

        Pure list (correct structure):
            critical_errors:
              - id: CM001
                severity: ERROR
                message: ...

        Mapping keyed by mistake ID (legacy structure):
            critical_errors:
              description: >
                ...
              CM001:
                id: CM001
                severity: ERROR
                message: ...

        The mapping structure is converted to a list by
        extracting the dict values and skipping any
        non-dict entries (e.g. description strings).

        Additionally flattens cross_phase_consistency_
        mistakes into the appropriate list based on
        each entry's severity field.

        File structure:
            description: ...
            metadata: ...
            critical_errors:          ← list or mapping
              - id: CM001             ← list form
                severity: ERROR
            warnings:                 ← list or mapping
              - id: CM013
                severity: WARNING
            cross_phase_consistency_mistakes:
              CC001:                  ← always a mapping
                id: CC001
                severity: ERROR
        """
        raw = self._load_yaml("common_mistakes.yaml")

        def _extract_mistake_list(
            raw_section: Any,
        ) -> list[dict[str, Any]]:
            """
            Extract a list of mistake dicts from either
            a pure list or a mapping-keyed structure.

            Pure list:
                [{"id": "CM001", ...}, {"id": "CM002"}]
                → returned as-is (non-dict items dropped)

            Mapping keyed by ID:
                {"description": "...", "CM001": {...}}
                → values extracted, non-dict items
                  (e.g. the description string) dropped

            Args:
                raw_section: The raw value of
                    critical_errors or warnings from
                    the YAML file.

            Returns:
                List of mistake dicts ready for Pydantic
                validation.
            """
            if isinstance(raw_section, list):
                # Correct structure — pure list
                # Drop any non-dict items defensively
                return [
                    item for item in raw_section
                    if isinstance(item, dict)
                ]
            if isinstance(raw_section, dict):
                # Legacy structure — mapping keyed by ID
                # Extract values, skip non-dict entries
                # such as description strings
                return [
                    v for v in raw_section.values()
                    if isinstance(v, dict)
                ]
            return []

        critical_errors = _extract_mistake_list(
            raw.get("critical_errors", [])
        )
        warnings = _extract_mistake_list(
            raw.get("warnings", [])
        )

        # Flatten cross-phase consistency mistakes into
        # the appropriate list based on their severity.
        # This section always uses a mapping structure
        # keyed by mistake ID (CC001, CC002, etc.).
        cross_phase = raw.get(
            "cross_phase_consistency_mistakes", {}
        )
        if isinstance(cross_phase, dict):
            for mistake_id, mistake_data in (
                cross_phase.items()
            ):
                if not isinstance(mistake_data, dict):
                    continue
                if (
                    mistake_data.get("severity")
                    == "ERROR"
                ):
                    critical_errors.append(mistake_data)
                else:
                    warnings.append(mistake_data)

        if not critical_errors and not warnings:
            raise KnowledgeBaseError(
                "common_mistakes.yaml: no mistakes "
                "found in critical_errors or warnings "
                "sections. Verify the file structure."
            )

        mistakes_data = {
            "critical_errors": critical_errors,
            "warnings": warnings,
        }
        try:
            self._common_mistakes = (
                CommonMistakesRegistry(**mistakes_data)
            )
            n_total = len(
                self._common_mistakes.get_all()
            )
            n_errors = len(
                self._common_mistakes.critical_errors
            )
            n_warnings = len(
                self._common_mistakes.warnings
            )
            logger.debug(
                f"Loaded {n_total} common mistake rules "
                f"({n_errors} errors, "
                f"{n_warnings} warnings)"
            )
        except ValidationError as e:
            raise KnowledgeBaseError(
                f"Invalid common mistakes schema: {e}"
            ) from e


# ─────────────────────────────────────────────
# Module-level singleton factory
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_knowledge_base(
    knowledge_base_dir: str = "knowledge_base",
) -> KnowledgeBase:
    """
    Return the singleton KnowledgeBase instance.

    Uses lru_cache to ensure the knowledge base is loaded
    only once per process. The directory path is passed as
    a string (not Path) to satisfy lru_cache's requirement
    for hashable arguments.

    Args:
        knowledge_base_dir: Path to the knowledge base
            directory as a string. Defaults to
            'knowledge_base' relative to the working dir.

    Returns:
        The loaded and validated KnowledgeBase instance.

    Example:
        kb = get_knowledge_base()
        kb = get_knowledge_base("/abs/path/to/kb")
        kb = get_knowledge_base("custom_kb_dir")
    """
    return KnowledgeBase(Path(knowledge_base_dir))