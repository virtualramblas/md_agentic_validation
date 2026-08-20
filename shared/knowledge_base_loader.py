# shared/knowledge_base_loader.py

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    """Raised when the knowledge base cannot be loaded or validated."""
    pass


class KnowledgeBase:
    """
    Central access point for all knowledge base data.

    Loads all YAML files at instantiation, validates them against
    Pydantic schemas, and exposes typed accessors for each section.

    All data is immutable after loading — the knowledge base is
    read-only at runtime.
    """

    def __init__(self, knowledge_base_dir: Path) -> None:
        self._kb_dir = Path(knowledge_base_dir)
        self._validate_directory_structure()

        logger.info(
            f"Loading knowledge base from {self._kb_dir}"
        )

        self._phases: dict[SimulationPhase, PhaseSchema] = {}
        self._force_fields: dict[str, ForceFieldSchema] = {}
        self._water_models: dict[str, WaterModelSchema] = {}
        self._ion_parameters: dict[str, IonParameterSchema] = {}
        self._compatibility_matrix: CompatibilityMatrixSchema
        self._common_mistakes: CommonMistakesRegistry

        self._load_all()
        logger.info("Knowledge base loaded and validated successfully")

    # ─────────────────────────────────────────
    # Public Accessors
    # ─────────────────────────────────────────

    def get_phase_schema(
        self, phase: SimulationPhase
    ) -> PhaseSchema:
        """Return the MDP parameter schema for a simulation phase."""
        if phase not in self._phases:
            raise KnowledgeBaseError(
                f"No schema found for phase: {phase.value}"
            )
        return self._phases[phase]

    def get_force_field(self, ff_name: str) -> ForceFieldSchema:
        """Return the force field definition by name."""
        ff_name_lower = ff_name.lower()
        if ff_name_lower not in self._force_fields:
            raise KnowledgeBaseError(
                f"Unknown force field: {ff_name}. "
                f"Available: {list(self._force_fields.keys())}"
            )
        return self._force_fields[ff_name_lower]

    def get_water_model(
        self, water_model_name: str
    ) -> WaterModelSchema:
        """Return the water model definition by name."""
        name_lower = water_model_name.lower()
        if name_lower not in self._water_models:
            raise KnowledgeBaseError(
                f"Unknown water model: {water_model_name}. "
                f"Available: {list(self._water_models.keys())}"
            )
        return self._water_models[name_lower]

    def get_ion_parameters(
        self, ion_param_name: str
    ) -> IonParameterSchema:
        """Return ion parameter set definition by name."""
        name_lower = ion_param_name.lower()
        if name_lower not in self._ion_parameters:
            raise KnowledgeBaseError(
                f"Unknown ion parameter set: {ion_param_name}. "
                f"Available: {list(self._ion_parameters.keys())}"
            )
        return self._ion_parameters[name_lower]

    def get_compatibility_matrix(
        self,
    ) -> CompatibilityMatrixSchema:
        """Return the full compatibility matrix."""
        return self._compatibility_matrix

    def get_common_mistakes(self) -> CommonMistakesRegistry:
        """Return the common mistakes registry."""
        return self._common_mistakes

    def list_force_fields(self) -> list[str]:
        """Return names of all known force fields."""
        return list(self._force_fields.keys())

    def list_water_models(self) -> list[str]:
        """Return names of all known water models."""
        return list(self._water_models.keys())

    def is_combination_forbidden(
        self,
        force_field: str,
        water_model: Optional[str] = None,
        disp_corr: Optional[str] = None,
        vdw_modifier: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a parameter combination is explicitly forbidden.

        Returns:
            (is_forbidden, reason_message)
        """
        for forbidden in (
            self._compatibility_matrix.forbidden_combinations
        ):
            combo = forbidden.combination
            match = True

            if "force_field" in combo:
                if combo["force_field"].lower() != force_field.lower():
                    match = False

            if water_model and "water" in combo:
                if combo["water"].lower() != water_model.lower():
                    match = False

            if disp_corr and "DispCorr" in combo:
                if combo["DispCorr"].lower() != disp_corr.lower():
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
    # Private Loading Methods
    # ─────────────────────────────────────────

    def _validate_directory_structure(self) -> None:
        """Verify all required knowledge base files exist."""
        required_files = [
            "phases/energy_minimization.yaml",
            "phases/nvt_equilibration.yaml",
            "phases/npt_equilibration.yaml",
            "phases/production_md.yaml",
            "forcefield_compatibility/force_fields.yaml",
            "forcefield_compatibility/water_models.yaml",
            "forcefield_compatibility/ion_parameters.yaml",
            "forcefield_compatibility/compatibility_matrix.yaml",
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

    def _load_yaml(self, relative_path: str) -> dict[str, Any]:
        """Load and parse a single YAML file."""
        full_path = self._kb_dir / relative_path
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                raise KnowledgeBaseError(
                    f"Empty YAML file: {full_path}"
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

    def _load_all(self) -> None:
        """Load and validate all knowledge base sections."""
        self._load_phases()
        self._load_force_fields()
        self._load_water_models()
        self._load_ion_parameters()
        self._load_compatibility_matrix()
        self._load_common_mistakes()

    def _load_phases(self) -> None:
        """Load all four simulation phase schemas."""
        phase_files = {
            SimulationPhase.ENERGY_MINIMIZATION:
                "phases/energy_minimization.yaml",
            SimulationPhase.NVT_EQUILIBRATION:
                "phases/nvt_equilibration.yaml",
            SimulationPhase.NPT_EQUILIBRATION:
                "phases/npt_equilibration.yaml",
            SimulationPhase.PRODUCTION_MD:
                "phases/production_md.yaml",
        }
        for phase, filepath in phase_files.items():
            data = self._load_yaml(filepath)
            try:
                self._phases[phase] = PhaseSchema(**data)
                logger.debug(
                    f"Loaded phase schema: {phase.value}"
                )
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid schema in {filepath}: {e}"
                ) from e

    def _load_force_fields(self) -> None:
        """Load all force field definitions."""
        data = self._load_yaml(
            "forcefield_compatibility/force_fields.yaml"
        )
        for ff_name, ff_data in data.items():
            try:
                self._force_fields[ff_name.lower()] = (
                    ForceFieldSchema(**ff_data)
                )
                logger.debug(f"Loaded force field: {ff_name}")
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid force field schema for "
                    f"{ff_name}: {e}"
                ) from e

    def _load_water_models(self) -> None:
        """Load all water model definitions."""
        data = self._load_yaml(
            "forcefield_compatibility/water_models.yaml"
        )
        for model_name, model_data in data.items():
            try:
                self._water_models[model_name.lower()] = (
                    WaterModelSchema(**model_data)
                )
                logger.debug(f"Loaded water model: {model_name}")
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid water model schema for "
                    f"{model_name}: {e}"
                ) from e

    def _load_ion_parameters(self) -> None:
        """Load all ion parameter set definitions."""
        data = self._load_yaml(
            "forcefield_compatibility/ion_parameters.yaml"
        )
        for param_name, param_data in data.items():
            try:
                self._ion_parameters[param_name.lower()] = (
                    IonParameterSchema(**param_data)
                )
                logger.debug(
                    f"Loaded ion parameters: {param_name}"
                )
            except ValidationError as e:
                raise KnowledgeBaseError(
                    f"Invalid ion parameter schema for "
                    f"{param_name}: {e}"
                ) from e

    def _load_compatibility_matrix(self) -> None:
        """Load the force field compatibility matrix."""
        data = self._load_yaml(
            "forcefield_compatibility/compatibility_matrix.yaml"
        )
        try:
            self._compatibility_matrix = (
                CompatibilityMatrixSchema(**data)
            )
            logger.debug("Loaded compatibility matrix")
        except ValidationError as e:
            raise KnowledgeBaseError(
                f"Invalid compatibility matrix schema: {e}"
            ) from e

    def _load_common_mistakes(self) -> None:
        """Load the common mistakes registry."""
        data = self._load_yaml("common_mistakes.yaml")
        try:
            self._common_mistakes = CommonMistakesRegistry(**data)
            n_mistakes = len(self._common_mistakes.get_all())
            logger.debug(
                f"Loaded {n_mistakes} common mistake rules"
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
    only once per process. Pass knowledge_base_dir as a
    string (not Path) to make it hashable for lru_cache.
    """
    return KnowledgeBase(Path(knowledge_base_dir))