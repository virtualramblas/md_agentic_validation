# shared/knowledge_base_models.py

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ParameterType(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    ENUM = "enum"
    STRING = "string"
    BOOLEAN = "boolean"


class SimulationPhase(str, Enum):
    ENERGY_MINIMIZATION = "energy_minimization"
    NVT_EQUILIBRATION = "nvt_equilibration"
    NPT_EQUILIBRATION = "npt_equilibration"
    PRODUCTION_MD = "production_md"


class SystemType(str, Enum):
    PROTEIN_IN_WATER = "protein_in_water"
    PROTEIN_LIGAND = "protein_ligand"
    MEMBRANE_PROTEIN = "membrane_protein"
    NUCLEIC_ACID = "nucleic_acid"
    IDP = "idp"


class CompatibilityRating(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    ACCEPTABLE = "ACCEPTABLE"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"
    FORBIDDEN = "FORBIDDEN"


class BoxType(str, Enum):
    CUBIC = "cubic"
    DODECAHEDRON = "dodecahedron"
    TRICLINIC = "triclinic"
    OCTAHEDRON = "octahedron"


# ─────────────────────────────────────────────
# Boolean coercion utilities
# ─────────────────────────────────────────────

_BOOL_TO_STR: dict[bool, str] = {
    True: "yes",
    False: "no",
}


def _coerce_bools_in_dict(data: Any) -> Any:
    """
    Recursively walk a nested dict/list structure and
    coerce all boolean values to their string equivalents
    ('yes' for True, 'no' for False).

    This is necessary because PyYAML's safe_load parses
    bare YAML boolean tokens as Python booleans:

      YAML token   Python value   Intended string
      ----------   ------------   ---------------
      no           False          "no"
      yes          True           "yes"
      No           False          "no"
      Yes          True           "yes"

    The coercion is applied recursively to ALL values
    throughout the entire nested structure so that no
    boolean survives to reach Pydantic field validation.

    This function is applied as a mode="before" validator
    on every top-level model so that coercion happens
    before Pydantic attempts to validate any field type,
    including list[str] fields that would reject a bool.

    Args:
        data: Any Python object (dict, list, scalar).

    Returns:
        The same structure with all booleans replaced
        by their string equivalents.
    """
    if isinstance(data, dict):
        return {
            key: _coerce_bools_in_dict(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            _coerce_bools_in_dict(item)
            for item in data
        ]
    if isinstance(data, bool):
        return _BOOL_TO_STR[data]
    return data


# ─────────────────────────────────────────────
# MDP Parameter Schema Models
# ─────────────────────────────────────────────

class NumericConstraint(BaseModel):
    model_config = ConfigDict(extra="allow")

    min: Optional[float] = None
    max: Optional[float] = None
    recommended: Optional[float] = None
    unit: Optional[str] = None


class DependencyRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    parameter: str
    if_value: Union[str, float, int]
    then_max: Optional[float] = None
    then_min: Optional[float] = None
    then_allowed_values: Optional[list[str]] = None


class MDPParameterSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: ParameterType
    required: bool = False
    allowed_values: Optional[list[str]] = None
    forbidden_values: Optional[list[str]] = None
    recommended: Optional[Union[str, float, int]] = None
    recommended_values: Optional[list[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unit: Optional[str] = None
    must_equal: Optional[str] = None
    depends_on: Optional[dict[str, DependencyRule]] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_enum_has_allowed_values(
        self,
    ) -> "MDPParameterSchema":
        """
        Enum parameters must declare their allowed values.
        Without allowed_values the schema checker cannot
        validate whether a proposed MDP value is permitted.
        """
        if self.type == ParameterType.ENUM:
            if not self.allowed_values:
                raise ValueError(
                    "Enum parameter must define "
                    "allowed_values"
                )
        return self

    @model_validator(mode="after")
    def validate_numeric_has_bounds(
        self,
    ) -> "MDPParameterSchema":
        """
        Numeric parameters must declare at least one
        bound (min or max) so the range validator can
        check proposed values.
        """
        if self.type in (
            ParameterType.FLOAT,
            ParameterType.INTEGER,
        ):
            if self.min is None and self.max is None:
                raise ValueError(
                    "Numeric parameter must define "
                    "at least one of min or max"
                )
        return self


class MDPSectionSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    parameters: dict[str, MDPParameterSchema]


class PhasePrerequisites(BaseModel):
    model_config = ConfigDict(extra="allow")

    must_follow: Optional[str] = None
    required_input_files: list[str] = Field(
        default_factory=list
    )
    continuation: bool = False
    posre_itp_required: bool = False
    notes: Optional[str] = None


class PhaseSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str
    prerequisites: Optional[PhasePrerequisites] = None
    run_control: MDPSectionSchema
    output_control: MDPSectionSchema
    neighbor_searching: MDPSectionSchema
    electrostatics: MDPSectionSchema
    vdw: MDPSectionSchema
    temperature_coupling: MDPSectionSchema
    pressure_coupling: MDPSectionSchema
    velocity_generation: MDPSectionSchema
    constraints: MDPSectionSchema
    position_restraints: Optional[MDPSectionSchema] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_all_bools_to_strings(
        cls, values: Any
    ) -> Any:
        """
        Recursively coerce all boolean values in the
        entire PhaseSchema input dict to their string
        equivalents before Pydantic validates any field.

        Must be applied at the PhaseSchema level —
        the outermost model — because Pydantic propagates
        validation top-down through nested models. By the
        time a mode="before" validator on a nested model
        (MDPSectionSchema or MDPParameterSchema) would
        fire, Pydantic has already attempted to coerce
        list[str] fields and rejected boolean values.

        Applying the coercion here, before any nested
        model is instantiated, guarantees that no boolean
        value survives to reach field validation anywhere
        in the PhaseSchema tree.
        """
        return _coerce_bools_in_dict(values)


# ─────────────────────────────────────────────
# Force Field Compatibility Models
# ─────────────────────────────────────────────

class SystemSuitability(BaseModel):
    model_config = ConfigDict(extra="allow")

    rating: str
    notes: Optional[str] = None


class MDPRequirements(BaseModel):
    model_config = ConfigDict(extra="allow")

    rcoulomb: Optional[float] = None
    rvdw: Optional[float] = None
    vdw_modifier: Optional[str] = None
    rvdw_switch: Optional[float] = None
    DispCorr: Optional[str] = None
    coulombtype: Optional[str] = None
    vdwtype: Optional[str] = None
    critical_note: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_mdp_requirements(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean values in MDP requirements.
        Applied here because MDPRequirements is
        instantiated from ForceFieldSchema which has
        its own top-level coercion, but also may be
        instantiated directly in tests or other contexts.
        """
        return _coerce_bools_in_dict(values)


class ForceFieldSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    full_name: str
    version_introduced: Optional[int] = None
    reference: Optional[str] = None
    type: str
    gromacs_directory: Optional[str] = None
    purpose: Optional[str] = None

    # Required for standalone biomolecular force fields
    # (AMBER, CHARMM, GROMOS, OPLS).
    # Optional for small molecule parameterization tools
    # (GAFF, GAFF2, CGenFF) which do not define water
    # models or ion parameters independently.
    suitable_for: dict[str, SystemSuitability] = Field(
        default_factory=dict
    )
    recommended_water_models: Optional[
        dict[str, Union[str, list[str]]]
    ] = None
    recommended_ions: Optional[
        dict[str, str]
    ] = None
    known_limitations: list[str] = Field(
        default_factory=list
    )
    mdp_specific_requirements: Optional[
        MDPRequirements
    ] = None
    notes: Optional[str] = None

    # Small molecule FF specific fields.
    # Present in GAFF, GAFF2, CGenFF but not in
    # standalone biomolecular force fields.
    compatible_protein_force_fields: Optional[
        list[str]
    ] = None
    preferred_with: Optional[str] = None
    charge_method_required: Optional[str] = None
    tools_for_parameterization: Optional[
        list[str]
    ] = None
    penalty_score_guidance: Optional[
        dict[str, str]
    ] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_force_field(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean values throughout the force field
        dict before Pydantic validates any nested model.
        Handles DispCorr: no in mdp_specific_requirements
        and any other boolean YAML tokens.
        """
        return _coerce_bools_in_dict(values)

    @model_validator(mode="after")
    def validate_standalone_ff_has_water_and_ions(
        self,
    ) -> "ForceFieldSchema":
        """
        Standalone biomolecular force fields (all-atom,
        united-atom) must define recommended_water_models
        and recommended_ions.

        Small molecule force fields (type contains the
        substring 'small molecule') are parameterization
        tools that supplement a primary force field and
        are exempt from this requirement.

        Examples:
          'all-atom'                → standalone → required
          'united-atom'             → standalone → required
          'all-atom small molecule' → tool       → optional
        """
        is_small_molecule_ff = (
            "small molecule" in self.type.lower()
        )
        if not is_small_molecule_ff:
            if self.recommended_water_models is None:
                raise ValueError(
                    f"Standalone force field "
                    f"'{self.full_name}' must define "
                    f"recommended_water_models. "
                    f"If this is a small molecule "
                    f"parameterization tool, set "
                    f"type to include 'small molecule'."
                )
            if self.recommended_ions is None:
                raise ValueError(
                    f"Standalone force field "
                    f"'{self.full_name}' must define "
                    f"recommended_ions. "
                    f"If this is a small molecule "
                    f"parameterization tool, set "
                    f"type to include 'small molecule'."
                )
        return self


class WaterModelProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    density_gcm3: Optional[float] = None
    diffusion_coefficient_m2s: Optional[float] = None
    dielectric_constant: Optional[float] = None
    melting_point_K: Optional[float] = None


class WaterModelSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    sites: int
    description: Optional[str] = None
    properties: Optional[WaterModelProperties] = None
    compatible_force_fields: dict[
        str, Union[str, list[str]]
    ]
    gromacs_topology_file: Optional[str] = None
    gromacs_water_model_flag: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_water_model(
        cls, values: Any
    ) -> Any:
        """Coerce boolean YAML tokens in water model
        data before field validation."""
        return _coerce_bools_in_dict(values)


class IonParameterSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference: str
    ions_covered: list[str]

    # Optional because some advanced ion models
    # (e.g. Li-Merz 12-6-4) do not specify a single
    # water model they were parameterized with, or
    # list compatibility separately via
    # compatible_water_models instead.
    parameterized_with: Optional[
        Union[str, list[str]]
    ] = None

    compatible_water_models: Optional[list[str]] = None
    compatible_force_fields: Optional[list[str]] = None
    not_recommended_with: Optional[list[str]] = None
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_ion_parameters(
        cls, values: Any
    ) -> Any:
        """Coerce boolean YAML tokens in ion parameter
        data before field validation."""
        return _coerce_bools_in_dict(values)


class CompatibilityEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    force_field: str
    water: Optional[str] = None
    ions: Optional[str] = None
    lipids: Optional[str] = None
    small_molecule_ff: Optional[str] = None
    charge_method: Optional[str] = None
    nucleic_acid_params: Optional[str] = None
    applicable_to: Optional[str] = None
    rating: CompatibilityRating
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_compatibility_entry(
        cls, values: Any
    ) -> Any:
        """Coerce boolean YAML tokens in compatibility
        entry data before field validation."""
        return _coerce_bools_in_dict(values)


class ForbiddenCombination(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    combination: dict[str, str]
    reason: str
    severity: Severity

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_forbidden_combination(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean values inside the combination dict
        and all other fields. YAML parses DispCorr: no
        as False — this must be "no" for string matching.
        """
        return _coerce_bools_in_dict(values)


class SystemTypeCompatibility(BaseModel):
    """
    Represents the compatibility entries for a single
    system type (e.g. standard_folded_protein) in the
    compatibility matrix.

    The YAML structure for each system type is:

        standard_folded_protein:
          description: >
            Globular, well-folded protein...
          top_choices:
            - force_field: amber99sb-ildn
              water: TIP3P
              rating: RECOMMENDED
              ...

    The description and top_choices keys are siblings
    under the system type key. This model captures both.
    """
    model_config = ConfigDict(extra="allow")

    description: Optional[str] = None
    top_choices: list[CompatibilityEntry] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_system_type(
        cls, values: Any
    ) -> Any:
        """Coerce boolean YAML tokens before field
        validation."""
        return _coerce_bools_in_dict(values)


class CompatibilityMatrixSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    protein_simulations: dict[
        str, SystemTypeCompatibility
    ]
    forbidden_combinations: list[ForbiddenCombination]

    @model_validator(mode="before")
    @classmethod
    def coerce_bools_in_matrix(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean YAML tokens throughout the entire
        compatibility matrix before field validation.
        Applied at this level to catch any booleans in
        the protein_simulations or forbidden_combinations
        sections before their nested models are
        instantiated.
        """
        return _coerce_bools_in_dict(values)


# ─────────────────────────────────────────────
# Box and Solvation Models
# ─────────────────────────────────────────────

class BoxTypeDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    gromacs_flag: str
    shape: str
    efficiency_vs_sphere: Optional[float] = None
    recommended_for: list[str] = Field(
        default_factory=list
    )
    not_recommended_for: list[str] = Field(
        default_factory=list
    )
    notes: Optional[str] = None


class BoxDimensionRules(BaseModel):
    model_config = ConfigDict(extra="allow")

    minimum_solute_to_box_edge_distance: dict[str, float]
    minimum_absolute_box_dimensions_nm: float
    notes: Optional[str] = None


class SolvationRules(BaseModel):
    model_config = ConfigDict(extra="allow")

    minimum_water_molecules: int
    recommended_minimum_water_molecules: int
    minimum_water_to_protein_ratio: float
    recommended_water_to_protein_ratio: float
    notes: Optional[str] = None


class IonizationRules(BaseModel):
    model_config = ConfigDict(extra="allow")

    physiological_NaCl_mM: float
    minimum_distance_from_solute_nm: float
    neutralization_required: bool
    notes: Optional[str] = None


# ─────────────────────────────────────────────
# Common Mistakes Models
# ─────────────────────────────────────────────

class CommonMistake(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    category: Optional[str] = None
    severity: Severity
    applicable_phases: Optional[list[str]] = None
    check_condition: Optional[str] = None
    check_description: Optional[str] = None
    message: str
    correction_suggestion: Optional[str] = None
    reference: Optional[str] = None


class CommonMistakesRegistry(BaseModel):
    model_config = ConfigDict(extra="allow")

    critical_errors: list[CommonMistake]
    warnings: list[CommonMistake]

    def get_all(self) -> list[CommonMistake]:
        """Return all mistakes regardless of severity."""
        return self.critical_errors + self.warnings

    def get_by_id(
        self, mistake_id: str
    ) -> Optional[CommonMistake]:
        """
        Return a mistake by its ID, or None if not found.
        """
        return next(
            (
                m for m in self.get_all()
                if m.id == mistake_id
            ),
            None,
        )

    def get_by_severity(
        self, severity: Severity
    ) -> list[CommonMistake]:
        """Return all mistakes of a given severity."""
        return [
            m for m in self.get_all()
            if m.severity == severity
        ]

    def get_by_phase(
        self, phase: str
    ) -> list[CommonMistake]:
        """
        Return all mistakes applicable to a given phase.
        Includes mistakes with no phase restriction
        (applicable_phases is None).
        """
        return [
            m for m in self.get_all()
            if (
                m.applicable_phases is None
                or phase in m.applicable_phases
            )
        ]