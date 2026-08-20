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

    @model_validator(mode="before")
    @classmethod
    def coerce_bool_to_str_in_value_lists(
        cls, values: Any
    ) -> Any:
        """
        YAML automatically parses bare 'no' and 'yes' tokens
        as Python booleans (False and True respectively).
        This validator coerces them back to their string
        equivalents before Pydantic validates the field types.

        Affected fields:
          - allowed_values  (list of strings)
          - forbidden_values (list of strings)
          - recommended     (scalar string/float/int)

        Example YAML that triggers this:
          allowed_values: [no]       # parsed as [False]
          forbidden_values: [yes]    # parsed as [True]
          recommended: no            # parsed as False

        After coercion:
          allowed_values: ["no"]
          forbidden_values: ["yes"]
          recommended: "no"
        """
        if not isinstance(values, dict):
            return values

        bool_to_str: dict[bool, str] = {
            True: "yes",
            False: "no",
        }

        # Coerce list fields
        for field_name in (
            "allowed_values",
            "forbidden_values",
            "recommended_values",
        ):
            raw = values.get(field_name)
            if isinstance(raw, list):
                values[field_name] = [
                    bool_to_str[item]
                    if isinstance(item, bool)
                    else item
                    for item in raw
                ]

        # Coerce scalar recommended value
        rec = values.get("recommended")
        if isinstance(rec, bool):
            values["recommended"] = bool_to_str[rec]

        return values

    @model_validator(mode="after")
    def validate_enum_has_allowed_values(
        self,
    ) -> "MDPParameterSchema":
        """
        Enum parameters must declare their allowed values.
        Without allowed_values, the schema checker cannot
        validate whether a proposed MDP value is permitted.
        """
        if self.type == ParameterType.ENUM:
            if not self.allowed_values:
                raise ValueError(
                    f"Enum parameter must define "
                    f"allowed_values"
                )
        return self

    @model_validator(mode="after")
    def validate_numeric_has_bounds(
        self,
    ) -> "MDPParameterSchema":
        """
        Numeric parameters must declare at least one bound
        (min or max) so the range validator can check values.
        """
        if self.type in (
            ParameterType.FLOAT,
            ParameterType.INTEGER,
        ):
            if self.min is None and self.max is None:
                raise ValueError(
                    f"Numeric parameter must define "
                    f"at least one of min or max"
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
    def coerce_bool_to_str_in_mdp_requirements(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean values in MDP requirements fields.
        Specifically handles DispCorr: no which YAML parses
        as DispCorr: False.
        """
        if not isinstance(values, dict):
            return values

        bool_to_str: dict[bool, str] = {
            True: "yes",
            False: "no",
        }

        for field_name in (
            "DispCorr",
            "vdw_modifier",
            "coulombtype",
            "vdwtype",
            "critical_note",
            "notes",
        ):
            val = values.get(field_name)
            if isinstance(val, bool):
                values[field_name] = bool_to_str[val]

        return values


class ForceFieldSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    full_name: str
    version_introduced: Optional[int] = None
    reference: Optional[str] = None
    type: str
    gromacs_directory: Optional[str] = None
    suitable_for: dict[str, SystemSuitability]
    recommended_water_models: dict[
        str, Union[str, list[str]]
    ]
    recommended_ions: dict[str, str]
    known_limitations: list[str] = Field(
        default_factory=list
    )
    mdp_specific_requirements: Optional[
        MDPRequirements
    ] = None
    notes: Optional[str] = None


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


class IonParameterSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference: str
    ions_covered: list[str]
    parameterized_with: Union[str, list[str]]
    compatible_water_models: Optional[list[str]] = None
    compatible_force_fields: Optional[list[str]] = None
    not_recommended_with: Optional[list[str]] = None
    notes: Optional[str] = None


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


class ForbiddenCombination(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    combination: dict[str, str]
    reason: str
    severity: Severity

    @model_validator(mode="before")
    @classmethod
    def coerce_bool_in_combination(
        cls, values: Any
    ) -> Any:
        """
        Coerce boolean values inside the combination dict.
        YAML may parse values like DispCorr: no as False.
        """
        if not isinstance(values, dict):
            return values

        bool_to_str: dict[bool, str] = {
            True: "yes",
            False: "no",
        }

        combo = values.get("combination")
        if isinstance(combo, dict):
            values["combination"] = {
                k: bool_to_str[v]
                if isinstance(v, bool)
                else v
                for k, v in combo.items()
            }

        return values


class CompatibilityMatrixSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    protein_simulations: dict[
        str, list[CompatibilityEntry]
    ]
    forbidden_combinations: list[ForbiddenCombination]


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
        Includes mistakes with no phase restriction.
        """
        return [
            m for m in self.get_all()
            if (
                m.applicable_phases is None
                or phase in m.applicable_phases
            )
        ]