# shared/knowledge_base_models.py

from __future__ import annotations
from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, model_validator


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
    min: Optional[float] = None
    max: Optional[float] = None
    recommended: Optional[float] = None
    unit: Optional[str] = None


class DependencyRule(BaseModel):
    parameter: str
    if_value: Union[str, float, int]
    then_max: Optional[float] = None
    then_min: Optional[float] = None
    then_allowed_values: Optional[list[str]] = None


class MDPParameterSchema(BaseModel):
    type: ParameterType
    required: bool = False
    allowed_values: Optional[list[str]] = None
    forbidden_values: Optional[list[str]] = None
    recommended: Optional[Union[str, float, int]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unit: Optional[str] = None
    must_equal: Optional[str] = None
    depends_on: Optional[dict[str, DependencyRule]] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_enum_has_allowed_values(self) -> MDPParameterSchema:
        if self.type == ParameterType.ENUM:
            if not self.allowed_values:
                raise ValueError(
                    "Enum parameters must define allowed_values"
                )
        return self

    @model_validator(mode="after")
    def validate_numeric_has_bounds(self) -> MDPParameterSchema:
        if self.type in (ParameterType.FLOAT, ParameterType.INTEGER):
            if self.min is None and self.max is None:
                raise ValueError(
                    "Numeric parameters must define at least "
                    "one of min or max"
                )
        return self


class MDPSectionSchema(BaseModel):
    parameters: dict[str, MDPParameterSchema]


class PhasePrerequisites(BaseModel):
    must_follow: Optional[SimulationPhase] = None
    required_input_files: list[str] = Field(default_factory=list)
    continuation: bool = False
    posre_itp_required: bool = False
    notes: Optional[str] = None


class PhaseSchema(BaseModel):
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
    rating: str
    notes: Optional[str] = None


class MDPRequirements(BaseModel):
    rcoulomb: Optional[float] = None
    rvdw: Optional[float] = None
    vdw_modifier: Optional[str] = None
    rvdw_switch: Optional[float] = None
    DispCorr: Optional[str] = None
    critical_note: Optional[str] = None


class ForceFieldSchema(BaseModel):
    full_name: str
    version_introduced: Optional[int] = None
    reference: Optional[str] = None
    type: str
    gromacs_directory: Optional[str] = None
    suitable_for: dict[str, SystemSuitability]
    recommended_water_models: dict[str, Union[str, list[str]]]
    recommended_ions: dict[str, str]
    known_limitations: list[str] = Field(default_factory=list)
    mdp_specific_requirements: Optional[MDPRequirements] = None
    notes: Optional[str] = None


class WaterModelProperties(BaseModel):
    density_gcm3: Optional[float] = None
    diffusion_coefficient_m2s: Optional[float] = None
    dielectric_constant: Optional[float] = None
    melting_point_K: Optional[float] = None


class WaterModelSchema(BaseModel):
    sites: int
    description: Optional[str] = None
    properties: Optional[WaterModelProperties] = None
    compatible_force_fields: dict[str, Union[str, list[str]]]
    gromacs_topology_file: Optional[str] = None
    gromacs_water_model_flag: Optional[str] = None
    notes: Optional[str] = None


class IonParameterSchema(BaseModel):
    reference: str
    ions_covered: list[str]
    parameterized_with: Union[str, list[str]]
    compatible_water_models: Optional[list[str]] = None
    compatible_force_fields: Optional[list[str]] = None
    not_recommended_with: Optional[list[str]] = None
    notes: Optional[str] = None


class CompatibilityEntry(BaseModel):
    force_field: str
    water: str
    ions: Optional[str] = None
    lipids: Optional[str] = None
    small_molecule_ff: Optional[str] = None
    rating: CompatibilityRating
    notes: Optional[str] = None


class ForbiddenCombination(BaseModel):
    combination: dict[str, str]
    reason: str
    severity: Severity


class CompatibilityMatrixSchema(BaseModel):
    protein_simulations: dict[str, list[CompatibilityEntry]]
    forbidden_combinations: list[ForbiddenCombination]


# ─────────────────────────────────────────────
# Box and Solvation Models
# ─────────────────────────────────────────────

class BoxTypeDefinition(BaseModel):
    gromacs_flag: str
    shape: str
    efficiency_vs_sphere: Optional[float] = None
    recommended_for: list[str] = Field(default_factory=list)
    not_recommended_for: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class BoxDimensionRules(BaseModel):
    minimum_solute_to_box_edge_distance: dict[str, float]
    minimum_absolute_box_dimensions_nm: float
    notes: Optional[str] = None


class SolvationRules(BaseModel):
    minimum_water_molecules: int
    recommended_minimum_water_molecules: int
    minimum_water_to_protein_ratio: float
    recommended_water_to_protein_ratio: float
    notes: Optional[str] = None


class IonizationRules(BaseModel):
    physiological_NaCl_mM: float
    minimum_distance_from_solute_nm: float
    neutralization_required: bool
    notes: Optional[str] = None


# ─────────────────────────────────────────────
# Common Mistakes Models
# ─────────────────────────────────────────────

class CommonMistake(BaseModel):
    id: str
    name: str
    check_description: str
    message: str
    severity: Severity
    applicable_phases: Optional[list[SimulationPhase]] = None


class CommonMistakesRegistry(BaseModel):
    critical_errors: list[CommonMistake]
    warnings: list[CommonMistake]

    def get_all(self) -> list[CommonMistake]:
        return self.critical_errors + self.warnings

    def get_by_id(self, mistake_id: str) -> Optional[CommonMistake]:
        return next(
            (m for m in self.get_all() if m.id == mistake_id),
            None
        )