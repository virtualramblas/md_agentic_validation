# layer1/tests/test_knowledge_base_loader.py

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from shared.knowledge_base_loader import (
    KnowledgeBase,
    KnowledgeBaseError,
)
from shared.knowledge_base_models import (
    CompatibilityRating,
    Severity,
    SimulationPhase,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _minimal_mdp_section(
    param_name: str = "integrator",
    param_type: str = "enum",
    allowed_values: list[str] | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> dict[str, Any]:
    """
    Build a minimal MDP section dict with one parameter.
    Used to construct minimal phase schemas for testing.
    """
    if param_type == "enum":
        param = {
            "type": "enum",
            "required": True,
            "allowed_values": allowed_values or ["steep"],
            "notes": "Test parameter",
        }
    else:
        param = {
            "type": param_type,
            "required": True,
            "min": min_val if min_val is not None else 1,
            "max": max_val if max_val is not None else 100,
            "notes": "Test parameter",
        }
    return {"parameters": {param_name: param}}


def _build_minimal_phase_schema() -> dict[str, Any]:
    """
    Build a minimal but valid phase schema dict that
    satisfies all required PhaseSchema fields.
    """
    return {
        "description": "Minimal test phase schema",
        "run_control": _minimal_mdp_section(
            "integrator", "enum", ["steep", "md"]
        ),
        "output_control": _minimal_mdp_section(
            "nstlog", "integer", min_val=1, max_val=1000
        ),
        "neighbor_searching": _minimal_mdp_section(
            "cutoff-scheme", "enum", ["Verlet"]
        ),
        "electrostatics": _minimal_mdp_section(
            "coulombtype", "enum", ["PME"]
        ),
        "vdw": _minimal_mdp_section(
            "rvdw", "float", min_val=0.8, max_val=1.4
        ),
        "temperature_coupling": _minimal_mdp_section(
            "tcoupl", "enum", ["no", "v-rescale"]
        ),
        "pressure_coupling": _minimal_mdp_section(
            "pcoupl", "enum", ["no", "berendsen"]
        ),
        "velocity_generation": _minimal_mdp_section(
            "gen-vel", "enum", ["no", "yes"]
        ),
        "constraints": _minimal_mdp_section(
            "constraints", "enum", ["none", "h-bonds"]
        ),
    }


def _build_minimal_kb(base: Path) -> Path:
    """
    Build a minimal valid knowledge base directory for
    unit testing. Contains only the fields required by
    Pydantic models.
    """
    kb_dir = base / "knowledge_base"

    # Create directory structure
    (kb_dir / "phases").mkdir(parents=True)
    (kb_dir / "forcefield_compatibility").mkdir(
        parents=True
    )
    (kb_dir / "box_solvation_rules").mkdir(parents=True)

    # ── Phase files ──────────────────────────────────────
    phase_schema = _build_minimal_phase_schema()
    phase_file_map = {
        "energy_minimization.yaml": "energy_minimization",
        "nvt_equilibration.yaml": "nvt_equilibration",
        "npt_equilibration.yaml": "npt_equilibration",
        "production_md.yaml": "production_md",
    }
    for filename, root_key in phase_file_map.items():
        with open(
            kb_dir / "phases" / filename, "w"
        ) as f:
            yaml.dump({root_key: phase_schema}, f)

    # ── Force fields ─────────────────────────────────────
    force_fields: dict[str, Any] = {
        "amber_family": {
            "description": "AMBER force fields",
            "amber99sb-ildn": {
                "full_name": "AMBER ff99SB-ILDN",
                "type": "all-atom",
                "gromacs_directory": "amber99sb-ildn.ff",
                "suitable_for": {
                    "proteins": {
                        "rating": "very_good",
                        "notes": "Good for folded proteins",
                    }
                },
                "recommended_water_models": {
                    "primary": "TIP3P",
                    "acceptable": ["TIP4P-EW"],
                },
                "recommended_ions": {
                    "monovalent": "Joung-Cheatham",
                },
                "known_limitations": [
                    "Some helix over-stabilization"
                ],
                "mdp_specific_requirements": {
                    "rcoulomb": 1.0,
                    "rvdw": 1.0,
                    "vdw_modifier": "Potential-shift",
                    "DispCorr": "EnerPres",
                },
            },
        },
        "charmm_family": {
            "description": "CHARMM force fields",
            "charmm36m": {
                "full_name": "CHARMM36m",
                "type": "all-atom",
                "gromacs_directory": "charmm36m.ff",
                "suitable_for": {
                    "proteins": {
                        "rating": "excellent",
                        "notes": "Excellent for folded and IDPs",
                    }
                },
                "recommended_water_models": {
                    "primary": "TIP3P-CHARMM",
                },
                "recommended_ions": {
                    "monovalent": "CHARMM36 ions",
                },
                "known_limitations": [
                    "Must use CHARMM-modified TIP3P"
                ],
                "mdp_specific_requirements": {
                    "rcoulomb": 1.2,
                    "rvdw": 1.2,
                    "vdw_modifier": "Force-switch",
                    "rvdw_switch": 1.0,
                    "DispCorr": "no",
                    "critical_note": (
                        "CHARMM36 requires different "
                        "cutoff settings"
                    ),
                },
            },
            "charmm36": {
                "full_name": "CHARMM36",
                "type": "all-atom",
                "gromacs_directory": "charmm36.ff",
                "suitable_for": {
                    "proteins": {
                        "rating": "very_good",
                        "notes": "Good for folded proteins",
                    }
                },
                "recommended_water_models": {
                    "primary": "TIP3P-CHARMM",
                },
                "recommended_ions": {
                    "monovalent": "CHARMM36 ions",
                },
                "known_limitations": [
                    "Must use CHARMM-modified TIP3P"
                ],
                "mdp_specific_requirements": {
                    "rcoulomb": 1.2,
                    "rvdw": 1.2,
                    "vdw_modifier": "Force-switch",
                    "rvdw_switch": 1.0,
                    "DispCorr": "no",
                },
            },
        },
        "gromos_family": {
            "description": "GROMOS force fields",
        },
        "opls_family": {
            "description": "OPLS force fields",
        },
    }
    with open(
        kb_dir / "forcefield_compatibility"
        / "force_fields.yaml",
        "w",
    ) as f:
        yaml.dump(force_fields, f)

    # ── Water models ─────────────────────────────────────
    water_models: dict[str, Any] = {
        "three_site_models": {
            "description": "Three-site water models",
            "TIP3P": {
                "sites": 3,
                "properties": {
                    "density_gcm3": 0.982,
                    "diffusion_coefficient_m2s": 5.19e-9,
                    "dielectric_constant": 94.0,
                },
                "compatible_force_fields": {
                    "primary": ["amber99sb-ildn"],
                    "acceptable": ["oplsaa"],
                },
                "gromacs_topology_file": "tip3p.itp",
                "gromacs_water_model_flag": "tip3p",
            },
            "TIP3P-CHARMM": {
                "sites": 3,
                "description": "CHARMM-modified TIP3P",
                "compatible_force_fields": {
                    "primary": ["charmm36", "charmm36m"],
                },
                "gromacs_topology_file": (
                    "tip3p_charmm.itp"
                ),
            },
        },
        "four_site_models": {
            "description": "Four-site water models",
        },
        "five_site_models": {
            "description": "Five-site water models",
        },
    }
    with open(
        kb_dir / "forcefield_compatibility"
        / "water_models.yaml",
        "w",
    ) as f:
        yaml.dump(water_models, f)

    # ── Ion parameters ───────────────────────────────────
    ion_parameters: dict[str, Any] = {
        "description": "Ion parameter sets",
        "Joung-Cheatham": {
            "reference": "Joung & Cheatham, JPCA 2008",
            "ions_covered": ["Na+", "K+", "Cl-"],
            "parameterized_with": ["TIP3P", "TIP4P-EW"],
            "compatible_water_models": [
                "TIP3P", "TIP4P-EW"
            ],
            "compatible_force_fields": [
                "amber99sb-ildn", "amber14sb"
            ],
        },
        "Aqvist": {
            "reference": "Aqvist, J Phys Chem 1990",
            "ions_covered": ["Na+", "K+", "Mg2+"],
            "parameterized_with": "SPC",
            "compatible_water_models": ["SPC", "SPCE"],
            "compatible_force_fields": [
                "gromos53a6", "gromos54a7"
            ],
            "not_recommended_with": ["TIP3P", "TIP4P"],
        },
    }
    with open(
        kb_dir / "forcefield_compatibility"
        / "ion_parameters.yaml",
        "w",
    ) as f:
        yaml.dump(ion_parameters, f)

    # ── Compatibility matrix ─────────────────────────────
    compatibility_matrix: dict[str, Any] = {
        "description": "Compatibility matrix",
        "protein_simulations": {
            "standard_folded_protein": [
                {
                    "force_field": "amber99sb-ildn",
                    "water": "TIP3P",
                    "ions": "Joung-Cheatham (TIP3P)",
                    "rating": "RECOMMENDED",
                    "notes": "Most widely used combination",
                },
                {
                    "force_field": "charmm36m",
                    "water": "TIP3P-CHARMM",
                    "ions": "CHARMM36 ions",
                    "rating": "RECOMMENDED",
                    "notes": "Excellent for folded proteins",
                },
            ]
        },
        "forbidden_combinations": [
            {
                "id": "FC001",
                "combination": {
                    "force_field": "charmm36",
                    "water": "TIP3P",
                },
                "reason": (
                    "Standard TIP3P lacks LJ on H "
                    "required by CHARMM36"
                ),
                "severity": "ERROR",
            },
            {
                "id": "FC003",
                "combination": {
                    "force_field": "charmm36",
                    "DispCorr": "EnerPres",
                },
                "reason": (
                    "CHARMM36 uses force-switch; "
                    "DispCorr double-counts long-range VdW"
                ),
                "severity": "ERROR",
            },
        ],
    }
    with open(
        kb_dir / "forcefield_compatibility"
        / "compatibility_matrix.yaml",
        "w",
    ) as f:
        yaml.dump(compatibility_matrix, f)

    # ── Box/solvation placeholder files ─────────────────
    for solvation_file in [
        "box_geometry.yaml",
        "solvation.yaml",
        "ionization.yaml",
        "validation_checks.yaml",
    ]:
        with open(
            kb_dir / "box_solvation_rules"
            / solvation_file,
            "w",
        ) as f:
            yaml.dump({"placeholder": True}, f)

    # ── Common mistakes ──────────────────────────────────
    common_mistakes: dict[str, Any] = {
        "description": "Common mistakes registry",
        "metadata": {
            "total_mistakes": 4,
        },
        "critical_errors": [
            {
                "id": "CM001",
                "name": "Berendsen barostat in production",
                "category": "barostat_thermostat",
                "severity": "ERROR",
                "applicable_phases": ["production_md"],
                "check_condition": (
                    "pcoupl == berendsen AND "
                    "phase == production_md"
                ),
                "message": (
                    "Berendsen barostat does not sample "
                    "correct NPT ensemble. Use "
                    "Parrinello-Rahman."
                ),
                "correction_suggestion": (
                    "Change pcoupl to parrinello-rahman."
                ),
            },
            {
                "id": "CM002",
                "name": (
                    "Position restraints in production"
                ),
                "category": "position_restraints",
                "severity": "ERROR",
                "applicable_phases": ["production_md"],
                "check_condition": (
                    "define contains POSRES AND "
                    "phase == production_md"
                ),
                "message": (
                    "Position restraints must be removed "
                    "for production MD."
                ),
                "correction_suggestion": (
                    "Remove define = -DPOSRES from "
                    "production MDP."
                ),
            },
            {
                "id": "CM005",
                "name": "Pressure coupling in NVT",
                "category": "barostat_thermostat",
                "severity": "ERROR",
                "applicable_phases": [
                    "nvt_equilibration"
                ],
                "check_condition": (
                    "pcoupl != no AND "
                    "phase == nvt_equilibration"
                ),
                "message": (
                    "Pressure coupling must be disabled "
                    "during NVT equilibration."
                ),
                "correction_suggestion": (
                    "Set pcoupl = no in NVT MDP file."
                ),
            },
        ],
        "warnings": [
            {
                "id": "CM004",
                "name": "No position restraints in NVT",
                "category": "position_restraints",
                "severity": "WARNING",
                "applicable_phases": [
                    "nvt_equilibration"
                ],
                "check_condition": (
                    "define does not contain POSRES AND "
                    "phase == nvt_equilibration"
                ),
                "message": (
                    "Position restraints strongly "
                    "recommended during NVT equilibration."
                ),
                "correction_suggestion": (
                    "Add define = -DPOSRES to NVT MDP."
                ),
            },
        ],
        "cross_phase_consistency_mistakes": {
            "CC001": {
                "id": "CC001",
                "name": (
                    "Temperature inconsistency "
                    "across phases"
                ),
                "category": "ensemble_consistency",
                "severity": "ERROR",
                "check_condition": (
                    "ref-t(NVT) != ref-t(NPT) OR "
                    "ref-t(NPT) != ref-t(Production)"
                ),
                "message": (
                    "Target temperature is inconsistent "
                    "across simulation phases."
                ),
                "correction_suggestion": (
                    "Set ref-t consistently in all "
                    "MDP files."
                ),
            },
        },
    }
    with open(kb_dir / "common_mistakes.yaml", "w") as f:
        yaml.dump(common_mistakes, f)

    return kb_dir


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """
    Create a minimal but valid knowledge base directory
    structure for testing. Uses the real knowledge base
    if available, otherwise builds a minimal one.
    """
    real_kb = Path("knowledge_base")
    if real_kb.exists():
        dest = tmp_path / "knowledge_base"
        shutil.copytree(real_kb, dest)
        return dest
    return _build_minimal_kb(tmp_path)


@pytest.fixture
def kb(kb_dir: Path) -> KnowledgeBase:
    """Return a loaded KnowledgeBase instance."""
    return KnowledgeBase(kb_dir)


# ─────────────────────────────────────────────
# Test Classes
# ─────────────────────────────────────────────

class TestKnowledgeBaseLoading:
    """Tests for successful knowledge base loading."""

    def test_loads_without_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Knowledge base loads from valid directory."""
        assert kb is not None

    def test_all_phases_loaded(
        self, kb: KnowledgeBase
    ) -> None:
        """All four simulation phases are loaded."""
        for phase in SimulationPhase:
            schema = kb.get_phase_schema(phase)
            assert schema is not None
            assert schema.description

    def test_force_fields_loaded(
        self, kb: KnowledgeBase
    ) -> None:
        """Force field definitions are loaded."""
        ff_list = kb.list_force_fields()
        assert len(ff_list) > 0

    def test_water_models_loaded(
        self, kb: KnowledgeBase
    ) -> None:
        """Water model definitions are loaded."""
        wm_list = kb.list_water_models()
        assert len(wm_list) > 0

    def test_compatibility_matrix_loaded(
        self, kb: KnowledgeBase
    ) -> None:
        """Compatibility matrix is loaded."""
        matrix = kb.get_compatibility_matrix()
        assert matrix is not None
        assert len(matrix.forbidden_combinations) > 0

    def test_common_mistakes_loaded(
        self, kb: KnowledgeBase
    ) -> None:
        """Common mistakes registry is loaded."""
        mistakes = kb.get_common_mistakes()
        assert len(mistakes.get_all()) > 0


class TestPhaseSchemaAccess:
    """Tests for phase schema accessor methods."""

    def test_get_em_phase_schema(
        self, kb: KnowledgeBase
    ) -> None:
        """EM phase schema is accessible and valid."""
        schema = kb.get_phase_schema(
            SimulationPhase.ENERGY_MINIMIZATION
        )
        assert schema.run_control is not None
        assert (
            "integrator"
            in schema.run_control.parameters
        )

    def test_em_integrator_is_enum(
        self, kb: KnowledgeBase
    ) -> None:
        """EM integrator parameter is typed as enum."""
        schema = kb.get_phase_schema(
            SimulationPhase.ENERGY_MINIMIZATION
        )
        integrator = (
            schema.run_control.parameters["integrator"]
        )
        assert integrator.type.value == "enum"
        assert "steep" in integrator.allowed_values

    def test_all_four_phases_have_run_control(
        self, kb: KnowledgeBase
    ) -> None:
        """All four phases have run_control section."""
        for phase in SimulationPhase:
            schema = kb.get_phase_schema(phase)
            assert schema.run_control is not None
            assert len(
                schema.run_control.parameters
            ) > 0

    def test_all_four_phases_have_constraints(
        self, kb: KnowledgeBase
    ) -> None:
        """All four phases have constraints section."""
        for phase in SimulationPhase:
            schema = kb.get_phase_schema(phase)
            assert schema.constraints is not None

    def test_invalid_phase_raises_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Requesting unknown phase raises error."""
        with pytest.raises(KnowledgeBaseError):
            kb.get_phase_schema("invalid_phase")  # type: ignore


class TestForceFieldAccess:
    """Tests for force field accessor methods."""

    def test_get_amber_force_field(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER ff99SB-ILDN is accessible."""
        ff = kb.get_force_field("amber99sb-ildn")
        assert ff.full_name == "AMBER ff99SB-ILDN"
        assert ff.type == "all-atom"

    def test_get_charmm36m_force_field(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36m is accessible."""
        ff = kb.get_force_field("charmm36m")
        assert ff.full_name == "CHARMM36m"

    def test_force_field_lookup_case_insensitive(
        self, kb: KnowledgeBase
    ) -> None:
        """Force field lookup is case-insensitive."""
        ff_lower = kb.get_force_field("amber99sb-ildn")
        ff_upper = kb.get_force_field("AMBER99SB-ILDN")
        assert ff_lower.full_name == ff_upper.full_name

    def test_unknown_force_field_raises_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Requesting unknown force field raises error."""
        with pytest.raises(
            KnowledgeBaseError,
            match="Unknown force field",
        ):
            kb.get_force_field("nonexistent_ff")

    def test_charmm36m_has_mdp_requirements(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36m has force-field-specific MDP reqs."""
        ff = kb.get_force_field("charmm36m")
        assert ff.mdp_specific_requirements is not None
        assert (
            ff.mdp_specific_requirements.rcoulomb == 1.2
        )
        assert (
            ff.mdp_specific_requirements.vdw_modifier
            == "Force-switch"
        )

    def test_amber_has_mdp_requirements(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER ff99SB-ILDN has MDP requirements."""
        ff = kb.get_force_field("amber99sb-ildn")
        assert ff.mdp_specific_requirements is not None
        assert (
            ff.mdp_specific_requirements.rcoulomb == 1.0
        )
        assert (
            ff.mdp_specific_requirements.vdw_modifier
            == "Potential-shift"
        )

    def test_list_force_fields_returns_all(
        self, kb: KnowledgeBase
    ) -> None:
        """list_force_fields returns non-empty list."""
        ff_list = kb.list_force_fields()
        assert isinstance(ff_list, list)
        assert len(ff_list) >= 2
        assert "amber99sb-ildn" in ff_list
        assert "charmm36m" in ff_list


class TestWaterModelAccess:
    """Tests for water model accessor methods."""

    def test_get_tip3p_water_model(
        self, kb: KnowledgeBase
    ) -> None:
        """TIP3P water model is accessible."""
        wm = kb.get_water_model("TIP3P")
        assert wm.sites == 3

    def test_water_model_lookup_case_insensitive(
        self, kb: KnowledgeBase
    ) -> None:
        """Water model lookup is case-insensitive."""
        wm_lower = kb.get_water_model("tip3p")
        wm_upper = kb.get_water_model("TIP3P")
        assert wm_lower.sites == wm_upper.sites

    def test_unknown_water_model_raises_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Requesting unknown water model raises error."""
        with pytest.raises(
            KnowledgeBaseError,
            match="Unknown water model",
        ):
            kb.get_water_model("nonexistent_water")

    def test_list_water_models_returns_all(
        self, kb: KnowledgeBase
    ) -> None:
        """list_water_models returns non-empty list."""
        wm_list = kb.list_water_models()
        assert isinstance(wm_list, list)
        assert len(wm_list) >= 1


class TestIonParameterAccess:
    """Tests for ion parameter accessor methods."""

    def test_get_joung_cheatham(
        self, kb: KnowledgeBase
    ) -> None:
        """Joung-Cheatham ion parameters are accessible."""
        ions = kb.get_ion_parameters("joung-cheatham")
        assert "Na+" in ions.ions_covered

    def test_get_aqvist(
        self, kb: KnowledgeBase
    ) -> None:
        """Aqvist ion parameters are accessible."""
        ions = kb.get_ion_parameters("aqvist")
        assert ions.reference is not None

    def test_unknown_ion_params_raises_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Requesting unknown ion params raises error."""
        with pytest.raises(
            KnowledgeBaseError,
            match="Unknown ion parameter set",
        ):
            kb.get_ion_parameters("nonexistent_ions")


class TestCompatibilityChecks:
    """Tests for compatibility matrix checks."""

    def test_charmm36_standard_tip3p_is_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36 + standard TIP3P is forbidden."""
        is_forbidden, reason = (
            kb.is_combination_forbidden(
                force_field="charmm36",
                water_model="TIP3P",
            )
        )
        assert is_forbidden is True
        assert reason is not None
        assert len(reason) > 0

    def test_charmm36_dispcorr_is_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36 + DispCorr=EnerPres is forbidden."""
        is_forbidden, reason = (
            kb.is_combination_forbidden(
                force_field="charmm36",
                disp_corr="EnerPres",
            )
        )
        assert is_forbidden is True

    def test_amber_tip3p_is_not_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER + TIP3P is not a forbidden combination."""
        is_forbidden, _ = kb.is_combination_forbidden(
            force_field="amber99sb-ildn",
            water_model="TIP3P",
        )
        assert is_forbidden is False

    def test_compatibility_matrix_has_entries(
        self, kb: KnowledgeBase
    ) -> None:
        """Compatibility matrix has protein simulation entries."""
        matrix = kb.get_compatibility_matrix()
        assert "standard_folded_protein" in (
            matrix.protein_simulations
        )
        entries = matrix.protein_simulations[
            "standard_folded_protein"
        ]
        assert len(entries) >= 1

    def test_recommended_entry_has_correct_rating(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER+TIP3P entry has RECOMMENDED rating."""
        matrix = kb.get_compatibility_matrix()
        entries = matrix.protein_simulations[
            "standard_folded_protein"
        ]
        amber_entry = next(
            (
                e for e in entries
                if e.force_field == "amber99sb-ildn"
            ),
            None,
        )
        assert amber_entry is not None
        assert (
            amber_entry.rating
            == CompatibilityRating.RECOMMENDED
        )


class TestCommonMistakesRegistry:
    """Tests for common mistakes registry."""

    def test_cm001_exists(
        self, kb: KnowledgeBase
    ) -> None:
        """CM001 (Berendsen in production) is in registry."""
        mistakes = kb.get_common_mistakes()
        cm001 = mistakes.get_by_id("CM001")
        assert cm001 is not None
        assert cm001.severity == Severity.ERROR

    def test_cm004_is_warning(
        self, kb: KnowledgeBase
    ) -> None:
        """CM004 (no POSRES in NVT) is a WARNING."""
        mistakes = kb.get_common_mistakes()
        cm004 = mistakes.get_by_id("CM004")
        assert cm004 is not None
        assert cm004.severity == Severity.WARNING

    def test_cc001_loaded_from_cross_phase(
        self, kb: KnowledgeBase
    ) -> None:
        """CC001 cross-phase mistake is loaded."""
        mistakes = kb.get_common_mistakes()
        cc001 = mistakes.get_by_id("CC001")
        assert cc001 is not None
        assert cc001.severity == Severity.ERROR

    def test_get_all_returns_all_mistakes(
        self, kb: KnowledgeBase
    ) -> None:
        """get_all() returns both errors and warnings."""
        mistakes = kb.get_common_mistakes()
        all_mistakes = mistakes.get_all()
        errors = mistakes.critical_errors
        warnings = mistakes.warnings
        assert len(all_mistakes) == (
            len(errors) + len(warnings)
        )

    def test_unknown_mistake_id_returns_none(
        self, kb: KnowledgeBase
    ) -> None:
        """get_by_id() returns None for unknown ID."""
        mistakes = kb.get_common_mistakes()
        result = mistakes.get_by_id("CM999")
        assert result is None

    def test_all_critical_errors_have_error_severity(
        self, kb: KnowledgeBase
    ) -> None:
        """All entries in critical_errors have ERROR severity."""
        mistakes = kb.get_common_mistakes()
        for mistake in mistakes.critical_errors:
            assert mistake.severity == Severity.ERROR, (
                f"Expected ERROR for {mistake.id}, "
                f"got {mistake.severity}"
            )

    def test_all_warnings_have_warning_severity(
        self, kb: KnowledgeBase
    ) -> None:
        """All entries in warnings have WARNING severity."""
        mistakes = kb.get_common_mistakes()
        for mistake in mistakes.warnings:
            assert mistake.severity == Severity.WARNING, (
                f"Expected WARNING for {mistake.id}, "
                f"got {mistake.severity}"
            )


class TestErrorHandling:
    """Tests for error handling in the loader."""

    def test_missing_directory_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Non-existent directory raises KnowledgeBaseError."""
        with pytest.raises(KnowledgeBaseError):
            KnowledgeBase(tmp_path / "nonexistent")

    def test_missing_file_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Missing required file raises KnowledgeBaseError."""
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)
        (broken_kb / "common_mistakes.yaml").unlink()

        with pytest.raises(
            KnowledgeBaseError, match="Missing"
        ):
            KnowledgeBase(broken_kb)

    def test_malformed_yaml_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Malformed YAML raises KnowledgeBaseError."""
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)

        with open(
            broken_kb / "common_mistakes.yaml", "w"
        ) as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(
            KnowledgeBaseError,
            match="YAML parse error",
        ):
            KnowledgeBase(broken_kb)

    def test_empty_yaml_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Empty YAML file raises KnowledgeBaseError."""
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)

        with open(
            broken_kb / "common_mistakes.yaml", "w"
        ) as f:
            f.write("")

        with pytest.raises(
            KnowledgeBaseError, match="Empty"
        ):
            KnowledgeBase(broken_kb)

    def test_wrong_root_key_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Phase file with wrong root key raises error."""
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)

        # Write a phase file with wrong root key
        with open(
            broken_kb / "phases"
            / "energy_minimization.yaml",
            "w",
        ) as f:
            yaml.dump(
                {"wrong_key": {"description": "test"}},
                f,
            )

        with pytest.raises(
            KnowledgeBaseError,
            match="Expected root key",
        ):
            KnowledgeBase(broken_kb)


class TestSingleton:
    """Tests for the singleton factory function."""

    def test_get_knowledge_base_returns_same_instance(
        self, kb_dir: Path
    ) -> None:
        """get_knowledge_base() returns same instance."""
        from shared.knowledge_base_loader import (
            get_knowledge_base,
        )

        get_knowledge_base.cache_clear()

        kb1 = get_knowledge_base(str(kb_dir))
        kb2 = get_knowledge_base(str(kb_dir))
        assert kb1 is kb2

        get_knowledge_base.cache_clear()

    def test_different_dirs_return_different_instances(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Different directories return different instances."""
        from shared.knowledge_base_loader import (
            get_knowledge_base,
        )

        get_knowledge_base.cache_clear()

        # Build a second minimal KB in a different location
        kb_dir_2 = _build_minimal_kb(
            tmp_path / "second"
        )

        kb1 = get_knowledge_base(str(kb_dir))
        kb2 = get_knowledge_base(str(kb_dir_2))
        assert kb1 is not kb2

        get_knowledge_base.cache_clear()