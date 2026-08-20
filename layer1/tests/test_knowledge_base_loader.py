# layer1/tests/test_knowledge_base_loader.py

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

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
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """
    Create a minimal but valid knowledge base directory
    structure for testing.
    """
    # Copy the real knowledge base into tmp_path
    # In practice, point this at the actual knowledge_base/ dir
    real_kb = Path("knowledge_base")
    if real_kb.exists():
        shutil.copytree(real_kb, tmp_path / "knowledge_base")
        return tmp_path / "knowledge_base"

    # If real KB not yet present, build a minimal valid one
    return _build_minimal_kb(tmp_path)


@pytest.fixture
def kb(kb_dir: Path) -> KnowledgeBase:
    """Return a loaded KnowledgeBase instance."""
    return KnowledgeBase(kb_dir)


def _build_minimal_kb(base: Path) -> Path:
    """
    Build a minimal valid knowledge base for unit testing.
    Contains only the fields required by Pydantic models.
    """
    kb_dir = base / "knowledge_base"

    # Create directory structure
    (kb_dir / "phases").mkdir(parents=True)
    (kb_dir / "forcefield_compatibility").mkdir(parents=True)
    (kb_dir / "box_solvation_rules").mkdir(parents=True)

    # Minimal EM phase schema
    em_schema = {
        "description": "Energy minimization phase",
        "run_control": {
            "parameters": {
                "integrator": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["steep", "cg", "l-bfgs"],
                    "recommended": "steep",
                    "forbidden_values": ["md", "sd"],
                    "notes": "Minimization integrator"
                },
                "nsteps": {
                    "type": "integer",
                    "required": True,
                    "min": 500,
                    "max": 50000,
                    "recommended": 5000,
                    "notes": "Maximum minimization steps"
                },
                "emtol": {
                    "type": "float",
                    "required": True,
                    "min": 1.0,
                    "max": 1000.0,
                    "recommended": 1000.0,
                    "unit": "kJ/mol/nm",
                    "notes": "Force convergence tolerance"
                }
            }
        },
        "output_control": {
            "parameters": {
                "nstlog": {
                    "type": "integer",
                    "required": True,
                    "min": 1,
                    "max": 1000,
                    "recommended": 500,
                    "notes": "Log output frequency"
                }
            }
        },
        "neighbor_searching": {
            "parameters": {
                "cutoff-scheme": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["Verlet"],
                    "forbidden_values": ["group"],
                    "notes": "Verlet mandatory"
                }
            }
        },
        "electrostatics": {
            "parameters": {
                "coulombtype": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["PME", "Cut-off"],
                    "recommended": "PME",
                    "notes": "Electrostatics method"
                },
                "rcoulomb": {
                    "type": "float",
                    "required": True,
                    "min": 0.8,
                    "max": 1.4,
                    "recommended": 1.0,
                    "unit": "nm",
                    "must_equal": "rvdw",
                    "notes": "Coulomb cutoff"
                }
            }
        },
        "vdw": {
            "parameters": {
                "rvdw": {
                    "type": "float",
                    "required": True,
                    "min": 0.8,
                    "max": 1.4,
                    "recommended": 1.0,
                    "unit": "nm",
                    "notes": "VdW cutoff"
                }
            }
        },
        "temperature_coupling": {
            "parameters": {
                "tcoupl": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["no"],
                    "forbidden_values": [
                        "berendsen", "v-rescale", "nose-hoover"
                    ],
                    "notes": "No T-coupling during EM"
                }
            }
        },
        "pressure_coupling": {
            "parameters": {
                "pcoupl": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["no"],
                    "forbidden_values": [
                        "berendsen", "parrinello-rahman"
                    ],
                    "notes": "No P-coupling during EM"
                }
            }
        },
        "velocity_generation": {
            "parameters": {
                "gen-vel": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["no"],
                    "forbidden_values": ["yes"],
                    "notes": "No velocities during EM"
                }
            }
        },
        "constraints": {
            "parameters": {
                "constraints": {
                    "type": "enum",
                    "required": True,
                    "allowed_values": ["none"],
                    "notes": "No constraints during EM"
                }
            }
        }
    }

    # Write minimal phase files (same structure for all phases
    # in this minimal fixture — real files differ per phase)
    for phase_file in [
        "energy_minimization.yaml",
        "nvt_equilibration.yaml",
        "npt_equilibration.yaml",
        "production_md.yaml",
    ]:
        with open(kb_dir / "phases" / phase_file, "w") as f:
            yaml.dump(em_schema, f)

    # Minimal force fields
    force_fields = {
        "amber99sb-ildn": {
            "full_name": "AMBER ff99SB-ILDN",
            "type": "all-atom",
            "gromacs_directory": "amber99sb-ildn.ff",
            "suitable_for": {
                "proteins": {
                    "rating": "very_good",
                    "notes": "Good for folded proteins"
                }
            },
            "recommended_water_models": {
                "primary": "TIP3P",
                "acceptable": ["TIP4P-EW"]
            },
            "recommended_ions": {
                "monovalent": "Joung-Cheatham"
            },
            "known_limitations": [
                "Some helix over-stabilization"
            ],
            "mdp_specific_requirements": {
                "rcoulomb": 1.0,
                "rvdw": 1.0,
                "vdw_modifier": "Potential-shift",
                "DispCorr": "EnerPres"
            }
        },
        "charmm36m": {
            "full_name": "CHARMM36m",
            "type": "all-atom",
            "gromacs_directory": "charmm36m.ff",
            "suitable_for": {
                "proteins": {
                    "rating": "excellent",
                    "notes": "Excellent for folded and IDPs"
                }
            },
            "recommended_water_models": {
                "primary": "TIP3P-CHARMM"
            },
            "recommended_ions": {
                "monovalent": "CHARMM36 ions"
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
                    "CHARMM36 requires different cutoff settings"
                )
            }
        }
    }
    with open(
        kb_dir / "forcefield_compatibility" / "force_fields.yaml",
        "w"
    ) as f:
        yaml.dump(force_fields, f)

    # Minimal water models
    water_models = {
        "tip3p": {
            "sites": 3,
            "properties": {
                "density_gcm3": 0.982,
                "diffusion_coefficient_m2s": 5.19e-9,
                "dielectric_constant": 94.0
            },
            "compatible_force_fields": {
                "primary": ["amber99sb-ildn"],
                "acceptable": ["oplsaa"]
            },
            "gromacs_topology_file": "tip3p.itp",
            "gromacs_water_model_flag": "tip3p"
        },
        "tip3p-charmm": {
            "sites": 3,
            "description": "CHARMM-modified TIP3P",
            "compatible_force_fields": {
                "primary": ["charmm36", "charmm36m"]
            },
            "gromacs_topology_file": "tip3p_charmm.itp"
        }
    }
    with open(
        kb_dir / "forcefield_compatibility" / "water_models.yaml",
        "w"
    ) as f:
        yaml.dump(water_models, f)

    # Minimal ion parameters
    ion_parameters = {
        "joung-cheatham": {
            "reference": "Joung & Cheatham, JPCA 2008",
            "ions_covered": ["Na+", "K+", "Cl-"],
            "parameterized_with": ["TIP3P", "TIP4P-EW"],
            "compatible_water_models": ["TIP3P", "TIP4P-EW"],
            "compatible_force_fields": [
                "amber99sb-ildn", "amber14sb"
            ]
        },
        "aqvist": {
            "reference": "Åqvist, J Phys Chem 1990",
            "ions_covered": ["Na+", "K+", "Mg2+"],
            "parameterized_with": "SPC",
            "compatible_water_models": ["SPC", "SPCE"],
            "compatible_force_fields": [
                "gromos53a6", "gromos54a7"
            ],
            "not_recommended_with": ["TIP3P", "TIP4P"]
        }
    }
    with open(
        kb_dir / "forcefield_compatibility" / "ion_parameters.yaml",
        "w"
    ) as f:
        yaml.dump(ion_parameters, f)

    # Minimal compatibility matrix
    compatibility_matrix = {
        "protein_simulations": {
            "standard_folded_protein": [
                {
                    "force_field": "amber99sb-ildn",
                    "water": "TIP3P",
                    "ions": "Joung-Cheatham (TIP3P)",
                    "rating": "RECOMMENDED",
                    "notes": "Most widely used combination"
                },
                {
                    "force_field": "charmm36m",
                    "water": "TIP3P-CHARMM",
                    "ions": "CHARMM36 ions",
                    "rating": "RECOMMENDED",
                    "notes": "Excellent for folded proteins"
                }
            ]
        },
        "forbidden_combinations": [
            {
                "combination": {
                    "force_field": "charmm36",
                    "water": "TIP3P"
                },
                "reason": (
                    "Standard TIP3P lacks LJ on H required "
                    "by CHARMM36"
                ),
                "severity": "ERROR"
            },
            {
                "combination": {
                    "force_field": "charmm36",
                    "DispCorr": "EnerPres"
                },
                "reason": (
                    "CHARMM36 uses force-switch; DispCorr "
                    "double-counts long-range VdW"
                ),
                "severity": "ERROR"
            }
        ]
    }
    with open(
        kb_dir / "forcefield_compatibility"
        / "compatibility_matrix.yaml",
        "w"
    ) as f:
        yaml.dump(compatibility_matrix, f)

    # Minimal box/solvation files (content not validated
    # by KnowledgeBase loader directly — used by tools)
    for solvation_file in [
        "box_geometry.yaml",
        "solvation.yaml",
        "ionization.yaml",
        "validation_checks.yaml",
    ]:
        with open(
            kb_dir / "box_solvation_rules" / solvation_file, "w"
        ) as f:
            yaml.dump({"placeholder": True}, f)

    # Minimal common mistakes
    common_mistakes = {
        "critical_errors": [
            {
                "id": "CM001",
                "name": "Berendsen barostat in production",
                "check_description": (
                    "pcoupl == berendsen AND "
                    "phase == production_md"
                ),
                "message": (
                    "Berendsen barostat does not sample correct "
                    "NPT ensemble. Use Parrinello-Rahman."
                ),
                "severity": "ERROR",
                "applicable_phases": ["production_md"]
            },
            {
                "id": "CM002",
                "name": "Position restraints in production",
                "check_description": (
                    "define contains POSRES AND "
                    "phase == production_md"
                ),
                "message": (
                    "Position restraints must be removed "
                    "for production MD."
                ),
                "severity": "ERROR",
                "applicable_phases": ["production_md"]
            },
            {
                "id": "CM005",
                "name": "Pressure coupling in NVT",
                "check_description": (
                    "pcoupl != no AND "
                    "phase == nvt_equilibration"
                ),
                "message": (
                    "Pressure coupling must be disabled "
                    "during NVT equilibration."
                ),
                "severity": "ERROR",
                "applicable_phases": ["nvt_equilibration"]
            }
        ],
        "warnings": [
            {
                "id": "CM004",
                "name": "No position restraints in NVT",
                "check_description": (
                    "define does not contain POSRES AND "
                    "phase == nvt_equilibration"
                ),
                "message": (
                    "Position restraints strongly recommended "
                    "during NVT equilibration."
                ),
                "severity": "WARNING",
                "applicable_phases": ["nvt_equilibration"]
            }
        ]
    }
    with open(kb_dir / "common_mistakes.yaml", "w") as f:
        yaml.dump(common_mistakes, f)

    return kb_dir


# ─────────────────────────────────────────────
# Test Classes
# ─────────────────────────────────────────────

class TestKnowledgeBaseLoading:
    """Tests for successful knowledge base loading."""

    def test_loads_without_error(self, kb: KnowledgeBase) -> None:
        """Knowledge base loads from valid directory."""
        assert kb is not None

    def test_all_phases_loaded(self, kb: KnowledgeBase) -> None:
        """All four simulation phases are loaded."""
        for phase in SimulationPhase:
            schema = kb.get_phase_schema(phase)
            assert schema is not None
            assert schema.description

    def test_force_fields_loaded(self, kb: KnowledgeBase) -> None:
        """Force field definitions are loaded."""
        ff_list = kb.list_force_fields()
        assert len(ff_list) > 0

    def test_water_models_loaded(self, kb: KnowledgeBase) -> None:
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
        assert "integrator" in schema.run_control.parameters
        assert "nsteps" in schema.run_control.parameters

    def test_em_integrator_is_enum(
        self, kb: KnowledgeBase
    ) -> None:
        """EM integrator parameter is correctly typed as enum."""
        schema = kb.get_phase_schema(
            SimulationPhase.ENERGY_MINIMIZATION
        )
        integrator = schema.run_control.parameters["integrator"]
        assert integrator.type.value == "enum"
        assert "steep" in integrator.allowed_values
        assert "md" in integrator.forbidden_values

    def test_em_nsteps_has_bounds(
        self, kb: KnowledgeBase
    ) -> None:
        """EM nsteps parameter has min and max bounds."""
        schema = kb.get_phase_schema(
            SimulationPhase.ENERGY_MINIMIZATION
        )
        nsteps = schema.run_control.parameters["nsteps"]
        assert nsteps.min == 500
        assert nsteps.max == 50000

    def test_invalid_phase_raises_error(
        self, kb: KnowledgeBase
    ) -> None:
        """Requesting unknown phase raises KnowledgeBaseError."""
        with pytest.raises(KnowledgeBaseError):
            kb.get_phase_schema("invalid_phase")


class TestForceFieldAccess:
    """Tests for force field accessor methods."""

    def test_get_amber_force_field(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER ff99SB-ILDN is accessible."""
        ff = kb.get_force_field("amber99sb-ildn")
        assert ff.full_name == "AMBER ff99SB-ILDN"
        assert ff.type == "all-atom"

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
        """Requesting unknown force field raises KnowledgeBaseError."""
        with pytest.raises(KnowledgeBaseError, match="Unknown force field"):
            kb.get_force_field("nonexistent_ff")

    def test_charmm36m_has_mdp_requirements(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36m has force-field-specific MDP requirements."""
        ff = kb.get_force_field("charmm36m")
        assert ff.mdp_specific_requirements is not None
        assert ff.mdp_specific_requirements.rcoulomb == 1.2
        assert (
            ff.mdp_specific_requirements.vdw_modifier
            == "Force-switch"
        )


class TestCompatibilityChecks:
    """Tests for compatibility matrix and forbidden combinations."""

    def test_charmm36_standard_tip3p_is_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36 + standard TIP3P is a forbidden combination."""
        is_forbidden, reason = kb.is_combination_forbidden(
            force_field="charmm36",
            water_model="TIP3P"
        )
        assert is_forbidden is True
        assert reason is not None
        assert len(reason) > 0

    def test_charmm36_dispcorr_is_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """CHARMM36 + DispCorr=EnerPres is a forbidden combination."""
        is_forbidden, reason = kb.is_combination_forbidden(
            force_field="charmm36",
            disp_corr="EnerPres"
        )
        assert is_forbidden is True

    def test_amber_tip3p_is_not_forbidden(
        self, kb: KnowledgeBase
    ) -> None:
        """AMBER + TIP3P is not a forbidden combination."""
        is_forbidden, _ = kb.is_combination_forbidden(
            force_field="amber99sb-ildn",
            water_model="TIP3P"
        )
        assert is_forbidden is False


class TestCommonMistakesRegistry:
    """Tests for common mistakes registry."""

    def test_cm001_exists(self, kb: KnowledgeBase) -> None:
        """CM001 (Berendsen in production) is in registry."""
        mistakes = kb.get_common_mistakes()
        cm001 = mistakes.get_by_id("CM001")
        assert cm001 is not None
        assert cm001.severity == Severity.ERROR

    def test_cm004_is_warning(self, kb: KnowledgeBase) -> None:
        """CM004 (no POSRES in NVT) is a WARNING not ERROR."""
        mistakes = kb.get_common_mistakes()
        cm004 = mistakes.get_by_id("CM004")
        assert cm004 is not None
        assert cm004.severity == Severity.WARNING

    def test_get_all_returns_all_mistakes(
        self, kb: KnowledgeBase
    ) -> None:
        """get_all() returns both errors and warnings."""
        mistakes = kb.get_common_mistakes()
        all_mistakes = mistakes.get_all()
        errors = mistakes.critical_errors
        warnings = mistakes.warnings
        assert len(all_mistakes) == len(errors) + len(warnings)

    def test_unknown_mistake_id_returns_none(
        self, kb: KnowledgeBase
    ) -> None:
        """get_by_id() returns None for unknown ID."""
        mistakes = kb.get_common_mistakes()
        result = mistakes.get_by_id("CM999")
        assert result is None


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
        # Copy KB and remove one required file
        import shutil
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)
        (broken_kb / "common_mistakes.yaml").unlink()

        with pytest.raises(KnowledgeBaseError, match="Missing"):
            KnowledgeBase(broken_kb)

    def test_malformed_yaml_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Malformed YAML raises KnowledgeBaseError."""
        import shutil
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)

        # Write invalid YAML
        with open(
            broken_kb / "common_mistakes.yaml", "w"
        ) as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(KnowledgeBaseError, match="YAML parse error"):
            KnowledgeBase(broken_kb)

    def test_empty_yaml_raises_error(
        self, kb_dir: Path, tmp_path: Path
    ) -> None:
        """Empty YAML file raises KnowledgeBaseError."""
        import shutil
        broken_kb = tmp_path / "broken_kb"
        shutil.copytree(kb_dir, broken_kb)

        with open(
            broken_kb / "common_mistakes.yaml", "w"
        ) as f:
            f.write("")

        with pytest.raises(KnowledgeBaseError, match="Empty"):
            KnowledgeBase(broken_kb)


class TestSingleton:
    """Tests for the singleton factory function."""

    def test_get_knowledge_base_returns_same_instance(
        self, kb_dir: Path
    ) -> None:
        """get_knowledge_base() returns the same instance."""
        from shared.knowledge_base_loader import get_knowledge_base

        # Clear cache to avoid interference between tests
        get_knowledge_base.cache_clear()

        kb1 = get_knowledge_base(str(kb_dir))
        kb2 = get_knowledge_base(str(kb_dir))
        assert kb1 is kb2

        get_knowledge_base.cache_clear()