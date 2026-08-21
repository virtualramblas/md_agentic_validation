# layer1/tests/test_mdp_parser.py

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from shared.mdp_models import (
    MDPParseError,
    MDPParameter,
    ParsedMDP,
    _normalise_key,
)
from shared.mdp_parser import MDPParser


# ─────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────

@pytest.fixture
def parser() -> MDPParser:
    """Return a fresh MDPParser instance."""
    return MDPParser()


def _parse(
    content: str,
    parser: MDPParser,
    source_path: Path | None = None,
) -> ParsedMDP:
    """
    Helper: parse a dedented multi-line string.
    Strips leading indentation so test strings can
    be written naturally inside test functions.
    """
    return parser.parse_string(
        textwrap.dedent(content).strip(),
        source_path=source_path,
    )


# ─────────────────────────────────────────────
# Minimal valid MDP content used across tests
# ─────────────────────────────────────────────

MINIMAL_PRODUCTION_MDP = """\
; Production MD
integrator               = md
nsteps                   = 5000000
dt                       = 0.002
nstlog                   = 5000
nstenergy                = 5000
nstxout-compressed       = 5000
cutoff-scheme            = Verlet
nstlist                  = 10
rcoulomb                 = 1.0
rvdw                     = 1.0
pbc                      = xyz
coulombtype              = PME
pme-order                = 4
fourierspacing           = 0.16
vdwtype                  = Cut-off
vdw-modifier             = Potential-shift
DispCorr                 = EnerPres
tcoupl                   = nose-hoover
tc-grps                  = Protein Non-Protein
tau-t                    = 0.5   0.5
ref-t                    = 300   300
pcoupl                   = parrinello-rahman
pcoupltype               = isotropic
tau-p                    = 2.0
ref-p                    = 1.0
compressibility          = 4.5e-5
gen-vel                  = no
constraints              = h-bonds
constraint-algorithm     = LINCS
lincs-order              = 4
lincs-iter               = 1
define                   =
"""

MINIMAL_EM_MDP = """\
; Energy minimisation
integrator               = steep
nsteps                   = 50000
emtol                    = 1000.0
emstep                   = 0.01
nstlog                   = 500
nstenergy                = 500
cutoff-scheme            = Verlet
nstlist                  = 1
rcoulomb                 = 1.0
rvdw                     = 1.0
pbc                      = xyz
coulombtype              = PME
vdwtype                  = Cut-off
vdw-modifier             = Potential-shift
DispCorr                 = EnerPres
tcoupl                   = no
pcoupl                   = no
gen-vel                  = no
constraints              = none
define                   = -DPOSRES
"""


# ─────────────────────────────────────────────
# Test Classes
# ─────────────────────────────────────────────

class TestKeyNormalisation:
    """Tests for the _normalise_key utility."""

    def test_lowercase(self) -> None:
        """Keys are lowercased."""
        assert _normalise_key("NSTEPS") == "nsteps"

    def test_hyphen_preserved(self) -> None:
        """Hyphens are preserved."""
        assert (
            _normalise_key("vdw-modifier")
            == "vdw-modifier"
        )

    def test_underscore_to_hyphen(self) -> None:
        """Underscores are converted to hyphens."""
        assert (
            _normalise_key("vdw_modifier")
            == "vdw-modifier"
        )

    def test_mixed_case_and_underscore(self) -> None:
        """Mixed case and underscores are normalised."""
        assert (
            _normalise_key("VDW_Modifier")
            == "vdw-modifier"
        )

    def test_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace stripped."""
        assert (
            _normalise_key("  ref-t  ")
            == "ref-t"
        )

    def test_already_normalised(self) -> None:
        """Already normalised keys are unchanged."""
        assert (
            _normalise_key("ref-t") == "ref-t"
        )

    def test_single_word_key(self) -> None:
        """Single-word keys are just lowercased."""
        assert (
            _normalise_key("Integrator")
            == "integrator"
        )

    def test_mixed_hyphen_underscore(self) -> None:
        """Mixed hyphens and underscores normalised."""
        assert (
            _normalise_key("nstxout_compressed")
            == "nstxout-compressed"
        )


class TestMDPParseError:
    """Tests for MDPParseError construction."""

    def test_message_only(self) -> None:
        """Error with message only."""
        err = MDPParseError("bad line")
        assert "bad line" in str(err)
        assert err.source_path is None
        assert err.line_number is None

    def test_with_source_path(self) -> None:
        """Error includes source path in message."""
        err = MDPParseError(
            "bad line",
            source_path=Path("test.mdp"),
        )
        assert "test.mdp" in str(err)
        assert err.source_path == Path("test.mdp")

    def test_with_line_number(self) -> None:
        """Error includes line number in message."""
        err = MDPParseError(
            "bad line",
            line_number=42,
        )
        assert "42" in str(err)
        assert err.line_number == 42

    def test_with_path_and_line(self) -> None:
        """Error includes both path and line number."""
        err = MDPParseError(
            "bad line",
            source_path=Path("prod.mdp"),
            line_number=17,
        )
        msg = str(err)
        assert "prod.mdp" in msg
        assert "17" in msg
        assert "bad line" in msg


class TestBasicParsing:
    """Tests for basic MDP file parsing."""

    def test_parse_empty_string(
        self, parser: MDPParser
    ) -> None:
        """Empty string parses to empty ParsedMDP."""
        mdp = parser.parse_string("")
        assert len(mdp) == 0
        assert mdp.comments == []

    def test_parse_comments_only(
        self, parser: MDPParser
    ) -> None:
        """File with only comments has no params."""
        content = """\
            ; This is a comment
            ; Another comment
        """
        mdp = _parse(content, parser)
        assert len(mdp) == 0
        assert len(mdp.comments) == 2

    def test_parse_blank_lines_only(
        self, parser: MDPParser
    ) -> None:
        """File with only blank lines has no params."""
        mdp = parser.parse_string("\n\n\n")
        assert len(mdp) == 0

    def test_parse_single_parameter(
        self, parser: MDPParser
    ) -> None:
        """Single parameter is parsed correctly."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.has("integrator")
        assert mdp.get_value("integrator") == "md"

    def test_parse_minimal_production_mdp(
        self, parser: MDPParser
    ) -> None:
        """Full minimal production MDP parses."""
        mdp = parser.parse_string(
            MINIMAL_PRODUCTION_MDP
        )
        assert len(mdp) > 0
        assert mdp.has("integrator")
        assert mdp.has("nsteps")
        assert mdp.has("dt")

    def test_parse_minimal_em_mdp(
        self, parser: MDPParser
    ) -> None:
        """Full minimal EM MDP parses."""
        mdp = parser.parse_string(MINIMAL_EM_MDP)
        assert mdp.has("integrator")
        assert mdp.get_value("integrator") == "steep"

    def test_source_path_stored(
        self, parser: MDPParser
    ) -> None:
        """source_path is stored on ParsedMDP."""
        path = Path("test.mdp")
        mdp = parser.parse_string(
            "integrator = md",
            source_path=path,
        )
        assert mdp.source_path == path

    def test_source_path_none_by_default(
        self, parser: MDPParser
    ) -> None:
        """source_path is None when not provided."""
        mdp = parser.parse_string("integrator = md")
        assert mdp.source_path is None

    def test_parameter_count(
        self, parser: MDPParser
    ) -> None:
        """Parameter count matches unique keys."""
        mdp = parser.parse_string(
            MINIMAL_PRODUCTION_MDP
        )
        assert len(mdp) == len(mdp.keys())


class TestParameterValues:
    """Tests for parameter value access methods."""

    def test_get_value_string(
        self, parser: MDPParser
    ) -> None:
        """get_value returns string value."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_value("integrator") == "md"

    def test_get_value_absent_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_value returns None for absent key."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_value("nsteps") is None

    def test_get_value_empty_returns_empty_string(
        self, parser: MDPParser
    ) -> None:
        """
        get_value returns '' for explicitly empty
        parameter (e.g. 'define = ').
        """
        mdp = parser.parse_string("define = ")
        assert mdp.get_value("define") == ""
        assert mdp.get_value("define") is not None

    def test_get_value_empty_no_space(
        self, parser: MDPParser
    ) -> None:
        """
        get_value returns '' for 'define =' with
        no trailing space.
        """
        mdp = parser.parse_string("define =")
        assert mdp.get_value("define") == ""

    def test_get_float_integer_value(
        self, parser: MDPParser
    ) -> None:
        """get_float parses integer-valued param."""
        mdp = parser.parse_string("rcoulomb = 1")
        assert mdp.get_float("rcoulomb") == 1.0

    def test_get_float_decimal_value(
        self, parser: MDPParser
    ) -> None:
        """get_float parses decimal value."""
        mdp = parser.parse_string("dt = 0.002")
        assert mdp.get_float("dt") == pytest.approx(
            0.002
        )

    def test_get_float_scientific_notation(
        self, parser: MDPParser
    ) -> None:
        """get_float parses scientific notation."""
        mdp = parser.parse_string(
            "compressibility = 4.5e-5"
        )
        assert mdp.get_float(
            "compressibility"
        ) == pytest.approx(4.5e-5)

    def test_get_float_absent_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_float returns None for absent key."""
        mdp = parser.parse_string("dt = 0.002")
        assert mdp.get_float("nsteps") is None

    def test_get_float_non_numeric_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_float returns None for string value."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_float("integrator") is None

    def test_get_float_multi_value_returns_none(
        self, parser: MDPParser
    ) -> None:
        """
        get_float returns None for multi-value param.
        Use get_values() for multi-value parameters.
        """
        mdp = parser.parse_string(
            "tau-t = 0.5 0.5"
        )
        assert mdp.get_float("tau-t") is None

    def test_get_float_empty_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_float returns None for empty value."""
        mdp = parser.parse_string("define = ")
        assert mdp.get_float("define") is None

    def test_get_int_integer_value(
        self, parser: MDPParser
    ) -> None:
        """get_int parses integer value."""
        mdp = parser.parse_string("nsteps = 5000000")
        assert mdp.get_int("nsteps") == 5000000

    def test_get_int_absent_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_int returns None for absent key."""
        mdp = parser.parse_string("nsteps = 5000000")
        assert mdp.get_int("dt") is None

    def test_get_int_float_value_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_int returns None for float value."""
        mdp = parser.parse_string("dt = 0.002")
        assert mdp.get_int("dt") is None

    def test_get_int_string_value_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_int returns None for string value."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_int("integrator") is None

    def test_get_int_multi_value_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_int returns None for multi-value."""
        mdp = parser.parse_string(
            "ref-t = 300 300"
        )
        assert mdp.get_int("ref-t") is None

    def test_get_values_single(
        self, parser: MDPParser
    ) -> None:
        """get_values returns one-element list."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_values("integrator") == ["md"]

    def test_get_values_multi(
        self, parser: MDPParser
    ) -> None:
        """get_values returns all tokens."""
        mdp = parser.parse_string(
            "tau-t = 0.5 0.5"
        )
        assert mdp.get_values("tau-t") == [
            "0.5", "0.5"
        ]

    def test_get_values_multi_extra_spaces(
        self, parser: MDPParser
    ) -> None:
        """get_values splits on any whitespace."""
        mdp = parser.parse_string(
            "ref-t                    = 300   300"
        )
        assert mdp.get_values("ref-t") == [
            "300", "300"
        ]

    def test_get_values_empty_returns_empty_list(
        self, parser: MDPParser
    ) -> None:
        """get_values returns [] for empty value."""
        mdp = parser.parse_string("define = ")
        assert mdp.get_values("define") == []

    def test_get_values_absent_returns_empty_list(
        self, parser: MDPParser
    ) -> None:
        """get_values returns [] for absent key."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get_values("nsteps") == []

    def test_get_returns_mdp_parameter(
        self, parser: MDPParser
    ) -> None:
        """get() returns MDPParameter instance."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        param = mdp.get("integrator")
        assert param is not None
        assert isinstance(param, MDPParameter)
        assert param.key == "integrator"
        assert param.raw_value == "md"

    def test_get_absent_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get() returns None for absent key."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert mdp.get("nsteps") is None


class TestKeyNormalisationInParsing:
    """Tests for key normalisation during parsing."""

    def test_uppercase_key_normalised(
        self, parser: MDPParser
    ) -> None:
        """Uppercase key is normalised to lowercase."""
        mdp = parser.parse_string("NSTEPS = 5000")
        assert mdp.has("nsteps")
        assert mdp.get_value("nsteps") == "5000"

    def test_underscore_key_normalised(
        self, parser: MDPParser
    ) -> None:
        """Underscore key is normalised to hyphen."""
        mdp = parser.parse_string(
            "vdw_modifier = Potential-shift"
        )
        assert mdp.has("vdw-modifier")
        assert (
            mdp.get_value("vdw-modifier")
            == "Potential-shift"
        )

    def test_raw_key_preserved(
        self, parser: MDPParser
    ) -> None:
        """Original key is preserved in raw_key."""
        mdp = parser.parse_string(
            "VDW_Modifier = Potential-shift"
        )
        param = mdp.get("vdw-modifier")
        assert param is not None
        assert param.raw_key == "VDW_Modifier"
        assert param.key == "vdw-modifier"

    def test_lookup_by_hyphen_or_underscore(
        self, parser: MDPParser
    ) -> None:
        """
        Parameter written with underscore is
        accessible via hyphen lookup and vice versa.
        """
        mdp = parser.parse_string(
            "vdw_modifier = Potential-shift"
        )
        # Written with underscore, looked up with
        # hyphen
        assert (
            mdp.get_value("vdw-modifier")
            == "Potential-shift"
        )
        # Also accessible via underscore lookup
        assert (
            mdp.get_value("vdw_modifier")
            == "Potential-shift"
        )

    def test_lookup_case_insensitive(
        self, parser: MDPParser
    ) -> None:
        """Lookup is case-insensitive."""
        mdp = parser.parse_string(
            "DispCorr = EnerPres"
        )
        assert mdp.get_value("dispcorr") == "EnerPres"
        assert mdp.get_value("DISPCORR") == "EnerPres"
        assert mdp.get_value("DispCorr") == "EnerPres"

    def test_nstxout_compressed_normalised(
        self, parser: MDPParser
    ) -> None:
        """nstxout-compressed normalised correctly."""
        mdp = parser.parse_string(
            "nstxout-compressed = 5000"
        )
        assert mdp.has("nstxout-compressed")
        assert (
            mdp.get_value("nstxout-compressed")
            == "5000"
        )
        # Also accessible via underscore form
        assert mdp.has("nstxout_compressed")


class TestCommentHandling:
    """Tests for comment parsing."""

    def test_comment_line_not_in_parameters(
        self, parser: MDPParser
    ) -> None:
        """Comment lines are not parsed as params."""
        content = """\
            ; This is a comment
            integrator = md
        """
        mdp = _parse(content, parser)
        assert not mdp.has("; This is a comment")
        assert mdp.has("integrator")

    def test_comment_line_stored(
        self, parser: MDPParser
    ) -> None:
        """Comment lines are stored in comments."""
        content = """\
            ; Run control
            integrator = md
        """
        mdp = _parse(content, parser)
        assert len(mdp.comments) == 1
        assert "; Run control" in mdp.comments[0]

    def test_multiple_comments_stored(
        self, parser: MDPParser
    ) -> None:
        """Multiple comment lines all stored."""
        content = """\
            ; Comment 1
            ; Comment 2
            ; Comment 3
            integrator = md
        """
        mdp = _parse(content, parser)
        assert len(mdp.comments) == 3

    def test_inline_comment_stripped_from_value(
        self, parser: MDPParser
    ) -> None:
        """Inline comment is stripped from value."""
        mdp = parser.parse_string(
            "integrator = md ; leap-frog integrator"
        )
        assert mdp.get_value("integrator") == "md"

    def test_inline_comment_with_spaces(
        self, parser: MDPParser
    ) -> None:
        """Inline comment with spaces stripped."""
        mdp = parser.parse_string(
            "dt = 0.002   ; 2 fs timestep"
        )
        assert mdp.get_float("dt") == pytest.approx(
            0.002
        )

    def test_value_with_semicolon_in_comment(
        self, parser: MDPParser
    ) -> None:
        """
        Value is correctly separated from inline
        comment even when comment contains '='.
        """
        mdp = parser.parse_string(
            "nsteps = 5000000 ; 10 ns at dt=0.002"
        )
        assert mdp.get_int("nsteps") == 5000000

    def test_comment_only_file_has_no_params(
        self, parser: MDPParser
    ) -> None:
        """File with only comments has no params."""
        content = """\
            ; GROMACS MDP file
            ; Generated by CHARMM-GUI
            ; Force field: CHARMM36m
        """
        mdp = _parse(content, parser)
        assert len(mdp) == 0
        assert len(mdp.comments) == 3


class TestDuplicateKeys:
    """Tests for duplicate key handling."""

    def test_duplicate_key_last_wins(
        self, parser: MDPParser
    ) -> None:
        """
        Last occurrence of a duplicate key wins,
        matching GROMACS behaviour.
        """
        content = """\
            nsteps = 1000
            nsteps = 5000000
        """
        mdp = _parse(content, parser)
        assert mdp.get_int("nsteps") == 5000000

    def test_duplicate_key_count_is_one(
        self, parser: MDPParser
    ) -> None:
        """
        Duplicate key results in one entry in
        parameters dict, not two.
        """
        content = """\
            integrator = steep
            integrator = md
        """
        mdp = _parse(content, parser)
        assert len(mdp) == 1
        assert mdp.get_value("integrator") == "md"

    def test_duplicate_normalised_key(
        self, parser: MDPParser
    ) -> None:
        """
        Keys that normalise to the same value are
        treated as duplicates. Last wins.
        """
        content = """\
            vdw_modifier = Force-switch
            vdw-modifier = Potential-shift
        """
        mdp = _parse(content, parser)
        assert len(mdp) == 1
        assert (
            mdp.get_value("vdw-modifier")
            == "Potential-shift"
        )

    def test_duplicate_different_case(
        self, parser: MDPParser
    ) -> None:
        """
        Keys differing only in case are treated as
        duplicates. Last wins.
        """
        content = """\
            NSTEPS = 1000
            nsteps = 5000000
        """
        mdp = _parse(content, parser)
        assert len(mdp) == 1
        assert mdp.get_int("nsteps") == 5000000


class TestMultiValueParameters:
    """Tests for multi-value parameter handling."""

    def test_tau_t_two_values(
        self, parser: MDPParser
    ) -> None:
        """tau-t with two coupling groups."""
        mdp = parser.parse_string(
            "tau-t = 0.5 0.5"
        )
        assert mdp.get_values("tau-t") == [
            "0.5", "0.5"
        ]

    def test_ref_t_two_values(
        self, parser: MDPParser
    ) -> None:
        """ref-t with two coupling groups."""
        mdp = parser.parse_string(
            "ref-t = 300 300"
        )
        assert mdp.get_values("ref-t") == [
            "300", "300"
        ]

    def test_tc_grps_two_groups(
        self, parser: MDPParser
    ) -> None:
        """tc-grps with two group names."""
        mdp = parser.parse_string(
            "tc-grps = Protein Non-Protein"
        )
        assert mdp.get_values("tc-grps") == [
            "Protein", "Non-Protein"
        ]

    def test_multi_value_raw_value_preserved(
        self, parser: MDPParser
    ) -> None:
        """
        raw_value preserves original spacing for
        multi-value parameters.
        """
        mdp = parser.parse_string(
            "tau-t                    = 0.5   0.5"
        )
        param = mdp.get("tau-t")
        assert param is not None
        # raw_value is stripped but internal
        # spacing preserved
        assert param.raw_value == "0.5   0.5"

    def test_multi_value_get_float_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_float returns None for multi-value."""
        mdp = parser.parse_string(
            "ref-t = 300 300"
        )
        assert mdp.get_float("ref-t") is None

    def test_multi_value_get_int_returns_none(
        self, parser: MDPParser
    ) -> None:
        """get_int returns None for multi-value."""
        mdp = parser.parse_string(
            "ref-t = 300 300"
        )
        assert mdp.get_int("ref-t") is None

    def test_three_values(
        self, parser: MDPParser
    ) -> None:
        """Parameter with three values."""
        mdp = parser.parse_string(
            "ref-t = 300 300 300"
        )
        assert mdp.get_values("ref-t") == [
            "300", "300", "300"
        ]


class TestEmptyValues:
    """Tests for empty-value parameter handling."""

    def test_define_empty_with_space(
        self, parser: MDPParser
    ) -> None:
        """'define = ' returns empty string."""
        mdp = parser.parse_string("define = ")
        assert mdp.has("define")
        assert mdp.get_value("define") == ""

    def test_define_empty_no_space(
        self, parser: MDPParser
    ) -> None:
        """'define =' returns empty string."""
        mdp = parser.parse_string("define =")
        assert mdp.has("define")
        assert mdp.get_value("define") == ""

    def test_empty_value_not_none(
        self, parser: MDPParser
    ) -> None:
        """
        Empty value returns '' not None.
        None means absent; '' means explicitly empty.
        """
        mdp = parser.parse_string("define = ")
        result = mdp.get_value("define")
        assert result is not None
        assert result == ""

    def test_empty_value_get_values_empty_list(
        self, parser: MDPParser
    ) -> None:
        """get_values returns [] for empty value."""
        mdp = parser.parse_string("define = ")
        assert mdp.get_values("define") == []

    def test_empty_value_has_returns_true(
        self, parser: MDPParser
    ) -> None:
        """has() returns True for empty-value param."""
        mdp = parser.parse_string("define = ")
        assert mdp.has("define") is True

    def test_define_with_posres(
        self, parser: MDPParser
    ) -> None:
        """'define = -DPOSRES' parses correctly."""
        mdp = parser.parse_string(
            "define = -DPOSRES"
        )
        assert mdp.get_value("define") == "-DPOSRES"

    def test_define_empty_vs_absent(
        self, parser: MDPParser
    ) -> None:
        """
        Explicitly empty 'define = ' is distinct
        from absent 'define'. Empty returns ''.
        Absent returns None.
        """
        mdp_with_empty = parser.parse_string(
            "define = "
        )
        mdp_without = parser.parse_string(
            "integrator = md"
        )
        assert mdp_with_empty.get_value("define") == ""
        assert mdp_without.get_value("define") is None


class TestLineNumberTracking:
    """Tests for line number tracking in parameters."""

    def test_first_line_parameter(
        self, parser: MDPParser
    ) -> None:
        """Parameter on line 1 has line_number 1."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        param = mdp.get("integrator")
        assert param is not None
        assert param.line_number == 1

    def test_parameter_after_comments(
        self, parser: MDPParser
    ) -> None:
        """Line number accounts for comment lines."""
        content = (
            "; comment\n"
            "; another\n"
            "integrator = md\n"
        )
        mdp = parser.parse_string(content)
        param = mdp.get("integrator")
        assert param is not None
        assert param.line_number == 3

    def test_parameter_after_blank_lines(
        self, parser: MDPParser
    ) -> None:
        """Line number accounts for blank lines."""
        content = (
            "\n"
            "\n"
            "integrator = md\n"
        )
        mdp = parser.parse_string(content)
        param = mdp.get("integrator")
        assert param is not None
        assert param.line_number == 3

    def test_multiple_parameters_line_numbers(
        self, parser: MDPParser
    ) -> None:
        """Each parameter has correct line number."""
        content = (
            "integrator = md\n"
            "nsteps = 5000000\n"
            "dt = 0.002\n"
        )
        mdp = parser.parse_string(content)
        assert mdp.get("integrator").line_number == 1
        assert mdp.get("nsteps").line_number == 2
        assert mdp.get("dt").line_number == 3

    def test_duplicate_key_line_number_updated(
        self, parser: MDPParser
    ) -> None:
        """
        After duplicate key overwrite, line_number
        reflects the last (winning) occurrence.
        """
        content = (
            "nsteps = 1000\n"
            "nsteps = 5000000\n"
        )
        mdp = parser.parse_string(content)
        param = mdp.get("nsteps")
        assert param is not None
        assert param.line_number == 2


class TestErrorHandling:
    """Tests for parse error handling."""

    def test_line_without_equals_raises_error(
        self, parser: MDPParser
    ) -> None:
        """Line without '=' raises MDPParseError."""
        with pytest.raises(MDPParseError):
            parser.parse_string(
                "this is not valid"
            )

    def test_error_includes_line_number(
        self, parser: MDPParser
    ) -> None:
        """MDPParseError includes line number."""
        content = (
            "integrator = md\n"
            "this is not valid\n"
        )
        with pytest.raises(MDPParseError) as exc_info:
            parser.parse_string(content)
        assert exc_info.value.line_number == 2

    def test_error_includes_source_path(
        self, parser: MDPParser
    ) -> None:
        """MDPParseError includes source path."""
        path = Path("bad.mdp")
        with pytest.raises(MDPParseError) as exc_info:
            parser.parse_string(
                "not valid", source_path=path
            )
        assert exc_info.value.source_path == path

    def test_missing_file_raises_error(
        self, parser: MDPParser, tmp_path: Path
    ) -> None:
        """Non-existent file raises MDPParseError."""
        with pytest.raises(MDPParseError):
            parser.parse_file(
                tmp_path / "nonexistent.mdp"
            )

    def test_valid_file_after_error_line(
        self, parser: MDPParser
    ) -> None:
        """
        Error on line N is reported at line N,
        not at end of file.
        """
        content = (
            "integrator = md\n"
            "nsteps = 5000000\n"
            "bad line here\n"
            "dt = 0.002\n"
        )
        with pytest.raises(MDPParseError) as exc_info:
            parser.parse_string(content)
        assert exc_info.value.line_number == 3


class TestFileIO:
    """Tests for file-based parsing."""

    def test_parse_file_roundtrip(
        self,
        parser: MDPParser,
        tmp_path: Path,
    ) -> None:
        """
        Parameters written to file and parsed back
        match the original values.
        """
        mdp_path = tmp_path / "test.mdp"
        mdp_path.write_text(
            MINIMAL_PRODUCTION_MDP,
            encoding="utf-8",
        )
        mdp = parser.parse_file(mdp_path)
        assert mdp.source_path == mdp_path
        assert mdp.get_value("integrator") == "md"
        assert mdp.get_int("nsteps") == 5000000
        assert mdp.get_float("dt") == pytest.approx(
            0.002
        )

    def test_parse_file_stores_source_path(
        self,
        parser: MDPParser,
        tmp_path: Path,
    ) -> None:
        """parse_file stores the file path."""
        mdp_path = tmp_path / "prod.mdp"
        mdp_path.write_text(
            "integrator = md",
            encoding="utf-8",
        )
        mdp = parser.parse_file(mdp_path)
        assert mdp.source_path == mdp_path

    def test_parse_file_missing_raises_error(
        self,
        parser: MDPParser,
        tmp_path: Path,
    ) -> None:
        """Missing file raises MDPParseError."""
        with pytest.raises(MDPParseError):
            parser.parse_file(
                tmp_path / "missing.mdp"
            )

    def test_parse_em_file(
        self,
        parser: MDPParser,
        tmp_path: Path,
    ) -> None:
        """EM MDP file parses correctly from disk."""
        mdp_path = tmp_path / "em.mdp"
        mdp_path.write_text(
            MINIMAL_EM_MDP, encoding="utf-8"
        )
        mdp = parser.parse_file(mdp_path)
        assert (
            mdp.get_value("integrator") == "steep"
        )
        assert mdp.get_float("emtol") == pytest.approx(
            1000.0
        )
        assert (
            mdp.get_value("define") == "-DPOSRES"
        )


class TestContainerProtocol:
    """Tests for ParsedMDP container behaviour."""

    def test_len(self, parser: MDPParser) -> None:
        """len() returns parameter count."""
        mdp = parser.parse_string(
            "integrator = md\nnsteps = 5000000"
        )
        assert len(mdp) == 2

    def test_contains_present_key(
        self, parser: MDPParser
    ) -> None:
        """'key in mdp' returns True for present."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert "integrator" in mdp

    def test_contains_absent_key(
        self, parser: MDPParser
    ) -> None:
        """'key in mdp' returns False for absent."""
        mdp = parser.parse_string(
            "integrator = md"
        )
        assert "nsteps" not in mdp

    def test_contains_normalised_key(
        self, parser: MDPParser
    ) -> None:
        """'key in mdp' works with unnormalised key."""
        mdp = parser.parse_string(
            "vdw_modifier = Potential-shift"
        )
        assert "vdw-modifier" in mdp
        assert "VDW_MODIFIER" in mdp

    def test_keys_returns_all_keys(
        self, parser: MDPParser
    ) -> None:
        """keys() returns all normalised keys."""
        mdp = parser.parse_string(
            "integrator = md\nnsteps = 5000000"
        )
        keys = mdp.keys()
        assert "integrator" in keys
        assert "nsteps" in keys
        assert len(keys) == 2

    def test_repr(self, parser: MDPParser) -> None:
        """repr() includes source and param count."""
        mdp = parser.parse_string(
            "integrator = md",
            source_path=Path("test.mdp"),
        )
        r = repr(mdp)
        assert "test.mdp" in r
        assert "1" in r


class TestRealWorldMDPPatterns:
    """
    Tests for real-world MDP patterns encountered
    in GROMACS simulation setups.
    """

    def test_charmm36m_mdp_settings(
        self, parser: MDPParser
    ) -> None:
        """
        CHARMM36m-specific MDP settings parse
        correctly. Tests the exact settings that
        differ from AMBER defaults.
        """
        content = """\
            rcoulomb                 = 1.2
            rvdw                     = 1.2
            vdw-modifier             = Force-switch
            rvdw-switch              = 1.0
            DispCorr                 = no
        """
        mdp = _parse(content, parser)
        assert mdp.get_float("rcoulomb") == pytest.approx(1.2)
        assert mdp.get_float("rvdw") == pytest.approx(1.2)
        assert (
            mdp.get_value("vdw-modifier")
            == "Force-switch"
        )
        assert mdp.get_float("rvdw-switch") == pytest.approx(1.0)
        assert mdp.get_value("dispcorr") == "no"

    def test_amber_mdp_settings(
        self, parser: MDPParser
    ) -> None:
        """
        AMBER-specific MDP settings parse correctly.
        """
        content = """\
            rcoulomb                 = 1.0
            rvdw                     = 1.0
            vdw-modifier             = Potential-shift
            DispCorr                 = EnerPres
        """
        mdp = _parse(content, parser)
        assert mdp.get_float("rcoulomb") == pytest.approx(1.0)
        assert (
            mdp.get_value("vdw-modifier")
            == "Potential-shift"
        )
        assert (
            mdp.get_value("dispcorr") == "EnerPres"
        )

    def test_nvt_equilibration_settings(
        self, parser: MDPParser
    ) -> None:
        """NVT equilibration MDP settings."""
        content = """\
            integrator               = md
            nsteps                   = 50000
            dt                       = 0.002
            tcoupl                   = v-rescale
            tc-grps                  = Protein Non-Protein
            tau-t                    = 0.1   0.1
            ref-t                    = 300   300
            pcoupl                   = no
            gen-vel                  = yes
            gen-temp                 = 300
            gen-seed                 = -1
            constraints              = h-bonds
            define                   = -DPOSRES
        """
        mdp = _parse(content, parser)
        assert (
            mdp.get_value("integrator") == "md"
        )
        assert mdp.get_value("tcoupl") == "v-rescale"
        assert mdp.get_value("pcoupl") == "no"
        assert mdp.get_value("gen-vel") == "yes"
        assert mdp.get_values("ref-t") == [
            "300", "300"
        ]
        assert (
            mdp.get_value("define") == "-DPOSRES"
        )

    def test_npt_equilibration_settings(
        self, parser: MDPParser
    ) -> None:
        """NPT equilibration MDP settings."""
        content = """\
            integrator               = md
            continuation             = yes
            pcoupl                   = berendsen
            pcoupltype               = isotropic
            tau-p                    = 2.0
            ref-p                    = 1.0
            compressibility          = 4.5e-5
            gen-vel                  = no
            define                   = -DPOSRES
        """
        mdp = _parse(content, parser)
        assert (
            mdp.get_value("continuation") == "yes"
        )
        assert (
            mdp.get_value("pcoupl") == "berendsen"
        )
        assert mdp.get_float("tau-p") == pytest.approx(
            2.0
        )
        assert mdp.get_float(
            "compressibility"
        ) == pytest.approx(4.5e-5)
        assert mdp.get_value("gen-vel") == "no"

    def test_production_md_no_posres(
        self, parser: MDPParser
    ) -> None:
        """
        Production MD has empty define (no POSRES).
        This is a critical correctness check —
        position restraints must be absent in
        production MD.
        """
        content = """\
            integrator               = md
            pcoupl                   = parrinello-rahman
            tcoupl                   = nose-hoover
            gen-vel                  = no
            define                   =
        """
        mdp = _parse(content, parser)
        assert mdp.get_value("define") == ""
        assert mdp.get_value("define") is not None
        assert (
            mdp.get_value("pcoupl")
            == "parrinello-rahman"
        )

    def test_whitespace_heavy_mdp(
        self, parser: MDPParser
    ) -> None:
        """
        MDP files with heavy alignment whitespace
        (common in GROMACS-generated files) parse
        correctly.
        """
        mdp = parser.parse_string(
            MINIMAL_PRODUCTION_MDP
        )
        # All parameters accessible despite
        # heavy alignment whitespace
        assert mdp.get_value("integrator") == "md"
        assert mdp.get_int("nsteps") == 5000000
        assert mdp.get_float("dt") == pytest.approx(
            0.002
        )
        assert (
            mdp.get_value("coulombtype") == "PME"
        )
        assert (
            mdp.get_value("vdw-modifier")
            == "Potential-shift"
        )
        assert (
            mdp.get_value("dispcorr") == "EnerPres"
        )
        assert mdp.get_value("define") == ""