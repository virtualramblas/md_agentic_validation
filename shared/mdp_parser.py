# shared/mdp_parser.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from shared.mdp_models import (
    MDPParameter,
    MDPParseError,
    ParsedMDP,
    _normalise_key,
)

logger = logging.getLogger(__name__)


class MDPParser:
    """
    Parses GROMACS MDP files into ParsedMDP objects.

    Handles the full range of MDP file syntax:
      - Comment lines (starting with ';')
      - Blank lines
      - 'key = value' parameter lines
      - 'key = value1 value2 ...' multi-value lines
      - 'key = ' empty-value lines
      - Inline comments ('key = value ; comment')
      - Duplicate keys (last occurrence wins,
        matching GROMACS behaviour)
      - Keys with hyphens or underscores
        (normalised to hyphenated form)
      - Keys and values with surrounding whitespace

    Usage:
        parser = MDPParser()

        # Parse from file
        mdp = parser.parse_file(
            Path("production.mdp")
        )

        # Parse from string
        mdp = parser.parse_string(
            "integrator = md\\nnsteps = 5000000"
        )

        # Access parameters
        integrator = mdp.get_value("integrator")
        dt = mdp.get_float("dt")
        nsteps = mdp.get_int("nsteps")
        ref_t_values = mdp.get_values("ref-t")
    """

    # ─────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────

    def parse_file(
        self, path: Path
    ) -> ParsedMDP:
        """
        Parse a GROMACS MDP file from disk.

        Args:
            path: Path to the .mdp file.

        Returns:
            ParsedMDP containing all parameters
            and comments from the file.

        Raises:
            MDPParseError: If the file cannot be
                read (OS error) or contains a line
                that cannot be parsed.
        """
        path = Path(path)
        try:
            content = path.read_text(
                encoding="utf-8"
            )
        except OSError as e:
            raise MDPParseError(
                f"Cannot read MDP file: {e}",
                source_path=path,
            ) from e

        logger.debug(f"Parsing MDP file: {path}")
        return self.parse_string(
            content, source_path=path
        )

    def parse_string(
        self,
        content: str,
        source_path: Optional[Path] = None,
    ) -> ParsedMDP:
        """
        Parse GROMACS MDP content from a string.

        Args:
            content:     MDP file content as a string.
            source_path: Optional path to associate
                         with the parsed result and
                         any error messages.

        Returns:
            ParsedMDP containing all parameters
            and comments from the content.

        Raises:
            MDPParseError: If any line cannot be
                parsed as a valid MDP line.
        """
        parameters: dict[str, MDPParameter] = {}
        comments: list[str] = []

        lines = content.splitlines()

        for line_number, raw_line in enumerate(
            lines, start=1
        ):
            self._parse_line(
                raw_line=raw_line,
                line_number=line_number,
                parameters=parameters,
                comments=comments,
                source_path=source_path,
            )

        mdp = ParsedMDP(
            source_path=source_path,
            parameters=parameters,
            comments=comments,
        )

        n = len(parameters)
        src = (
            str(source_path)
            if source_path
            else "<string>"
        )
        logger.debug(
            f"Parsed {n} parameters from {src}"
        )

        return mdp

    # ─────────────────────────────────────────
    # Private — line parsing
    # ─────────────────────────────────────────

    def _parse_line(
        self,
        raw_line: str,
        line_number: int,
        parameters: dict[str, MDPParameter],
        comments: list[str],
        source_path: Optional[Path],
    ) -> None:
        """
        Parse a single MDP file line and update
        parameters or comments in place.

        Line types handled:
          1. Blank line          → ignored
          2. Comment line        → appended to comments
          3. Parameter line      → parsed and stored
          4. Unrecognised line   → MDPParseError raised

        Duplicate key handling:
          If a key already exists in parameters, the
          new value silently overwrites the old one.
          This matches GROMACS behaviour where the
          last occurrence of a key wins.

        Args:
            raw_line:    The raw line string from the
                         file (not stripped).
            line_number: 1-based line number for error
                         messages.
            parameters:  Dict to update in place.
            comments:    List to append comments to.
            source_path: For error messages.

        Raises:
            MDPParseError: If the line is not blank,
                not a comment, and not a valid
                'key = value' parameter line.
        """
        # Strip trailing whitespace only — preserve
        # leading whitespace detection for blank check
        stripped = raw_line.strip()

        # ── Blank line ───────────────────────────
        if not stripped:
            return

        # ── Comment line ─────────────────────────
        if stripped.startswith(";"):
            comments.append(stripped)
            return

        # ── Parameter line ───────────────────────
        # Must contain '=' to be a valid parameter
        if "=" not in stripped:
            raise MDPParseError(
                f"Unrecognised line (no '=' found): "
                f"{stripped!r}",
                source_path=source_path,
                line_number=line_number,
            )

        raw_key, raw_value = self._split_parameter(
            stripped,
            line_number=line_number,
            source_path=source_path,
        )

        normalised_key = _normalise_key(raw_key)

        if not normalised_key:
            raise MDPParseError(
                f"Empty parameter key on line: "
                f"{stripped!r}",
                source_path=source_path,
                line_number=line_number,
            )

        # Split value into tokens for multi-value
        # support. Empty raw_value → empty list.
        values = (
            raw_value.split()
            if raw_value
            else []
        )

        param = MDPParameter(
            key=normalised_key,
            raw_key=raw_key,
            raw_value=raw_value,
            values=values,
            line_number=line_number,
        )

        # Duplicate key: silently overwrite,
        # matching GROMACS last-occurrence-wins rule.
        if normalised_key in parameters:
            logger.debug(
                f"Duplicate key '{normalised_key}' "
                f"at line {line_number} — "
                f"overwriting previous value "
                f"'{parameters[normalised_key].raw_value}' "
                f"with '{raw_value}'"
            )

        parameters[normalised_key] = param

    def _split_parameter(
        self,
        line: str,
        line_number: int,
        source_path: Optional[Path],
    ) -> tuple[str, str]:
        """
        Split a parameter line into key and value.

        Handles:
          - 'key = value'
          - 'key = value ; inline comment'
          - 'key = value1 value2'
          - 'key = '  (empty value)
          - 'key ='   (empty value, no trailing space)

        The split is on the FIRST '=' only, so that
        values containing '=' are handled correctly
        (rare but possible in define strings).

        Inline comments ('; ...') are stripped from
        the value portion after splitting.

        Args:
            line:        Stripped parameter line
                         known to contain '='.
            line_number: For error messages.
            source_path: For error messages.

        Returns:
            (raw_key, raw_value) tuple where:
              raw_key   is stripped of whitespace
              raw_value is stripped of whitespace
                        and inline comments

        Raises:
            MDPParseError: If the key portion is
                empty after stripping.
        """
        # Split on first '=' only
        eq_index = line.index("=")
        raw_key = line[:eq_index].strip()
        value_portion = line[eq_index + 1:]

        # Strip inline comment from value portion.
        # Find the first ';' that is not inside
        # a quoted string. MDP files do not use
        # quoted strings so a simple find suffices.
        semicolon_index = value_portion.find(";")
        if semicolon_index != -1:
            value_portion = value_portion[
                :semicolon_index
            ]

        raw_value = value_portion.strip()

        if not raw_key:
            raise MDPParseError(
                f"Empty parameter key: {line!r}",
                source_path=source_path,
                line_number=line_number,
            )

        return raw_key, raw_value