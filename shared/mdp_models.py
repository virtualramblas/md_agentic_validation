# shared/mdp_models.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class MDPParseError(Exception):
    """
    Raised when an MDP file cannot be parsed.

    Includes the source path (if known) and the
    1-based line number where the error occurred
    (if applicable).

    Attributes:
        message:     Human-readable error description.
        source_path: Path to the MDP file, or None if
                     parsing from a string.
        line_number: 1-based line number of the error,
                     or None if not line-specific.
    """

    def __init__(
        self,
        message: str,
        source_path: Optional[Path] = None,
        line_number: Optional[int] = None,
    ) -> None:
        self.source_path = source_path
        self.line_number = line_number

        # Build a location prefix for the message
        # so the caller always sees context without
        # having to inspect the attributes manually.
        parts: list[str] = []
        if source_path is not None:
            parts.append(str(source_path))
        if line_number is not None:
            parts.append(f"line {line_number}")
        location = (
            f"[{', '.join(parts)}] "
            if parts
            else ""
        )

        super().__init__(f"{location}{message}")


@dataclass
class MDPParameter:
    """
    A single parsed MDP parameter.

    Stores both the normalised key (used for lookup)
    and the original key (preserved for round-trip
    fidelity and error messages).

    Attributes:
        key:         Normalised lowercase hyphenated key,
                     e.g. 'vdw-modifier', 'ref-t'.
                     Used for all dict lookups.
        raw_key:     Original key as written in the MDP
                     file, e.g. 'vdw_modifier', 'Ref-T'.
                     Preserved for error messages and
                     round-trip output.
        raw_value:   Raw value string as written in the
                     MDP file, with inline comments
                     stripped but whitespace otherwise
                     preserved, e.g. '0.5   0.5'.
                     Empty string for parameters with
                     no value (e.g. 'define = ').
        values:      Space-split list of value tokens,
                     e.g. ['0.5', '0.5'] for 'tau-t'.
                     Empty list for empty-value params.
        line_number: 1-based line number in the source
                     file where this parameter appears.
                     Used for error messages.
    """

    key: str
    raw_key: str
    raw_value: str
    values: list[str]
    line_number: int


@dataclass
class ParsedMDP:
    """
    A fully parsed GROMACS MDP file.

    Provides dict-like access to parameters by their
    normalised lowercase hyphenated key. All lookups
    are case-insensitive and handle both hyphen and
    underscore key variants transparently.

    Attributes:
        source_path: Path to the source MDP file, or
                     None if parsed from a string.
        parameters:  Dict mapping normalised key →
                     MDPParameter. Keys are always
                     lowercase hyphenated.
        comments:    List of comment lines (including
                     the leading ';') in the order
                     they appear in the file.

    Key normalisation:
        All keys are normalised to lowercase with
        hyphens before storage and lookup:
          'vdw_modifier'  → 'vdw-modifier'
          'VDW-Modifier'  → 'vdw-modifier'
          'NSTEPS'        → 'nsteps'

    Duplicate key handling:
        Follows GROMACS behaviour — the last
        occurrence of a key wins. Earlier values
        are silently overwritten.

    Multi-value parameters:
        Parameters like 'tau-t = 0.5 0.5' store
        the full raw value in raw_value and the
        individual tokens in values. Use get_values()
        for multi-value parameters. get_float() and
        get_int() return None for multi-value params.

    Empty value parameters:
        'define = ' stores raw_value = '' and
        values = []. get_value() returns '' (empty
        string), not None, for explicitly set empty
        parameters.
    """

    source_path: Optional[Path]
    parameters: dict[str, MDPParameter] = field(
        default_factory=dict
    )
    comments: list[str] = field(
        default_factory=list
    )

    # ─────────────────────────────────────────
    # Parameter access
    # ─────────────────────────────────────────

    def get(
        self, key: str
    ) -> Optional[MDPParameter]:
        """
        Return the MDPParameter for a key, or None
        if the key is not present.

        Args:
            key: Parameter key. Case-insensitive.
                 Hyphens and underscores are
                 interchangeable.

        Returns:
            MDPParameter if found, None otherwise.
        """
        return self.parameters.get(
            _normalise_key(key)
        )

    def get_value(
        self, key: str
    ) -> Optional[str]:
        """
        Return the raw value string for a key.

        For single-value parameters returns the
        value string, e.g. 'md', '1.0', 'PME'.

        For multi-value parameters returns the full
        raw value string, e.g. '0.5   0.5'. Use
        get_values() to get individual tokens.

        For explicitly empty parameters (e.g.
        'define = ') returns '' (empty string).

        Returns None only if the key is not present
        in the MDP file at all.

        Args:
            key: Parameter key. Case-insensitive.

        Returns:
            Raw value string, '' for empty params,
            None if key not present.
        """
        param = self.get(key)
        if param is None:
            return None
        return param.raw_value

    def get_float(
        self, key: str
    ) -> Optional[float]:
        """
        Return the value as a float.

        Returns None if:
          - The key is not present
          - The value is empty
          - The value has multiple tokens
            (use get_values() instead)
          - The value cannot be parsed as float

        Args:
            key: Parameter key. Case-insensitive.

        Returns:
            Float value, or None.
        """
        param = self.get(key)
        if param is None:
            return None
        if len(param.values) != 1:
            return None
        try:
            return float(param.values[0])
        except ValueError:
            return None

    def get_int(
        self, key: str
    ) -> Optional[int]:
        """
        Return the value as an integer.

        Returns None if:
          - The key is not present
          - The value is empty
          - The value has multiple tokens
            (use get_values() instead)
          - The value cannot be parsed as int

        Note: '5000000' parses correctly.
              '0.002' returns None (use get_float).

        Args:
            key: Parameter key. Case-insensitive.

        Returns:
            Integer value, or None.
        """
        param = self.get(key)
        if param is None:
            return None
        if len(param.values) != 1:
            return None
        try:
            return int(param.values[0])
        except ValueError:
            return None

    def get_values(
        self, key: str
    ) -> list[str]:
        """
        Return the list of space-split value tokens.

        For single-value parameters returns a
        one-element list, e.g. ['md'].

        For multi-value parameters returns all
        tokens, e.g. ['0.5', '0.5'] for
        'tau-t = 0.5 0.5'.

        For empty-value parameters returns [].

        For absent parameters returns [].

        Args:
            key: Parameter key. Case-insensitive.

        Returns:
            List of value token strings. Never None.
        """
        param = self.get(key)
        if param is None:
            return []
        return list(param.values)

    def has(self, key: str) -> bool:
        """
        Return True if the key is present in the
        MDP file, regardless of its value.

        Args:
            key: Parameter key. Case-insensitive.

        Returns:
            True if key is present, False otherwise.
        """
        return _normalise_key(key) in self.parameters

    def keys(self) -> list[str]:
        """
        Return all normalised parameter keys in the
        order they were last written (insertion order
        of the parameters dict, which reflects the
        last occurrence of each key in the file).

        Returns:
            List of normalised lowercase hyphenated
            parameter key strings.
        """
        return list(self.parameters.keys())

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self.parameters)

    def __contains__(self, key: object) -> bool:
        """Support 'key in mdp' syntax."""
        if not isinstance(key, str):
            return False
        return self.has(key)

    def __repr__(self) -> str:
        n = len(self.parameters)
        src = (
            str(self.source_path)
            if self.source_path
            else "<string>"
        )
        return (
            f"ParsedMDP(source={src!r}, "
            f"n_parameters={n})"
        )


# ─────────────────────────────────────────────
# Key normalisation utility
# ─────────────────────────────────────────────

def _normalise_key(key: str) -> str:
    """
    Normalise an MDP parameter key to lowercase
    hyphenated form.

    GROMACS treats hyphens and underscores as
    interchangeable in parameter names, and is
    case-insensitive. This function applies the
    same normalisation so that lookups are
    consistent regardless of how the key was
    written in the MDP file or by the caller.

    Examples:
        'vdw_modifier'  → 'vdw-modifier'
        'VDW-Modifier'  → 'vdw-modifier'
        'NSTEPS'        → 'nsteps'
        'ref_t'         → 'ref-t'
        'Ref-T'         → 'ref-t'

    Args:
        key: Raw parameter key string.

    Returns:
        Normalised lowercase hyphenated key string.
    """
    return key.strip().lower().replace("_", "-")