"""The ``cli`` interface adapter: the ``ahn_cli`` Click command group.

This package is the composition root that maps Click subcommands onto the
``fetch`` and ``prep`` bounded contexts. It owns argument parsing, validation,
and translation of typed context errors into user-facing Click errors; it holds
no acquisition or transform logic of its own.
"""

from ahn_cli.cli.app import cli

__all__ = ["cli"]
