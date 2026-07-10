"""The ``ahn_cli`` Click group and its subcommands (WP2 stub).

RED stub: the group exists but exposes no subcommands, so the WP2 tests fail at
assertion time. The green implementation registers ``fetch`` and ``prep``.
"""

import click


@click.group()
def cli() -> None:
    """Acquire (``fetch``) and transform (``prep``) Dutch elevation data."""
