"""Prepare pinned anatomical subgraphs. Does not launch any simulation."""
from pathlib import Path
import json

import typer

from .connectome import SOURCES, prepare_mushroom_body

app = typer.Typer(no_args_is_help=True)


@app.command()
def sources():
    """Print exact source URLs and expected hashes; no implicit download."""
    print(json.dumps(SOURCES, indent=2))


@app.command()
def prepare(source: Path, output: Path, side: str = "left"):
    """Validate source files, retain KC/APL + actual ALPN providers and all cut edges."""
    graph = prepare_mushroom_body(source, side)
    graph.save(output)
    print(json.dumps(graph.summary(), indent=2))


if __name__ == "__main__":
    app()
