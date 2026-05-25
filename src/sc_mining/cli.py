from pathlib import Path

import typer
from rich import print
from rich.table import Table

from sc_mining.domain.calculator import calculate
from sc_mining.domain.config_loader import load_build, load_heads, load_modules
from sc_mining.domain.models import BeamState, CalculationInput, RockInput


app = typer.Typer(help="Star Citizen mining baseline calculator")


@app.callback()
def main() -> None:
    """
    Star Citizen mining assistant CLI.
    """


@app.command()
def calc(
    build_path: Path = typer.Option(..., "--build", help="Path to build YAML"),
    mass: float = typer.Option(..., "--mass", help="Rock mass"),
    resistance: float = typer.Option(..., "--resistance", help="Rock resistance"),
    instability: float = typer.Option(0.0, "--instability", help="Rock instability"),
    distance: float = typer.Option(100.0, "--distance", help="Distance to rock"),
    beam: list[str] = typer.Option(..., "--beam", help="Beam in slot:power format"),
) -> None:
    heads = load_heads("configs/heads.yaml")
    modules = load_modules("configs/modules.yaml")
    build = load_build(build_path)

    beams: list[BeamState] = []

    for raw_beam in beam:
        try:
            slot, power = raw_beam.split(":")
            beams.append(
                BeamState(
                    slot=slot,
                    power_percent=float(power),
                    active_modules=[],
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid beam format: {raw_beam}. Expected slot:power"
            ) from exc

    calc_input = CalculationInput(
        rock=RockInput(
            mass=mass,
            resistance=resistance,
            instability=instability,
            distance=distance,
        ),
        build=build,
        beams=beams,
    )

    result = calculate(calc_input, heads=heads, modules=modules)

    table = Table(title="SC Mining Baseline Calculation")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Build", build.build_id)
    table.add_row("Ship", build.ship_type)
    table.add_row("Required power", str(result.required_power))
    table.add_row("Effective power", str(result.effective_power))
    table.add_row("Margin", str(result.margin))
    table.add_row("Risk score", str(result.risk_score))
    table.add_row("Verdict", result.verdict)

    print(table)

    if result.notes:
        print("\n[bold]Notes:[/bold]")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":
    app()