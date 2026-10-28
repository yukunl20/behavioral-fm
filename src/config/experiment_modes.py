from dataclasses import dataclass


@dataclass(frozen=True)  # prevent accidental mutation of an experiment specification
class ExperimentSpec:
    corruption: str  # Raw-value corruption: zero fill or donor replacement
    status_mode: str  # Encoder implementation behavior


# CLI labels → encoder/corruption behavior
EXPERIMENTS = {
    "zero": ExperimentSpec(
        corruption="zero", status_mode="ignore",  # No visible status signal.
    ),
    "status": ExperimentSpec(
        corruption="zero", status_mode="status_embed",  # Learned observation-state embedding.
    ),
    "token": ExperimentSpec(
        corruption="zero", status_mode="mask_token",  # Learned mask tokens for hidden fields.
    ),
    "vime": ExperimentSpec(
        corruption="donor", status_mode="ignore",  # Hide donor-corruption labels from inputs.
    ),
}


def experiment_spec(name: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown experiment={name}; choose {sorted(EXPERIMENTS)}"
        ) from error
