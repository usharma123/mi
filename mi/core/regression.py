from __future__ import annotations

from pathlib import Path
from glob import glob

from mi.core.schema import ValidationArtifact


EXIT_CODE_BY_VERDICT = {
    "supported": 0,
    "weak": 2,
    "untested": 3,
    "contradicted": 4,
}


def regression_exit_code(validation: ValidationArtifact) -> int:
    if not validation.results:
        return EXIT_CODE_BY_VERDICT["untested"]
    return max(EXIT_CODE_BY_VERDICT[result.verdict] for result in validation.results)


def claim_paths_from_globs(patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate)
            continue
        matches = sorted(Path(match) for match in glob(pattern))
        paths.extend(path for path in matches if path.is_file())
    return paths
