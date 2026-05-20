from __future__ import annotations


class PathPatchingNotImplemented(NotImplementedError):
    pass


def require_path_patching() -> None:
    raise PathPatchingNotImplemented("Path patching is planned for mi v0.2.")
