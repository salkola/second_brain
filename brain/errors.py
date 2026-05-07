"""Domain exceptions (small, explicit API per project conventions)."""


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""
