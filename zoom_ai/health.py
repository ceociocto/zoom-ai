"""Health check for Docker container."""

import sys


def check() -> bool:
    """Check if the application is healthy."""
    try:
        # Import main modules to check they're working
        from zoom_ai import config, camera, tts, avatar
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
