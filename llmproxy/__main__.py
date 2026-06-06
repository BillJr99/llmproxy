"""
__main__.py — CLI entry point for llmproxy.

Usage
-----
Run as a module:
    python -m llmproxy [options]

Or via run.py (no install needed):
    python run.py [options]

Or, after pip install:
    llmproxy [options]

Options
-------
  (no flags)       Start the proxy server using the saved configuration.
  --setup          Launch the interactive setup wizard.
  --config PATH    Override the default config file location.
  --host HOST      Override the bind host (default: from config or 0.0.0.0).
  --port PORT      Override the bind port (default: from config or 8080).
  --log-level LVL  Override the log level (DEBUG|INFO|WARNING|ERROR).
  --version        Print version and exit.
  --list-providers Print configured providers and exit.
"""

import argparse
import logging
import os
import tempfile

from . import __version__
from .config import get_config_path, heal_config, load_config, save_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmproxy",
        description=(
            "OpenAI-compatible multi-provider LLM proxy.\n\n"
            "Model IDs follow the convention:  <provider>/<upstream_model_id>\n"
            "Example: openrouter/anthropic/claude-3.5-sonnet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        default=False,
        help="Launch the interactive configuration wizard.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "Path to the config file. "
            "Defaults to ~/.config/llmproxy/config.json or $LLMPROXY_CONFIG."
        ),
    )
    parser.add_argument(
        "--host",
        metavar="HOST",
        default=None,
        help="Bind host override (e.g. 127.0.0.1). Overrides the config file value.",
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=None,
        help="Bind port override. Overrides the config file value.",
    )
    parser.add_argument(
        "--log-level",
        metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log level override (DEBUG|INFO|WARNING|ERROR).",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        default=False,
        help="Print configured provider names and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llmproxy {__version__}",
    )
    return parser


def _gunicorn_worker_tmp_dir() -> str | None:
    """Pick a writable directory for gunicorn's per-worker heartbeat files.

    Gunicorn touches a small temp file per worker (default: the system temp
    dir). In containers run read-only or with an arbitrary ``--user``, Python's
    tempfile resolution can fall through to the current working directory
    (``/app`` in our image), which a non-root user cannot write — gunicorn then
    crashes at startup with a PermissionError. Prefer ``/dev/shm`` (memory-
    backed; gunicorn's documented Docker recommendation), then the system temp
    dir. Return None to accept gunicorn's default only when nothing else is
    writable.
    """
    for candidate in ("/dev/shm", tempfile.gettempdir()):
        if candidate and os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return None


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # If --config was given, write the resolved absolute path into the
    # environment variable immediately.  Every subsequent load_config() call
    # anywhere in the process (including inside Flask route handlers) reads
    # LLMPROXY_CONFIG dynamically, so this single assignment propagates the
    # override without threading it through every call site.
    if args.config:
        os.environ["LLMPROXY_CONFIG"] = os.path.abspath(args.config)

    # ------------------------------------------------------------------ setup
    if args.setup:
        from .setup_wizard import run_setup
        run_setup()
        return

    # ------------------------------------------------------- list-providers
    if args.list_providers:
        config = load_config()
        providers = config.get("providers", {})
        if not providers:
            print("No providers configured. Run 'llmproxy --setup' to add one.")
        else:
            config_path = get_config_path()
            print(f"Config: {config_path}\n")
            for name, cfg in providers.items():
                base = cfg.get("base_url", "(none)")
                filt = cfg.get("model_filter")
                filt_str = f"filter={filt}" if filt else "all models"
                print(f"  {name:20s}  {base}  ({filt_str})")
        return

    # ----------------------------------------------------------- run server
    from .server import app

    # Apply any CLI overrides to the config's server section.
    config = load_config()
    server_cfg = config.setdefault("server", {})

    if args.host is not None:
        server_cfg["host"] = args.host
    if args.port is not None:
        server_cfg["port"] = args.port
    if args.log_level is not None:
        server_cfg["log_level"] = args.log_level

    log_level = server_cfg.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    host: str = server_cfg.get("host", "0.0.0.0")
    port: int = int(server_cfg.get("port", 8080))

    config_path = get_config_path()
    log = logging.getLogger("llmproxy")
    log.info("Config: %s", config_path)

    # Backfill template-derived provider fields missing from older configs
    # (e.g. models_url added after the config was first written). Auto-fixes
    # are logged and persisted; fields we can't reconstruct are warned about.
    # Heal a freshly-loaded copy so any --host/--port/--log-level CLI overrides
    # applied to the in-memory server config above are not persisted to disk.
    healed_config, healed, messages = heal_config(load_config(force_reload=True))
    for level, text in messages:
        getattr(log, level)(text)
    if healed:
        if save_config(healed_config):
            log.info("Persisted auto-healed config to %s", get_config_path())
        else:
            log.warning(
                "Failed to persist auto-healed config to %s; the server will "
                "run with the healed values in memory but the on-disk config "
                "remains unhealed and model discovery may break on next start.",
                get_config_path(),
            )

    # Attempt to import gunicorn; if available, use it for production robustness.
    # Fall back to the Flask dev server for simple / local use.
    try:
        from gunicorn.app.base import BaseApplication

        class _StandaloneApp(BaseApplication):
            def __init__(self, application, options=None):
                self.options = options or {}
                self.application = application
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        workers = 2
        options = {
            "bind": f"{host}:{port}",
            "workers": workers,
            "worker_class": "gthread",
            "threads": 4,
            "timeout": max(server_cfg.get("stream_timeout", 300), 120),
            "loglevel": log_level.lower(),
            "accesslog": "-",
            "worker_tmp_dir": _gunicorn_worker_tmp_dir(),
        }
        logging.getLogger("llmproxy").info(
            "Starting with gunicorn — %s:%d (%d workers x 4 threads)",
            host, port, workers,
        )
        _StandaloneApp(app, options).run()

    except ImportError:
        logging.getLogger("llmproxy").info(
            "gunicorn not found; using Flask development server — %s:%d", host, port
        )
        app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
