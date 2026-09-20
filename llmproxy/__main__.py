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
from .config import (
    get_config_path,
    heal_config,
    load_config,
    resolve_env_refs,
    save_config,
)


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
    admin_group = parser.add_mutually_exclusive_group()
    admin_group.add_argument(
        "--admin",
        dest="admin",
        action="store_true",
        default=None,
        help="Enable the web admin UI at /admin (sets LLMPROXY_ADMIN_ENABLED=1, "
             "overriding config).",
    )
    admin_group.add_argument(
        "--no-admin",
        dest="admin",
        action="store_false",
        default=None,
        help="Disable the web admin UI (sets LLMPROXY_ADMIN_ENABLED=0, "
             "overriding config).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llmproxy {__version__}",
    )
    return parser


def _config_int_from(cfg: dict, key: str, default: int) -> int:
    """Read an int from a config block, falling back to *default* on any problem.

    Local to this module on purpose: startup must not import server.py just to
    read one number, and a malformed value should start the proxy on the default
    rather than refuse to start at all.
    """
    try:
        raw = cfg.get(key, default)
        if isinstance(raw, bool) or raw is None:
            return default
        return int(raw)
    except (TypeError, ValueError):
        return default


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



# Libraries that log at DEBUG per HTTP request, per connection, or per retry.
# urllib3 alone emits a line for every upstream call llmproxy makes, and the
# proxy's whole job is making upstream calls.
_THIRD_PARTY_LOGGERS = (
    "urllib3",
    "requests",
    "werkzeug",
    "charset_normalizer",
    "asyncio",
)


def _quiet_third_party_loggers(server_cfg: dict) -> None:
    """Keep ``server.log_level`` about llmproxy's own output.

    ``logging.basicConfig`` sets the ROOT level, so ``log_level: DEBUG`` — set
    to see llmproxy's routing decisions — also turns on urllib3, requests and
    werkzeug debug output for the whole process. On a busy proxy that is the
    majority of the log by volume, it costs real time formatting lines nobody
    asked for, and it buries the lines that were the point.

    A setting named ``server.log_level`` should mean llmproxy's level, so the
    libraries are pinned at WARNING instead. ``server.third_party_log_level``
    is the escape hatch for anyone genuinely debugging a transport problem:
    set it to DEBUG to get the old behaviour, or to any level name to choose
    your own. It cannot make a library more verbose than the root level the
    handler filters at, so pairing it with a quiet server.log_level is a no-op
    rather than a surprise.
    """
    raw = server_cfg.get("third_party_log_level") or "WARNING"
    level = getattr(logging, str(raw).upper(), logging.WARNING)
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)


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
    if args.admin is not None:
        # Propagate the toggle through the environment (like LLMPROXY_CONFIG for
        # --config) rather than mutating the in-memory config, which the admin
        # blueprint never reads — it reloads config from disk on every request.
        os.environ["LLMPROXY_ADMIN_ENABLED"] = "1" if args.admin else "0"

    log_level = server_cfg.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _quiet_third_party_loggers(server_cfg)

    host: str = server_cfg.get("host", "0.0.0.0")
    port: int = int(server_cfg.get("port", 8080))

    config_path = get_config_path()
    log = logging.getLogger("llmproxy")
    log.info("Config: %s", config_path)

    # Report the web admin UI status and warn about insecure exposure. Use the
    # blueprint's own predicate (which honors LLMPROXY_ADMIN_ENABLED) so the log
    # cannot diverge from what the server actually enforces per request.
    from .admin import _admin_enabled
    admin_cfg = config.get("admin", {})
    if _admin_enabled(config):
        # Resolve the token the same way the admin API does (env override first,
        # then a config value that may itself be a ${VAR} reference) so the log
        # and the non-loopback warning reflect the effective auth state.
        token_set = bool(
            os.environ.get("LLMPROXY_ADMIN_TOKEN")
            or resolve_env_refs(admin_cfg.get("token"))
        )
        log.info(
            "Web admin UI enabled at http://%s:%d/admin (auth: %s)",
            host, port, "token" if token_set else "localhost-only",
        )
        is_loopback = host in ("127.0.0.1", "::1", "localhost")
        if not is_loopback and not token_set:
            log.warning(
                "Admin UI is bound to a non-loopback host (%s) without an admin "
                "token; the admin API will refuse non-localhost requests. Set "
                "LLMPROXY_ADMIN_TOKEN (or config['admin']['token']) to allow "
                "remote admin access.",
                host,
            )

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

        # Fire the one-time startup tasks (warm the virtual-model route cache
        # and, if enabled, run the free-models updater) inside each worker after
        # it boots. The route cache is a per-process global, so a master-process
        # call would not propagate to forked workers; post_worker_init runs in the
        # worker. The task spawns its own daemon thread, so it never blocks boot.
        def _post_worker_init(worker):  # noqa: ANN001 — gunicorn hook signature
            from .server import _run_startup_tasks_once
            # Drop any config cache state inherited from the pre-fork master so
            # this worker reads providers fresh from disk rather than serving a
            # snapshot the master happened to cache before forking.
            load_config(force_reload=True)
            _run_startup_tasks_once()

        # One worker by default, deliberately. Every piece of state the routing
        # layer depends on — the saturation registry, per-model health scores,
        # free-tier request and token counters — lives in process memory and is
        # not shared between workers. With two of them a 429 that cools a
        # candidate in one worker is invisible to the other, which hits the same
        # exhausted endpoint on the very next request, and free_limits quotas are
        # counted twice over so the proxy believes it has roughly double the
        # headroom it really has. gthread's four threads still provide
        # concurrency. Operators who front several machines, or who genuinely
        # want the CPU parallelism and accept the accounting drift, can raise it.
        workers = max(1, _config_int_from(server_cfg, "workers", 1))
        options = {
            "bind": f"{host}:{port}",
            "workers": workers,
            "worker_class": "gthread",
            "threads": 4,
            "timeout": max(server_cfg.get("stream_timeout", 300), 120),
            "loglevel": log_level.lower(),
            "accesslog": "-",
            "worker_tmp_dir": _gunicorn_worker_tmp_dir(),
            "post_worker_init": _post_worker_init,
        }
        logging.getLogger("llmproxy").info(
            "Starting with gunicorn — %s:%d (%d worker(s) x 4 threads)",
            host, port, workers,
        )
        if workers > 1:
            logging.getLogger("llmproxy").warning(
                "server.workers=%d: quota, health and saturation state is per-process "
                "and is NOT shared between workers, so free-tier limits are counted "
                "per worker and cooldowns do not propagate.",
                workers,
            )
        _StandaloneApp(app, options).run()

    except ImportError:
        logging.getLogger("llmproxy").info(
            "gunicorn not found; using Flask development server — %s:%d", host, port
        )
        # Eagerly warm the virtual-model route cache (and run the free-models
        # updater if enabled) before serving, so virtual models are populated at
        # startup rather than on the first /v1/models request.
        from .server import _run_startup_tasks_once
        _run_startup_tasks_once()
        app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
