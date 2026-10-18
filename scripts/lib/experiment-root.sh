# Shared experiment-root option for shell entrypoints. Source after repo_root
# is resolved, call with "$@", then use RAPID_EXPERIMENT_ARGS as the remaining args.
rapid_experiment_root_args() {
  local value="${RAPID_EXPERIMENT_ROOT:-suite_runs}"
  RAPID_EXPERIMENT_ARGS=()
  while (($#)); do
    case "$1" in
      --experiment-root)
        if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
          echo "error: --experiment-root requires a directory" >&2
          return 2
        fi
        value="$2"
        shift 2
        ;;
      --experiment-root=*)
        value="${1#*=}"
        [[ -n "$value" ]] || { echo "error: --experiment-root requires a directory" >&2; return 2; }
        shift
        ;;
      --)
        RAPID_EXPERIMENT_ARGS+=("$@")
        break
        ;;
      *) RAPID_EXPERIMENT_ARGS+=("$1"); shift ;;
    esac
  done
  while [[ "$value" == ./* ]]; do value="${value#./}"; done
  case "$value" in
    /*) ;;
    suite_runs|suite_runs/*) value="${repo_root}/${value}" ;;
    *) value="${repo_root}/suite_runs/${value}" ;;
  esac
  export RAPID_EXPERIMENT_ROOT
  RAPID_EXPERIMENT_ROOT="$(realpath -m -- "$value")"
}
