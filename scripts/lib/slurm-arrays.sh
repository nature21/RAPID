# Shared Slurm array layout and resource contract.
RAPID_PRIMITIVE_NAMES=(flip push grasp press_move pivot topple)

rapid_strategy_ids() {
  local raw="${1//,/ }" id
  local -a ids=()
  local -A seen=()
  read -r -a ids <<< "$raw"
  ((${#ids[@]})) || { echo "error: empty strategy selection" >&2; return 2; }
  for id in "${ids[@]}"; do
    [[ "$id" =~ ^[1-8]$ && -z "${seen[$id]:-}" ]] || {
      echo "error: strategy IDs must be unique integers 1-8: $id" >&2; return 2;
    }
    seen[$id]=1
  done
  (IFS=,; echo "${ids[*]}")
}

rapid_primitive_ids() {
  local raw="${1//,/ }" name index found
  local -a names=() ids=()
  local -A seen=()
  read -r -a names <<< "$raw"
  ((${#names[@]})) || { echo "error: empty primitive selection" >&2; return 2; }
  for name in "${names[@]}"; do
    found=0
    for index in "${!RAPID_PRIMITIVE_NAMES[@]}"; do
      if [[ "$name" == "${RAPID_PRIMITIVE_NAMES[$index]}" && -z "${seen[$name]:-}" ]]; then
        ids+=("$((index + 1))"); seen[$name]=1; found=1; break
      fi
    done
    ((found)) || { echo "error: unknown or duplicate primitive: $name" >&2; return 2; }
  done
  (IFS=,; echo "${ids[*]}")
}

rapid_slurm_task_args() {
  # --nodelist can require ALL listed nodes on older Slurm versions. Restrict
  # the eligible pool by excluding its complement, and request exactly one node.
  local pool="${1// /,}" expanded all_nodes node
  local -a excluded=()
  local -A allowed=() available=()
  expanded="$(scontrol show hostnames "$pool")" || return
  all_nodes="$(sinfo --Node --noheader --format=%N)" || return
  while read -r node; do
    [[ -n "$node" ]] && available[$node]=1
  done <<< "$all_nodes"
  while read -r node; do
    [[ -n "$node" && -n "${available[$node]:-}" ]] || {
      echo "error: unknown or empty Slurm node: $node" >&2; return 2;
    }
    allowed[$node]=1
  done <<< "$expanded"
  while read -r node; do
    [[ -n "$node" && -z "${allowed[$node]:-}" ]] && excluded+=("$node")
  done <<< "$all_nodes"
  RAPID_SLURM_TASK_ARGS=(--nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:1 --kill-on-invalid-dep=yes)
  if ((${#excluded[@]})); then
    RAPID_SLURM_TASK_ARGS+=("--exclude=$(IFS=,; echo "${excluded[*]}")")
  fi
}
