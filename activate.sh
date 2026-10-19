# Usage: source activate.sh, after creating a project-local .venv.
_MATH_SMALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_MATH_SMALL_ROOT/.venv/bin/activate"
export HF_HOME="${HF_HOME:-$_MATH_SMALL_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCHINDUCTOR_CACHE_DIR="$_MATH_SMALL_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$_MATH_SMALL_ROOT/.cache/triton"
_MATH_SMALL_PTXAS=$(python -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13/bin/ptxas")')
if [ -x "$_MATH_SMALL_PTXAS" ]; then export TRITON_PTXAS_PATH="$_MATH_SMALL_PTXAS"; fi
unset _MATH_SMALL_ROOT _MATH_SMALL_PTXAS
