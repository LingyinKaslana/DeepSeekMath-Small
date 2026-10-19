# Usage: source ./activate.sh
_MATH_SMALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_MATH_SMALL_ROOT/.venv/bin/activate"
export HF_HOME="$_MATH_SMALL_ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="$_MATH_SMALL_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$_MATH_SMALL_ROOT/.cache/triton"
export TRITON_PTXAS_PATH="$_MATH_SMALL_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
unset _MATH_SMALL_ROOT
