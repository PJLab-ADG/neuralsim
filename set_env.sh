# NOTE: Insert project directory into PYTHONPATH
# Usage: source set_env.sh

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_dir=$(realpath $script_dir)
export PYTHONPATH="${script_dir}":$PYTHONPATH
echo "Added $script_dir to PYTHONPATH"
