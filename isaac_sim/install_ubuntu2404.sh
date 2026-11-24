#!/usr/bin/env bash
# Ubuntu 24.04 x86_64 / NVIDIA 580.178.04 / CUDA Toolkit 12.8 / Isaac Sim 5.1.
# Run as your desktop user. Only OS package installation uses sudo.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
driver_version=580.178.04
isaac_version=5.1.0
miniforge_version=26.7.2-0
cuda_package=cuda-toolkit-12-8
stage=all
vision=cuda
sim_test=scene
install_driver=false
plan=false
install_root="$HOME"
cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/hw2-installer"
log_dir="$script_dir/output/install"
repo_url=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64

usage() {
    cat <<'HELP'
Ubuntu 24.04 从零安装（从项目目录以普通用户执行）
  bash isaac_sim/install_ubuntu2404.sh --plan
  bash isaac_sim/install_ubuntu2404.sh --stage system --install-driver
  # 如驱动刚安装：自行重启，再执行下面的 apps 阶段。
  bash isaac_sim/install_ubuntu2404.sh --stage apps --test yolo

已有 NVIDIA 580.178.04，尚未安装其他依赖：
  bash isaac_sim/install_ubuntu2404.sh --stage all --test yolo

选项：
  --stage all|system|apps|verify  默认 all；verify 不安装软件
  --install-driver              在 system 阶段安装并锁定 580.178.04 open 驱动
  --vision cuda|cpu             默认 cuda（PyTorch 2.7.1 + cu128）
  --test scene|yolo|none         默认 scene；none 仍验证依赖、权重与 CUDA 运算
  --install-root PATH           Isaac/Miniforge 安装根目录，默认用户主目录
  --cache-dir PATH              可断点续传的下载缓存目录
  --plan                        仅打印计划，不写文件、不安装、不调用 sudo
  -h, --help                    显示帮助

需要完整项目的 src/code 模型和权重；不需要 ROS、Gazebo、MoveIt。
system 阶段修改 apt 软件包；apps 安装用户环境并执行验证。
驱动安装后退出码 20 表示需要重启；失败返回非零并保留日志。
不自动重启、不卸载现有驱动、不降级其他驱动版本、不修改 .bashrc。
HELP
}

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
note() { printf '\n[hw2-install] %s\n' "$*"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 缺少参数"; }

parse_args() {
    while (($#)); do
        case "$1" in
            --stage) need_value "$@"; stage="$2"; shift 2 ;;
            --vision) need_value "$@"; vision="$2"; shift 2 ;;
            --test) need_value "$@"; sim_test="$2"; shift 2 ;;
            --install-root) need_value "$@"; install_root="$2"; shift 2 ;;
            --cache-dir) need_value "$@"; cache_dir="$2"; shift 2 ;;
            --install-driver) install_driver=true; shift ;;
            --plan) plan=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "未知选项：$1（使用 --help）" ;;
        esac
    done
    case "$stage" in all|system|apps|verify) ;; *) die "不支持的 stage：$stage" ;; esac
    case "$vision" in cuda|cpu) ;; *) die "不支持的 vision：$vision" ;; esac
    case "$sim_test" in scene|yolo|none) ;; *) die "不支持的 test：$sim_test" ;; esac
    [[ "$install_root" == /* && "$cache_dir" == /* ]] || die "安装和缓存路径必须是绝对路径"
    isaac_dir="$install_root/isaacsim-$isaac_version"
    conda_dir="$install_root/miniforge3"
    env_name=hw2_vision_cu128
    env_file="$script_dir/vision_environment_cuda128.yml"
    device=0
    if [[ "$vision" == cpu ]]; then
        env_name=hw2_vision
        env_file="$script_dir/vision_environment.yml"
        device=cpu
    fi
    vision_prefix="$conda_dir/envs/$env_name"
}

show_plan() {
    cat <<PLAN
系统：Ubuntu 24.04 x86_64；阶段：$stage
驱动：$driver_version；安装驱动开关：$install_driver（其他版本不会被自动替换）
CUDA Toolkit：12.8（仅安装 $cuda_package，不使用会升级驱动的 cuda 元包）
Isaac Sim：$isaac_version -> $isaac_dir
Miniforge：$miniforge_version -> $conda_dir
视觉：$vision -> $vision_prefix
下载缓存：$cache_dir
测试：原草莓权重 + $vision 推理 + 仿真 $sim_test
日志：$log_dir
完成后启动：$log_dir/launch.sh --mode yolo
PLAN
}

check_os() {
    [[ $EUID -ne 0 ]] || die "请用普通用户运行脚本，不要 sudo 整个脚本；内部会按需调用 sudo。"
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] || die "仅支持 Ubuntu 24.04"
    [[ "$(uname -m)" == x86_64 ]] || die "仅支持 x86_64；不适用于 Jetson/ARM"
    [[ -f "$script_dir/../src/code/Yolov8_ros/yolov8_ros/weights/strawberry_best.pt" ]] || die "请复制完整项目，包括 src/code 和原草莓权重"
}

loaded_driver() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u || true
    fi
}

check_driver() {
    local current
    current="$(loaded_driver)"
    [[ "$current" == "$driver_version" ]] || die "当前加载驱动为 '${current:-不可用}'，本脚本要求 $driver_version。新安装驱动请重启；其他版本请先自行迁移，不会自动降级。"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
}

fetch() {
    local url="$1" destination="$2"
    if [[ -s "$destination" ]]; then return; fi
    mkdir -p -- "$(dirname -- "$destination")"
    curl --fail --location --retry 6 --retry-delay 3 --connect-timeout 30 \
        --continue-at - --output "$destination.part" "$url"
    mv -- "$destination.part" "$destination"
}

system_install() {
    local current installed_package needs_reboot=false
    current="$(loaded_driver)"
    if [[ -n "$current" && "$current" != "$driver_version" ]]; then
        die "检测到驱动 $current。请先在目标机器配置 $driver_version；脚本不会替换正在使用的其他版本。"
    fi
    if [[ "$current" != "$driver_version" && "$install_driver" != true ]]; then
        die "尚未加载 $driver_version。全新系统请使用 --stage system --install-driver。"
    fi
    sudo -v
    sudo apt-get update
    sudo apt-get install -y curl ca-certificates unzip git python3 python3-venv \
        build-essential pkg-config pciutils mokutil libgl1 libegl1 libglu1-mesa \
        libvulkan1 vulkan-tools libglib2.0-0t64 libsm6 libice6 libxt6 libxrender1 \
        libxrandr2 libxi6 libxinerama1 libxcursor1 libxcomposite1 libasound2t64 \
        libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0
    fetch "$repo_url/cuda-keyring_1.1-1_all.deb" "$cache_dir/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$cache_dir/cuda-keyring_1.1-1_all.deb"
    sudo apt-get update
    if [[ "$install_driver" == true ]]; then
        # NVIDIA's version-locking package pins the entire matching driver family.
        if [[ -x /usr/bin/nvidia-uninstall ]]; then
            die "检测到 .run 方式安装的驱动。请先按原安装方式迁移至 apt，避免混装。"
        fi
        if mokutil --sb-state 2>/dev/null | grep -q 'SecureBoot enabled'; then
            note 'Secure Boot 已启用：请完成 apt 的 MOK 密码设置，并在重启时选择 Enroll MOK。'
        fi
        sudo apt-get install -y "linux-headers-$(uname -r)" dkms
        installed_package="$(dpkg-query -W -f='${Version}' nvidia-driver-580-open 2>/dev/null || true)"
        sudo apt-get install -y "nvidia-driver-pinning-$driver_version"
        sudo apt-get install -y "nvidia-driver-580-open=$driver_version-1ubuntu1"
        if [[ "$current" != "$driver_version" || "$installed_package" != "$driver_version-1ubuntu1" ]]; then
            needs_reboot=true
        fi
    fi
    sudo apt-get install -y "$cuda_package"
    check_toolkit
    if [[ "$needs_reboot" == true ]]; then
        note '驱动与 CUDA 已安装。保存工作后手动 sudo reboot；重启后执行 --stage apps --test yolo。'
        exit 20
    fi
    check_driver
}

check_toolkit() {
    [[ -x /usr/local/cuda-12.8/bin/nvcc ]] || die "缺少 CUDA Toolkit 12.8，请先运行 system 阶段"
    /usr/local/cuda-12.8/bin/nvcc --version
    /usr/local/cuda-12.8/bin/nvcc --version | grep -q 'release 12.8' || die "nvcc 不是 CUDA 12.8"
}

check_space() {
    local available
    mkdir -p -- "$install_root" "$cache_dir"
    available="$(df -Pk "$install_root" | awk 'NR==2 {print $4}')"
    [[ "$available" -ge $((60*1024*1024)) ]] || die "安装盘需要至少 60 GiB 可用空间（含仿真、CUDA 视觉环境及缓存）"
    available="$(df -Pk "$cache_dir" | awk 'NR==2 {print $4}')"
    [[ "$available" -ge $((10*1024*1024)) ]] || die "下载缓存盘需要至少 10 GiB 可用空间"
}

install_isaac() {
    local archive="$cache_dir/isaac-sim-standalone-$isaac_version-linux-x86_64.zip"
    local staging="$isaac_dir.installing"
    if [[ -e "$isaac_dir" ]]; then
        [[ -x "$isaac_dir/python.sh" && -f "$isaac_dir/VERSION" ]] || die "$isaac_dir 已存在但不完整，请移走后重试。"
        grep -q '^5\.1\.0' "$isaac_dir/VERSION" || die "Isaac 目录的版本与 5.1.0 不符"
        note "复用 $isaac_dir"
        return
    fi
    fetch "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-$isaac_version-linux-x86_64.zip" "$archive"
    note '检查 Isaac ZIP 完整性（CRC），然后解压；这一步可能需要几分钟。'
    unzip -tq "$archive" >/dev/null || die "安装包损坏：$archive；请移走该文件后重新下载。"
    mkdir -p -- "$staging"
    unzip -oq "$archive" -d "$staging"
    [[ -x "$staging/python.sh" && -f "$staging/VERSION" ]] || die "安装包缺少启动文件"
    mv -- "$staging" "$isaac_dir"
    (cd -- "$isaac_dir" && env -u PYTHONPATH -u CONDA_PREFIX ./post_install.sh)
}

install_conda() {
    local installer="$cache_dir/Miniforge3-$miniforge_version-Linux-x86_64.sh"
    local url="https://github.com/conda-forge/miniforge/releases/download/$miniforge_version/Miniforge3-$miniforge_version-Linux-x86_64.sh"
    if [[ ! -x "$conda_dir/bin/conda" ]]; then
        [[ ! -e "$conda_dir" ]] || die "Miniforge 目录不完整：$conda_dir；请移走后重试"
        fetch "$url" "$installer"
        fetch "$url.sha256" "$installer.sha256"
        (cd -- "$cache_dir" && sha256sum -c "$(basename -- "$installer").sha256")
        env -u PYTHONPATH bash "$installer" -b -p "$conda_dir"
    fi
    if [[ -x "$vision_prefix/bin/python" ]]; then
        "$conda_dir/bin/conda" env update --prefix "$vision_prefix" --file "$env_file"
    else
        "$conda_dir/bin/conda" env create --prefix "$vision_prefix" --file "$env_file"
    fi
}

write_launcher() {
    # %q quotes machine-local paths safely, including spaces and shell metacharacters.
    {
        printf '# Generated by install_ubuntu2404.sh\n'
        printf 'export ISAAC_SIM_PATH=%q\n' "$isaac_dir"
        printf 'export HW2_VISION_PYTHON=%q\n' "$vision_prefix/bin/python"
        # shellcheck disable=SC2016 # Expanded when the generated file is sourced.
        printf 'export PATH=%q:"$PATH"\n' '/usr/local/cuda-12.8/bin'
    } > "$log_dir/environment.sh"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'source %q\n' "$log_dir/environment.sh"
        # shellcheck disable=SC2016 # Preserve the launcher's runtime variables/arguments.
        printf 'exec %q --mode yolo --vision-python "$HW2_VISION_PYTHON" --device %q "$@"\n' "$script_dir/run.sh" "$device"
    } > "$log_dir/launch.sh"
    chmod +x "$log_dir/launch.sh"
}

verify_install() {
    check_driver
    check_toolkit
    [[ -x "$isaac_dir/python.sh" && -x "$vision_prefix/bin/python" ]] || die "应用未安装，请执行 --stage apps"
    "$vision_prefix/bin/python" -m pip check
    "$vision_prefix/bin/python" "$script_dir/scripts/check_install.py" \
        --vision "$vision" --output "$log_dir/vision_check.json"
    "$isaac_dir/python.sh" "$script_dir/run.py" --prepare-only
    if [[ "$sim_test" != none ]]; then
        local test_output
        test_output="$log_dir/$(date +%Y%m%d-%H%M%S)-$sim_test"
        mkdir -p -- "$test_output"
        note "运行 $sim_test 仿真验证，结果：$test_output"
        ISAAC_SIM_PATH="$isaac_dir" "$script_dir/run.sh" --headless --exit-on-complete \
            --mode "$sim_test" --vision-python "$vision_prefix/bin/python" --device "$device" \
            --output "$test_output"
        python3 - "$test_output/result.json" "$sim_test" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
expected = 'scene_ready' if sys.argv[2] == 'scene' else 'completed'
assert r['status'] == expected, r
if sys.argv[2] == 'yolo':
    assert set(r['sorted']) == {'strawberry1', 'strawberry2', 'strawberry3'}, r
print('仿真验收通过:', sys.argv[1])
PY
    fi
    write_launcher
    note "安装验证完成。打开图形场景：$log_dir/launch.sh --mode scene"
    note "运行 GPU/CPU 视觉分拣：$log_dir/launch.sh --mode yolo"
}

main() {
    parse_args "$@"
    show_plan
    [[ "$plan" != true ]] || return 0
    check_os
    mkdir -p -- "$log_dir"
    exec > >(tee -a "$log_dir/install-$(date +%Y%m%d-%H%M%S).log") 2>&1
    trap 'printf "安装失败（行 %s），请检查 %s 下的日志。\n" "$LINENO" "$log_dir" >&2' ERR
    # Prevent ROS / activated Conda / user site packages from contaminating Isaac.
    unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD CONDA_PREFIX
    export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
    if [[ "$stage" == system || "$stage" == all ]]; then system_install; fi
    if [[ "$stage" == apps || "$stage" == all ]]; then
        check_driver
        check_toolkit
        check_space
        for program in curl unzip python3; do command -v "$program" >/dev/null || die "缺少 $program，请先执行 system 阶段"; done
        install_isaac
        # Constraints prevent pip from upgrading Isaac 5.1's bundled NumPy to 2.x.
        "$isaac_dir/python.sh" -m pip install -r "$script_dir/requirements.txt" 'numpy<2'
        install_conda
        verify_install
    elif [[ "$stage" == verify ]]; then
        verify_install
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
