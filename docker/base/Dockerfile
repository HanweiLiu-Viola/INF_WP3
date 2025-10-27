# ---- INF_WP3 :: BASE (Octave + FieldTrip + VTK) ----
FROM ghcr.io/mne-tools/mne-python-plot:latest

USER root
WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# System deps: Octave + 构建工具 + 少量无头渲染运行库（供 VTK 导入）
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      git curl unzip ca-certificates \
      build-essential pkg-config make \
      octave gnuplot \
      libgl1 libglx-mesa0 libxrender1 libxext6 libxt6; \
    # Octave 开发包（不同发行版包名不同：liboctave-dev 或 octave-dev）
    if apt-cache show liboctave-dev >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends liboctave-dev; \
    elif apt-cache show octave-dev >/dev/null 2>&1; then \
      apt-get install -y --no-install-recommends octave-dev; \
    else \
      echo "No octave dev package found (liboctave-dev/octave-dev)"; exit 1; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# FieldTrip（Octave 启动即自动 ft_defaults）
RUN git clone --depth=1 https://github.com/fieldtrip/fieldtrip.git /opt/fieldtrip && \
    make -C /opt/fieldtrip mex OCTAVE=/usr/bin/octave || true && \
    install -d /usr/share/octave/site/m/startup && \
    printf "addpath('/opt/fieldtrip'); ft_defaults;\n" > /usr/share/octave/site/m/startup/octaverc




