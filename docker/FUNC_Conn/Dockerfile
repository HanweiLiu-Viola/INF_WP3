# ---- INF_WP3 :: FUNC_Conn ----
FROM inf_wp3/base:2025-10-09   

USER root
WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# Java（供 IDTxl 的 JIDT 估计器）
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV JDK_HOME=${JAVA_HOME}
    
RUN pip list --format=freeze \
    | sed -e '/^bctpy==/d' \
          -e '/^snakemake==/d' \
          -e '/^mne-connectivity==/d' \
          -e '/^JPype1==/d' \
          -e '/^pyvista==/d' \
          -e '/^ipyvtklink==/d' \
          -e '/^ipywidgets==/d' \
    > /tmp/base-constraints.txt

# Python Deps（overlay.txt）
COPY docker/FUNC_Conn/overlay.txt /tmp/overlay.txt
RUN pip install --no-cache-dir -r /tmp/overlay.txt \
      -c /tmp/base-constraints.txt \
      --upgrade-strategy only-if-needed \
 && pip check

# JIDT（供 IDTxl/JIDT 路线使用）
RUN mkdir -p /opt/jidt && \
    curl -fsSL -o /opt/jidt/jidt.zip \
      https://github.com/jlizier/jidt/releases/download/v1.6.1/infodynamics-dist-1.6.1.zip && \
    unzip -q /opt/jidt/jidt.zip -d /opt/jidt && \
    ln -sf /opt/jidt/infodynamics-dist-1.6.1/infodynamics.jar /opt/jidt/infodynamics.jar && \
    rm -f /opt/jidt/jidt.zip

ENV CLASSPATH=/opt/jidt/infodynamics.jar:$CLASSPATH

# 3D Backend
ENV MNE_3D_BACKEND=trame
ENV PYVISTA_JUPYTER_BACKEND=1

