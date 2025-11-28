# ===== Base image with conda/mamba =====
FROM condaforge/mambaforge:23.11.0-0

# Required by you
USER root
WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]

# ---- System deps for MNE 2D/3D viz + common tools ----
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    libgl1 libglx-mesa0 libegl1 libglu1-mesa \
    libxrender1 libsm6 libxext6 libfontconfig1 libglib2.0-0 \
    libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 \
    mesa-utils xvfb xauth libgl1-mesa-dri \
    ffmpeg graphviz \
 && rm -rf /var/lib/apt/lists/*

# ---- Faster, stable solver behavior ----
RUN conda config --system --set channel_priority strict \
 && conda config --system --add channels conda-forge

# ---- Create the 'inf' environment (no Octave/FieldTrip) ----
# NOTE: don't pin mne-icalabel/mne-qt-browser to old versions to avoid solving errors
RUN mamba create -y -n inf \
    python=3.10 \
    mne=1.6.* \
    mne-qt-browser \
    mne-icalabel \
    autoreject=0.4.3 \
    numpy scipy pandas numba scikit-learn \
    matplotlib ipympl seaborn \
    pyvista=0.43.7 pyvistaqt=0.11.1 "vtk>=9.2,<9.3" pyqt \
    jupyterlab ipywidgets \
    nibabel nilearn \
    bids-validator pybids \
    tqdm joblib \
 && mamba clean -a -y

# (Optional) If you will use automatic IC labeling inference later, uncomment:
# RUN mamba run -n inf pip install --no-cache-dir onnxruntime

# ---- Env vars for 3D backend ----
ENV MNE_3D_BACKEND=pyvistaqt \
    PYVISTA_OFF_SCREEN=false \
    PYTHONUNBUFFERED=1

# ---- Sanity check (no 'conda activate' needed) ----
RUN mamba run -n inf python - <<'PY'
import mne, pyvista, pyvistaqt, vtk, matplotlib, sys
print("MNE:", mne.__version__)
print("PyVista:", pyvista.__version__)
print("PyVistaQt:", pyvistaqt.__version__)
print("VTK:", vtk.__version__, "Python:", sys.version)
PY

# ---- Copy project last to maximize cache reuse ----
COPY . /workspace

EXPOSE 8888

# ---- Default command: headless 3D via Xvfb + JupyterLab ----
CMD bash -lc '\
  mamba run -n inf \
  xvfb-run -s "-screen 0 1920x1080x24" \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root \
  --NotebookApp.token="" --NotebookApp.password="" \
'


