FROM ubuntu:22.04 AS step1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ANTs 2.5.4
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        file \
        libtbb2 \
        libgomp1 \
        libatomic1 \
        gdb \
        unzip && \
    mkdir /opt/ants && \
    curl -fsSL https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip -o ants.zip && \
    unzip ants.zip -d /opt && \
    rm ants.zip && \
    file /opt/ants-2.5.4/bin/antsRegistration
ENV ANTSPATH="/opt/ants-2.5.4/bin" 

FROM afni/afni_make_build AS afni

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bc \
        binutils \
        bzip2 \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        dcm2niix \
        evince \
        firefox \
        gedit \
        git \
        gnupg \
        gnome-terminal \
        gnome-tweaks \
        gsl-bin \
        jq \
        lsb-release \
        libcurl4-openssl-dev \
        libgdal-dev \
        libgfortran-11-dev \
        libglu1-mesa-dev \
        libglw1-mesa \
        libjpeg62 \
        libnode-dev \
        libopenblas-dev \
        libssl-dev \
        libudunits2-dev \
        libxml2-dev \
        libxm4 \
        netbase \
        netpbm \
        pipx \
        python-is-python3 \
        python3 \
        python3-flask \
        python3-flask-cors \
        python3-matplotlib \
        python3-nibabel \
        python3-numpy \
        python3-pil \
        python3-pip \
        r-base-dev \
        tcsh \
        unzip \
        vim \
        xfonts-100dpi \
        xfonts-base \
        xterm \
        xvfb \
        gdb \
        zip && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy dependencies from other stages
COPY --from=afni /opt /opt
COPY --from=step1 /opt/ants-2.5.4 /opt/ants-2.5.4

# Install and set up Miniconda
RUN curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-py39_25.1.1-2-Linux-x86_64.sh && \
    bash Miniconda3-py39_25.1.1-2-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-py39_25.1.1-2-Linux-x86_64.sh

ENV PATH=/usr/local/miniconda/bin:$PATH \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONNOUSERSITE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC 

RUN conda update -y conda && \
    conda install -y --channel conda-forge \
        matplotlib \
        nibabel \
        nipype \
        numpy \
        python \
        scipy \
        nilearn \
        libgcc-ng \
        ncurses \
        libstdcxx-ng && \
    conda clean -afy

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libatomic.so.1" \
    MKL_DEBUG_CPU_TYPE=5 \
    ANTSPATH="/opt/ants-2.5.4/bin" \
    PATH="/usr/local/miniconda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/miniconda/lib:$LD_LIBRARY_PATH" \
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 \
    GLIBCXX_FORCE_NEW=1

# Create the FW environment
ENV FLYWHEEL=/flywheel/v0
RUN mkdir -p ${FLYWHEEL}

# Copy application files
COPY ./input/ ${FLYWHEEL}/input/
COPY ./workflows/ ${FLYWHEEL}/workflows/
COPY ./pipeline_rPOP.sh ${FLYWHEEL}/
COPY ./rPOP-master ${FLYWHEEL}/rPOP-master
COPY ./workflows/init_ants_reg.sh ${FLYWHEEL}/workflows/
COPY ./workflows/full_ants_reg.sh ${FLYWHEEL}/workflows/
COPY ./workflows/afni.sh ${FLYWHEEL}/workflows/

# Set permissions
RUN chmod -R 777 ${FLYWHEEL}

# Configure entrypoint
ENTRYPOINT ["/bin/bash", "/flywheel/v0/pipeline_rPOP.sh"]
