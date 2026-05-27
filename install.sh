# install CUDA 11.3.1 toolkit from NVIDIA's Conda channel
conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit -y

# 1. Tell the system your CUDA directory is your current Conda environment
export CUDA_HOME=$CONDA_PREFIX

# 2. Shove the Conda environment's bin folder to the very front of the line
export PATH=$CUDA_HOME/bin:$PATH

# 3. Ensure the C++ compiler links against the correct 11.3 libraries
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH