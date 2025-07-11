# AIME MLC - Machine Learning Container management system

Easily install, run and manage Docker containers for Pytorch and Tensorflow deep learning frameworks.

## Features

* Setup and run a specific version of Pytorch or Tensorflow with one simple command
* Run different versions of machine learning frameworks and required libraries in parallel
* manages required libraries (CUDA, CUDNN, CUBLAS, etc.) in containers, without compromising the host installation
* Clear separation of user code and framework installation, test your code with a different framework version in minutes
* multi session: open and run many shell session on a single container simultaneously
* multi user: separate container space for each user
* multi GPU: allocate GPUs per user, container or session
* Runs with the same performance as a bare metal installation
* Repository of all major deep learning framework versions as containers
* (New): create and configure containers in interactive mode
* (New): GPU architecture switchable by flag or an environment variable

## Installation

AIME machines come pre installed with AIME machine learning container management system for more information see: https://www.aime.info/mlc

Please read on in the [AIME MLC Installation Guide](aime-mlc-installation-guide.md) how to install AIME MLC on your PC, workstation or server.

## Usage

To view detailed information about all available commands, use:
```
mlc -h
```

To get help for a specific command (e.g., create), use:
```
mlc create -h
```


### Create a machine learning container

**mlc create container_name framework version [-w workspace\_dir] [-d data\_dir] [-m models\_dir] [-s|--script] [-arch|--architecture gpu_architecture] [-g|--num_gpus all]**

Create a new machine learning container

Available frameworks:

**Pytorch**, **Tensorflow**

The following architectures are currently available: 

**CUDA_BLACKWELL**: NVIDIA RTX 50x0, RTX PRO 5000/6000 Blackwell

**CUDA_ADA**: NVIDIA RTX 40x0, RTX 5000/6000 Ada, L40, H100, H200

**CUDA_AMPERE**: NVIDIA RTX 30x0, RTX A5000/A6000, A100

**CUDA**: NVIDIA GTX 1080TI, RTX 2080TI, Titan RTX, Quadro 5000/6000, Tesla V100

**ROCM6**: AMD RX 7900, RX 9070, MI210, MI250X, MI300X, MI325X

**ROCM5**: AMD Radeon VII, MI100

Frameworks of later version will be mostly supported by the previous GPU generations if the installed driver on the host system doe support the required CUDA/ROCM version. Frameworks compiled for older GPU architectures are not compatible to newer GPU generations.

To print the current available gpu architectures, frameworks and corresponding versions, use:

```
mlc create --info
```

For CUDA_ADA architecture it will output:

```
Available gpu architectures (currently used):
CUDA, *CUDA_ADA*, CUDA_AMPERE, CUDA_BLACKWELL

Available frameworks and versions:

Pytorch:
2.7.0-aime, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.0, 2.3.1-aime, 2.3.0, 2.2.2, 2.2.0, 2.1.2-aime, 2.1.1-aime, 2.1.0-aime, 2.1.0,
2.0.1-aime, 2.0.1, 2.0.0, 1.14.0a-nvidia, 1.13.1-aime, 1.13.0a-nvidia, 1.12.1-aime

Tensorflow:
2.16.1, 2.15.0, 2.14.0, 2.13.1-aime, 2.13.0, 2.12.0, 2.11.0-nvidia, 2.11.0-aime, 2.10.1-nvidia, 2.10.0-nvidia,
2.9.1-nvidia
```


Example to create a container in script mode using Pytorch 2.4.0 with the name 'my-container' and with mounted user home directory as workspace, /data and /models as data and models directory, use:

```
mlc create my-container Pytorch 2.6.0 -w /home/user_name/workspace -d /data -m /models -s -arch CUDA_AMPERE
```

To provide greater flexibility in selecting a GPU architecture, users can specify the desired architecture for the current container using the -arch cuda_architecture flag (default: host gpu architecture, auto-detected). If a fixed architecture is preferred for an entire session, it can be set by saving the desired GPU architecture in the MLC_ARCH environment variable, for example: export MLC_ARCH=CUDA_AMPERE


### Open a machine learning container

**mlc open container_name [-s|--script]**

To open the created machine learning container "my-container" using the script mode

```
mlc open my-container -s
```

Will output:

```
[my-container] starting container
[my-container] opening shell to container

________                               _______________
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


You are running this container as user with ID 1000 and group 1000,
which should map to the ID and group for your user on the Docker host. Great!

[my-container] admin@aime01:/workspace$
```

The container is run with the access rights of the user. To use privileged rights like for installing packages with 'apt' within the container use 'sudo'. The default is that no password is needed for sudo, to change this behaviour set a password with 'passwd'.

Multiple instances of a container can be opened with mlc open. Each instance runs in its own process.

To exit an opened shell to the container type 'exit' on the command line. The last exited shell will automatically stop the container.


### List available machine learning containers

**mlc list** will list all available containers for the current user


```
mlc list
```

will output for example:

```
Available ml-containers are:

CONTAINER           FRAMEWORK                  STATUS
[torch-vid2vid]     Pytorch-1.2.0              Up 2 days
[tf1.15.0]          Tensorflow-1.15.0          Up 8 minutes
[tf1-nvidia]        Tensorflow-1.14.0_nvidia   Exited (137) 1 week ago
[tf1.13.2]          Tensorflow-1.13.2          Exited (137) 2 weeks ago
[torch1.3]          Pytorch-1.3.0              Exited (137) 3 weeks ago
[tf2-gpt2]          Tensorflow-2.0.0           Exited (137) 7 hours ago
```

In case that you wish to see all containers created by all users:

```
mlc list -au
```


### List the stats of active machine learning containers

**mlc stats** show all current running ml containers and their CPU and memory usage

```
mlc stats

Running ml-containers are:

CONTAINER           CPU %               MEM USAGE / LIMIT
[torch-vid2vid]     4.93%               8.516GiB / 63.36GiB
[tf1.15.0]          7.26%               9.242GiB / 63.36GiB
```

### Start machine learning containers

**mlc start container_name [-s|--script]** to explicitly start a container

'mlc start' is a way to start the container to run installed background processes, like an installed web server, on the container without the need to open an interactive shell to it.

For opening a shell to the container just use 'mlc open', which will automatically start the container if the container is not already running.


### Stop machine learning containers

**mlc stop container_name [-s|--script] [-f|--force]** to explicitly stop a container.

'mlc stop' on a container is comparable to a shutdown of a computer, all activate processes and open shells to the container will be terminated.

To force a stop on a container use:

```
mlc stop my-container -f
```

As by the other commands, the script mode is available using the flag -s:

```
mlc stop my-container -s
```


### Remove/Delete a machine learning container

**mlc remove container_name [-s|--script] [-f|--force]** to remove the container.

Warning: the container will be unrecoverable deleted only data stored in the /workspace directory will be kept. Only use to clean up containers which are not needed any more.

```
mlc remove my-container
```

### Update MLC

**mlc update-sys** to update the container managment system to latest version.

The container system and container repo will be updated to latest version. Run this command to check if new framework versions are available. On most systems privileged access (sudo password) is required to do so.

```
mlc update-sys
```

The force option (-f) is available too.

## Supported ML containers

### Pytorch Containers

| Container Name | GPU Arch. | Build | Pytorch | Ubuntu Version | Python Version | Package Manager | CUDA/ROCM Version | CuDNN/MIOpen Version | NVIDIA/ROCM driver version |
|:----------------:|:--------:|:--------:|:----------:|:----------------:|:----------------:|:-----------------:|:--------------:|:---------------:|:-----------------------:|
| 2.7.1-aime | CUDA_BLACKWELL | AIME | 2.7.1 | 24.04 | 3.12.3 | pip 24.0 | 12.8.93 | 9.7.1.26 | 570.124.06 |
| 2.7.1 | CUDA_BLACKWELL | Official | 2.7.1 | 22.04 | 3.11.13 | conda 25.5.0 | 12.8.61 | 9.7.1.26 | 570.86.10 |
| 2.7.1-aime | CUDA_ADA | AIME | 2.7.1 | 22.04 | 3.10.12 | pip 22.0.2 | 12.6.20 | 9.5.1.17 | 560.28.03 |
| 2.7.1 | CUDA_ADA | Official | 2.7.1 | 22.04 | 3.11.13 | conda 25.5.0 | 12.6.85 | 9.5.1.17 | 560.35.05 |
| 2.7.1-aime | CUDA_AMPERE | AIME | 2.7.1 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 9.1.0.70 | 520.61.05 |
| 2.7.1 | CUDA_AMPERE | Official | 2.7.1 | 22.04 | 3.11.13 | conda 25.5.0 | 11.8.89 | 9.1.0.70 | 520.61.05 |
| 2.7.0-aime | CUDA_BLACKWELL | AIME | 2.7.0 | 24.04 | 3.12.3 | pip 24.0 | 12.8.93 | 9.7.1.26 | 570.124.06 |
| 2.7.0 | CUDA_BLACKWELL | Official | 2.7.0 | 22.04 | 3.11.12 | conda 25.3.1 | 12.8.61 | 9.7.1.26 | 570.86.10 |
| 2.7.0-aime | CUDA_ADA | AIME | 2.7.0 | 22.04 | 3.10.12 | pip 22.0.2 | 12.6.20 | 9.5.1.17 | 560.28.03 |
| 2.7.0 | CUDA_ADA | Official | 2.7.0 | 22.04 | 3.11.12 | conda 25.3.1 | 12.6.85 | 9.5.1.17 | 560.35.05 |
| 2.7.0-aime | CUDA_AMPERE | AIME | 2.7.0 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 9.1.0.70 | 520.61.05 |
| 2.7.0 | CUDA_AMPERE | Official | 2.7.0 | 22.04 | 3.11.12 | conda 25.3.1 | 11.8.89 | 9.1.0.70 | 520.61.05 |
| 2.6.0 | ROCM6 | Official | 2.6.0 | 22.04 | 3.10.12 | pip 22.0.2 | 6.2.2 | 3.2.0 | 6.10.5 |
| 2.6.0 | CUDA_ADA | AIME | 2.6.0 | 22.04 | 3.10.12 | pip 22.0.2 | 12.4.131 | 9.1.0.70 | 550.54.15 |
| 2.6.0 | CUDA_AMPERE | AIME | 2.6.0 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 8.9.6.50 | 520.61.05 |
| 2.5.1 | CUDA_ADA | AIME | 2.5.1 | 22.04 | 3.10.12 | pip 22.0.2 | 12.1.105 | 8.9.0.131 | 530.30.02 |
| 2.5.1 | CUDA_AMPERE | AIME | 2.5.1 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 8.9.6.50 | 520.61.05 |
| 2.5.0 | CUDA_ADA | Official | 2.5.0 | 22.04 | 3.11.10 | conda 24.9.2 | 12.1.105 | 9.1.0 | 530.30.02 |
| 2.5.0 | CUDA_AMPERE | Official | 2.5.0 | 22.04 | 3.11.10 | conda 24.9.2 | 11.8.89 | 9.1.0 | 520.61.05 |
| 2.4.0 | CUDA_ADA | Official | 2.4.0 | 22.04 | 3.11.9 | conda 24.5 | 12.1.105 | 9.1.0 | 530.30.02 |
| 2.4.0 | CUDA_AMPERE | Official | 2.4.0 | 22.04 | 3.11.9 | conda 24.5 | 11.8.89 | 9.1.0 | 520.61.05 |
| 2.3.1-aime | CUDA_ADA | AIME | 2.3.1 | 22.04 | 3.10.12 | pip 22.0.2 | 12.1.105 | 8.9.7.29 | 530.30.02 |
| 2.3.1-aime | CUDA_AMPERE | AIME | 2.3.1 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.87 | 8.9.6.50 | 520.61.05 |
| 2.3.0 | CUDA_ADA | Official | 2.3.0 | 22.04 | 3.10.14 | conda 23.5.2 | 12.1.105 | 8.9.7.29 | 530.30.02 |
| 2.2.2 | CUDA_ADA | AIME | 2.2.2 | 22.04 | 3.10.12 | pip 22.0.2 | 12.1.105 | 8.9.7.29 | 530.30.02 |
| 2.2.0 | CUDA_ADA | Official | 2.2.0 | 22.04 | 3.10.13 | conda 23.9.0 | 12.1.105 | 8.9.0.131 | 530.30.02 |
| 2.1.2-aime | CUDA_ADA | AIME | 2.1.2 | 22.04 | 3.10.12 | pip 22.0.2 | 12.1.105 | 8.9.0.131 | 530.30.02 |
| 2.1.1-aime | CUDA_ADA | AIME | 2.1.1 | 22.04 | 3.10.12 | pip 22.0.2 | 12.1.105 | 8.9.0.131 | 530.30.02 |
| 2.1.0-aime | CUDA_ADA, CUDA_AMPERE | AIME | 2.1.0 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 8.9.0.131 | 520.61.05 |
| 2.1.0 | CUDA_ADA | Official | 2.1.0 | 22.04 | 3.10.13 | conda 23.9.0 | 12.1.105 | 8.9.0.131 | 530.30.02 |
| 2.0.1-aime | CUDA_ADA, CUDA_AMPERE | AIME | 2.0.1 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 8.9.0.131 | 520.61.05 |
| 2.0.1 | CUDA_ADA, CUDA_AMPERE | Official | 2.0.1 | 20.04          | 3.8.10         | pip 20.0.2      | 11.8.89      | 8.9.0.131     | 520.61.05             |
| 2.0.0 | CUDA_ADA, CUDA_AMPERE | Official | 2.0.0 | 20.04          | 3.8.10         | pip 20.0.2      | 11.8.89      | 8.6.0.163     | 520.61.05             |
| 1.14.0a-nvidia | CUDA_ADA | NVIDIA   | 1.14.0a0          | 20.04          | 3.8.10         | pip 21.2.4      | 12.0.146     | 8.7.0.84      | 525.85.11
| 1.13.1-aime    | CUDA_ADA | AIME     | 1.13.1            | 20.04          | 3.8.10         | pip 20.0.2      | 11.8.89      | 8.6.0.163     | 520.61.05
| 1.13.0a-nvidia | CUDA_ADA | NVIDIA   | 1.13.0a0          | 20.04          | 3.8.13         | pip 21.2.4      | 11.8.89      | 8.6.0.163     | 520.61.03
| 1.12.1-aime    | CUDA_ADA | AIME     | 1.12.1            | 20.04          | 3.8.10         | pip 20.0.2      | 11.8.89      | 8.6.0.163     | 520.61.05           

### Tensorflow Containers

| Container Name | GPU Arch. | Build  | Tensorflow Version | Ubuntu Version | Python Version | Package Manager | CUDA Version | CuDNN Version | NVIDIA driver version |
|:----------------:|:--------:|:--------:|:--------------------:|:----------------:|:----------------:|:-----------------:|:--------------:|:---------------:|:-----------------------:|
| 2.16.1 | CUDA_ADA | Official | 2.16.1 | 22.04 | 3.11.0rc1 | pip 24.0 | 12.3.107 | 8.9.6.50 | 545.23.06 |
| 2.15.0 | CUDA_ADA | Official | 2.15.0 | 22.04 | 3.11.0rc1 | pip 23.3.1 | 12.3.103 | 8.9.6.50 | 545.23.06 |
| 2.14.0 | CUDA_ADA, CUDA_AMPERE | Official | 2.14.0 | 22.04 | 3.11.0rc1 | pip 23.2.1 | 11.8.89 | 8.6.0.163 | 525.85.12 |
| 2.13.1-aime | CUDA_ADA, CUDA_AMPERE | AIME | 2.13.1 | 22.04 | 3.10.12 | pip 22.0.2 | 11.8.89 | 8.9.0.131 | 520.61.05 |
| 2.13.0 | CUDA_ADA, CUDA_AMPERE | Official | 2.13.0 | 20.04 | 3.8.10 | pip 23.0.1 | 11.8.89 | 8.6.0.163 | 525.85.12 |
| 2.12.0 | CUDA_ADA, CUDA_AMPERE | Official | 2.12.0 | 20.04 | 3.8.10 | pip 23.0.1 | 11.8.89 | 8.6.0.163 | 525.85.12 |
| 2.11.0-nvidia  | CUDA_ADA | NVIDIA | 2.11.0             | 20.04          | 3.8.10         | pip 22.3.1      | 12.0.146     | 8.7.0.84      | 525.85.12 |
| 2.11.0-aime    | CUDA_ADA | AIME   | 2.11.0             | 20.04          | 3.8.10         | pip 20.0.2      | 11.8.89      | 8.6.0.163     | 520.61.05 |
| 2.10.1-nvidia  | CUDA_ADA | NVIDIA | 2.10.1             | 20.04          | 3.8.10         | pip 22.3.1      | 11.8.89      | 8.7.0.84      | 520.61.05             |
| 2.10.0-nvidia  | CUDA_ADA | NVIDIA | 2.10.0             | 20.04          | 3.8.10         | pip 22.2.2      | 11.8.89      | 8.6.0.163     | 520.61.05             |
| 2.9.1-nvidia   | CUDA_ADA | NVIDIA | 2.9.1              | 20.04          | 3.8.10         | pip 22.2.2      | 11.8.89      | 8.6.0.163     | 520.61.03             |

