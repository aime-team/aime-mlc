# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 



import sys           # System-specific functions
import os            # OS interactions
import subprocess    # Run external commands
import argparse      # Parse CLI arguments
import json          # Handle JSON data
import pathlib       # File system paths
import csv           # Read/write CSV files
import re            # Regular expressions

from collections import defaultdict

# Set Default values  AIME mlc
mlc_container_version = 4     # Version number of AIME MLC setup (mlc create). In version 4: data and models directories included
mlc_version = "2.1.0"         # Version number of AIME MLC

# Obtain user and group id, user name for different tasks by create, open,...
user_id = os.getuid()
user_name = os.getlogin()
group_id = os.getgid()      

# Coloring the frontend (ANSI escape codes) and i/o 
ERROR = "\033[91m"          # Red
NEUTRAL = "\033[37m"        # White
INFO = "\033[32m"           # Dark Green
INFO_HEADER = "\033[92m"    # Green
REQUEST = "\033[96m"        # Cyan
WARNING = "\033[38;5;208m"  # Orange
INPUT = "\033[38;5;214m"    # Light orange
HINT = "\033[93m"           # Yellow
AIME_LOGO = "\033[38;5;214m"# Light orange

RESET = "\033[0m"


aime_copyright_claim = f"""{AIME_LOGO}
     ▗▄▄▖   ▄  ▗▖  ▗▖ ▄▄▄▖    ▗▖  ▗▖▗▖    ▗▄▄▄
    ▐▌  ▐▌  █  ▐▛▚▞▜▌         ▐▛▚▞▜▌▐▌   ▐▌   
    ▐▛  ▜▌  █  ▐▌  ▐▌ ▀▀▀     ▐▌  ▐▌▐▌   ▐▌   
    ▐▌  ▐▌  █  ▐▌  ▐▌ ▄▄▄▖    ▐▌  ▐▌▐▙▄▄▖▝▚▄▄▄ 
                                         
                version {mlc_version} 
                 MIT License
    Copyright (c) AIME GmbH and affiliates.                               
{RESET}"""

# Customization of the argument parser
class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"\n{ERROR}Please provide one of the following valid commands:{RESET}\ncreate, list, open, remove, start, stats, stop, update-sys\n")
        exit(1)


def get_flags():
    """_summary_

    Returns:
        _type_: _description_
    """
    #parser = CustomArgumentParser argparse.ArgumentParser
    parser = CustomArgumentParser(
        #ToDo: improve the description using a customized class
        description=f'{aime_copyright_claim}{AIME_LOGO}AIME Machine Learning Container management system.\nEasily install, run and manage Docker containers\nfor Pytorch and Tensorflow deep learning frameworks.{RESET}',
        usage = f"\nmlc [-h] [-v] <command> [-h]",
        formatter_class = argparse.RawTextHelpFormatter  
    )
    
    parser.add_argument(
        '-v', '--version', 
        action = 'version',
        version = f'{INPUT}AIME MLC version: {mlc_version}{RESET}'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=False, help='Sub-command to execute.')

    # Parser for the "create" command
    parser_create = subparsers.add_parser(
        'create',
        description= "Create a new container.",
        help='Create a new container.',
        usage = f"\n{INPUT}mlc create <container_name> <framework_name> <framework_version> "
                f"\n    -w <workspace_directory> -d <data_directory> -m <models_directory>"
                f"\n    -s -arch <gpu_architecture> -ng <number of gpus> {RESET}", 
        formatter_class = argparse.RawTextHelpFormatter
    ) 
    parser_create.add_argument(
        'container_name', 
        nargs='?', 
        type=str, 
        help='Name of the container.'
    )
    parser_create.add_argument(
        'framework', 
        nargs='?', 
        type=str, 
        help='Framework to use.'
    )
    parser_create.add_argument(
        'version', 
        nargs='?', 
        type=str, 
        help='Version of the framework.'
    )
    parser_create.add_argument(
        '-arch', '--architecture', 
        type=str,
        metavar='', 
        help=f"Set the gpu architecture to be used. Default: host gpu architecture (auto-detected)."
             f"\nThere are 2 options to change the default value:"
             f"\n  1.-using the -arch flag."  
             f"\n  2.-adding the environment variable MLC_ARCH, with export MLC_ARCH=gpu_arch."
             f"\nThe flag -arch overrides MLC_ARCH and MLC_ARCH overrides the default value."
    )
    parser_create.add_argument(
        '-d', '--data_dir', 
        type=str,
        metavar='',
        help='Location of the data directory.'
    )
    parser_create.add_argument(
        '-g', '--num_gpus', 
        type=str, 
        default='all',
        metavar='', 
        help='Number of GPUs to be used. Default: all.'
    )
    parser_create.add_argument(
        '-i', '--info', 
        action='store_true',
        help='Show the available AI frameworks and versions (default: interactive mode).'
    )
    parser_create.add_argument(
        '-m', '--models_dir', 
        type=str,
        metavar='', 
        help='Location of the models directory.'
    )
    parser_create.add_argument(
        '-s', '--script', 
        action='store_true',
        help='Enable script mode (default: interactive mode).'
    )
    parser_create.add_argument(
        '-w', '--workspace_dir', 
        default=None, 
        type=str,
        metavar='',
        help='Location of the workspace directory. Default: /home/$USER/workspace.'
    )
  
    # Parser for the "list" command
    parser_list = subparsers.add_parser(
        'list',
        usage= f"\n{INPUT}mlc list [-a|--all]{RESET}",
        description = "List of created containers.",
        help="List of created containers."
    )
    parser_list.add_argument(
        '-a', '--all', 
        action = "store_true", 
        help='Show the full info of the created container/s.'
    )
    parser_list.add_argument(
        '-au', '--all_users', 
        action = "store_true", 
        help='Show the full info of the created container/s of all users.'
    )
    parser_list.add_argument(
        '-arch', '--architecture', 
        action = "store_true", 
        help='Show the gpu architecture info of the created container/s.'
    )
    parser_list.add_argument(
        '-d', '--data', 
        action = "store_true", 
        help='Show the data directories info of the created container/s.'
    )   
    parser_list.add_argument(
        '-m', '--models', 
        action = "store_true", 
        help='Show the models directories info of the created container/s.'
    )      
    parser_list.add_argument(
        '-s', '--size', 
        action = "store_true", 
        help='Show the size info of the created container/s.'
    ) 
    parser_list.add_argument(
        '-w', '--workspace', 
        action = "store_true", 
        help='Show the workspace directories info of the created container/s.'
    )        
       
    # Parser for the "open" command
    parser_open = subparsers.add_parser(
        'open', 
        description= "Open an existing and no running container.",
        help="Open an existing and no running container.", 
        usage = f"\n{INPUT}mlc open container_name -s{RESET}",
        formatter_class = argparse.RawTextHelpFormatter
    )
    parser_open.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help="Name of the container to be opened."
    )
    parser_open.add_argument(
        '-s', '--script', 
        action='store_true', 
        help="Enable script mode (default: interactive mode)."
    )
    
    # Parser for the "remove" command
    parser_remove = subparsers.add_parser(
        'remove',
        usage=f"\n{INPUT}mlc remove <container_name> [-s|--script] [-f|--force]{RESET}",
        description="Remove an existing and no running machine learning container.",
        help="Remove an existing and no running machine learning container."
    )
    parser_remove.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help='Name of the container to be removed.'
    )
    parser_remove.add_argument(
        '-f', '--force', 
        action = "store_true", 
        help='Force to remove the container without asking the user.'
    )
    parser_remove.add_argument(
        '-s', '--script', 
        action='store_true', 
        help="Enable script mode (default: interactive mode)."
    )
    
    # Parser for the "start" command
    parser_start = subparsers.add_parser(
        'start', 
        usage = f"\n{INPUT}mlc start [-s|--script]{RESET}",
        description= "Start an existing and no running container.",
        help="Start an existing and no running container."
    )
    parser_start.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help="Name of the container to be started."
    )
    parser_start.add_argument(
        '-s', '--script', 
        action='store_true', 
        help="Enable script mode (default: interactive mode)."
    )
    
    # Parser for the "stats" command
    parser_stats = subparsers.add_parser(
        'stats',
        usage = f"\n{INPUT}mlc stats{RESET}",
        description= "Show the most important statistics of the running containers.",
        help="Show the most important statistics of the running containers."
    )
    
    # Parser for the "stop" command
    parser_stop = subparsers.add_parser(
        'stop',
        usage= f"\n{INPUT}mlc stop <container_name> [-f|--force] [-s|--script]{RESET}",
        description = "Stop an existing an running container.",
        help="Stop an existing an running container."
    )
    parser_stop.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help="Name of the container to be stopped."
    )
    parser_stop.add_argument(
        '-f', '--force', 
        action = "store_true", 
        help="Force to stop the container without asking the user."
    ) 
    parser_stop.add_argument(
        '-s', '--script', 
        action='store_true', 
        help="Enable script mode (default: interactive mode)."
    )
    # Parser for the "update-sys" command
    parser_update_sys = subparsers.add_parser(
        'update-sys',
        usage= f"\n{INPUT}mlc update-sys [-f|--force]{RESET}",
        description = "Update of the system.",
        help="Update of the system."
    )
    parser_update_sys.add_argument(
        '-f', '--force', 
        action = "store_true", 
        help="Force to update directly without asking user."
    ) 
            
    # Extract subparser names
    subparser_names = subparsers.choices.keys()
    # ToDo: check if needed
    #available_commands = list(subparser_names)

    # Parse arguments
    args = parser.parse_args()
         
    return args


################################################################################################################################################


def are_you_sure(selected_container_name, command, script = False):
    """Ask the user for a confirmation before an action is started.

    Args:
        selected_container_name (str): name of the container.
        command (str): type of the command (create, open, remove,...). The command is used only for printing a message.
        script (bool, optional): script mode is provided. Defaults to False.
    """   
    
    if not script:        
        if command == "create":            
            print(f"\n{WARNING}Verify if the provided setup is correct. The creation of a container may take a little time.{RESET}")
            printed_verb = command + "d"
            prompt = f"\n{INPUT}[{selected_container_name}]{RESET} {REQUEST}will be {printed_verb}. Are you sure(Y/n)?: {RESET}"
            yes_answers = ["y", "yes", ""]
            no_answers = ["n", "no"]
        elif command == "remove":            
            print(f"\n{WARNING}Caution: After your selection, there is no option to recover the container.{RESET}")            
            printed_verb = command + "d"
            prompt = f"\n{INPUT}[{selected_container_name}]{RESET} {REQUEST}will be {printed_verb}. Are you sure(y/N)?: {RESET}"
            yes_answers = ["y", "yes"]
            no_answers = ["n", "no", ""]          
        elif command == "stop":            
            print(f"\n{WARNING}Caution: All running processes of the selected container will be terminated.{RESET}")
            printed_verb = command + "ped"
            prompt = f"\n{INPUT}[{selected_container_name}]{RESET} {REQUEST}will be {printed_verb}. Are you sure(y/N)?: {RESET}"
            yes_answers = ["y", "yes"]
            no_answers = ["n", "no", ""]           
        else:
            exit(1)            
        
        while True:
            are_you_sure_answer = input(prompt).strip().lower()            
            
            if are_you_sure_answer in yes_answers: 
                break            
            elif are_you_sure_answer in no_answers:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}will not be {printed_verb}.\n{RESET}")                
                exit(0)                
            else:                
                print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}") 
                

# ToDo: improve this function using run_docker_command
def check_container_exists(name):
    """Check if a container with the provided tag already exists.

    Args:
        name (str): name of the container tag.

    Returns:
        str: name of the container .
    """
    
    result = subprocess.run(['docker', 'container', 'ps', '-a', '--filter', f'name={name}', '--filter', 'label=aime.mlc', '--format', '{{.Names}}'], capture_output=True, text=True)
    return result.stdout.strip()


def check_container_running(container_tag):
    """Check if the container with the provided container tag is running.

    Args:
        container_tag (str): provided container tag.

    Returns:
        str: name of the container tag associated to the provided container tag.
    """   
    
    docker_command = f'docker container ps --filter=name=^/{container_tag}$ --filter=label=aime.mlc --format "{{{{.Names}}}}"'
    output, _, _ = run_docker_command(docker_command)
    return output
       

def display_gpu_architectures(architectures):
    """Display the available gpu architectures located in the file ml_images.repo.

    Args:
        architectures (list): available gpu architectures.
    """
    
    print(f"\n{INFO}Available gpu architectures:{RESET}")

    for i, architecture in enumerate(architectures, start=1):
        print(f"{i}) {architecture}")
    
    
def display_frameworks(frameworks_dict):
    """Display the available AI frameworks.

    Args:
        frameworks_dict (dict): dictionary whose keys are the available frameworks.

    Returns:
        list: return a list with the available frameworks.
    """ 
     
    print(f"\n{REQUEST}Select a framework:{RESET}")
    framework_list = list(frameworks_dict.keys())

    for i, framework in enumerate(framework_list, start=1):
        print(f"{i}) {framework}")
    return framework_list


def display_versions(framework, versions):
    """Display the corresponding versions to a provided framework.

    Args:
        framework (str): predefined framework.
        versions (tuple): tuple containing (version(str), image(str)).
    """    
    
    print(f"\n{INFO}Available versions for {framework}:{RESET}")
    for i, (version, _) in enumerate(versions, start=1):
        print(f"{i}) {version}")


def extract_from_ml_images(filename, filter_architecture = None):
    """Extract the information from the file corresponding to the supported frameworks, versions, cuda architectures and docker images.

    Args:
        filename (str): name of the file where the framework, version, cuda archicture and docker image name are provided.
        filter_architecture (str, optional): the cuda architecture, for example, "CUDA_ADA". Defaults to None.

    Returns:
        dict, list: provides a dictionary and a list of the available frameworks.
    """
    if filter_architecture is None:
        _, filter_architecture, _ = get_host_gpu_architecture()
        
    frameworks_dict = {}
    headers = ['framework', 'version', 'architecture', 'docker image']
    separator = ";"
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file, fieldnames=headers)
        for row in reader:
            stripped_row = {key: value.strip() if isinstance(value, str) else value for key, value in row.items() }
            framework = stripped_row['framework']
            version = stripped_row['version']            
            architecture = stripped_row['architecture'].strip("[]").split(separator)
            docker_image = stripped_row['docker image']
            if filter_architecture in architecture:

                if framework not in frameworks_dict:
                    frameworks_dict[framework] = []
                    frameworks_dict[framework].append((version, docker_image))
                else:
                    frameworks_dict[framework].append((version, docker_image)) 
    return frameworks_dict


def existing_user_containers(user_name, mlc_command):
    """Provide 2 lists of existing containers and corresponding container tags created previously by the current user.

    Args:
        user_name (str): current user name.
        mlc_command (str): current mlc command.
    Returns:
        list, list: returns 2 lists with existing containers and corresponding container tags.
    """    
 
    # List all containers with the 'aime.mlc' label owned by the current user
    docker_command = f"docker container ps -a --filter=label=aime.mlc.USER={user_name} --format '{{{{.Names}}}}'"
    output, _,_ = run_docker_command(docker_command)
    container_tags = output.splitlines()
    
    # check that at least 1 container has been created previously
    if not container_tags and mlc_command != 'create':
        print(f"\n{ERROR}Create at least one container. If not, mlc {mlc_command} does not work.{RESET}\n")
        exit(0)

    # Extract base names from full container names
    container_names = [re.match(r"^(.*?)(?:\._\.\w+)?$", container).group(1) for container in container_tags]

    return container_names, container_tags


def filter_by_state(state, running_containers, *lists):
    """Filters multiple lists based on the provided state (True/False).

    Args:
        state (bool): the state to filter by (True for running, False for not running).
        running_containers (list): A list of boolean values indicating running status.
        *lists: variable number of lists to filter based on the running_containers.
    Returns:
        list: a list of filtered lists.
    """
    
    return [
        [item for item, running in zip(lst, running_containers) if running == state]
        for lst in lists
    ]


def filter_running_containers(running_containers, *lists):
    """Filters multiple lists (e.g., running_containers and running_container_tags) 
    based on the running_containers_state list, using filter_by_state.

    Args:
        running_containers (list): a list of boolean values indicating running status.
        *lists: variable number of lists to filter based on running_containers.
    Returns:
        tuple: a flattened tuple of no_running and running filtered lists and lengths of the lists.
    """
    
    no_running_results = filter_by_state(False, running_containers, *lists) 
    running_results = filter_by_state(True, running_containers, *lists) 
    
    # Calculate lengths
    no_running_length = len(no_running_results[0])  
    running_length = len(running_results[0])  

    # Return a flattened tuple with no_running followed by running results
    return (*no_running_results, no_running_length, *running_results, running_length)


def format_container_stats(container_stats_dict):
    """Format the container info provided by the stats.

    Args:
        container_stats_dict (dict): dictionay containing the whole stats of a container.

    Returns:
        list: stats line representing the columns of the output.
    """  

    #ToDo: check if labels_string is needed  
    # Extract the 'Labels' field
    #labels_string = container_stats_dict.get('Labels', {})
    # Retrieve the value for 'aime.mlc.USER'
    container_name = container_stats_dict["Name"].split('._.')[0]
    cpu_usage_perc = container_stats_dict["CPUPerc"]
    memory_usage = container_stats_dict["MemUsage"]
    memory_usage_perc = container_stats_dict["MemPerc"]
    processes_active = container_stats_dict["PIDs"]
    stats_line_to_be_printed = [f"[{container_name}]", cpu_usage_perc, memory_usage, memory_usage_perc, processes_active]

    return stats_line_to_be_printed


def get_gpu_architectures(filename):
    """Get current gpu architectures from repo file.

    Args:
        filename (str): name of the file where the framework, version, gpu architecture and docker image name are provided.

    Returns:
        list: provides a list of the available gpu architectures.
    """
    
    # Creating a set to keep only unique items
    unique_architectures = set() 
    
    headers = ['framework', 'version', 'architecture', 'docker image']
    separator = ";"
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file, fieldnames=headers)
        for row in reader:
            stripped_row = {key: value.strip() if isinstance(value, str) else value for key, value in row.items()}
            architecture = stripped_row['architecture'].strip("[]").split(separator)
            unique_architectures.update(architecture)

    available_architectures = list(unique_architectures)
    if not available_architectures:
        print(f"{ERROR}No gpu architectures found.{RESET}")
        exit(1)
    
    return available_architectures   


def get_user_selection(prompt, max_value):
    """The user provides a position in the list corresponding to a desire.

    Args:
        prompt (str): prompt with the request.
        max_value (int): maximal value of the list corresponding to the available options.

    Returns:
        int: positive integer of the selected position.
    """   
        
    while True:
        try:
            selection = int(input(prompt))
            if 1 <= selection <= max_value:
                return selection
            else:
                print(f"{ERROR}Please enter a number between 1 and {max_value}.{RESET}")
        except ValueError:
            print(f"{ERROR}Invalid input. Please enter a valid number.{RESET}")
        

def get_container_image(container_tag):
    """Get the image of the container corresponding to a provided container tag.  

    Args:
        container_tag (str): container tag.
    Returns:
        str: image corresponding to a provided container tag.
    """    
    
   # Get the image associated with the container
    docker_command_get_image = [
        'docker', 
        'container', 
        'ps', 
        '-a', 
        '--filter', f'name={container_tag}', 
        '--filter', 'label=aime.mlc', 
        '--format', '{{.Images}}'
        ]    
    output, _, _ = run_docker_command(docker_command_get_image)
    return output


def get_container_name(container_name, user_name, command, script=False):
    """Get and check whether a container name is provided, and in this case, check that the container name contains valid characters. 

    Args:
        container_name (str): name of the container.
        user_name (str): name of the user.
        command (str): mlc command used.
        script (bool, optional): script mode on=True or off=False. Defaults to False.

    Returns:
        str: returns a validated container name
    """    

    # ToDo: customize i/o usign user_name. 

    if script and not container_name:
        print(f"\n{ERROR}Container name is missing.{RESET}\n")
        exit(1)
    elif not script and container_name:
        while True:
            try:                 
                return validate_container_name(container_name, command, script)            
            except ValueError as e:
                print(e)
                container_name = input(f"\n{REQUEST}Enter a container name (valid characters: a-z, A-Z, 0-9, _,-,#): {RESET}")
    elif not script and not container_name:   
        while True:                           
            container_name = input(f"\n{REQUEST}Enter a container name (valid characters: a-z, A-Z, 0-9, _,-,#): {RESET}")
            try:
                return validate_container_name(container_name, command, script)
            except ValueError as e:
                print(e)
    else:
        return validate_container_name(container_name, command, script) 
    

def get_docker_image(version, images):
    """Get the docker image corresponding to the provided version.

    Args:
        version (str): selected version. Example: 2.4.0.
        images (list of tuples): list containing tuples, which contains version and the docker image location. Example: [('2.4.0', 'aimehub/pytorch-2.4.0-cuda12.1'),...].

    Raises:
        ValueError: if the user provides an unavailable version.

    Returns:
        str: _description_
    """

    for tup in images:
        if tup[0] == version:
            return tup[1]
        
    # Raise an exception if no matching tuple is found              
    raise ValueError("No version available") 


def get_host_gpu_architecture():
    """Detects the GPU architecture (CUDA or ROCm) installed on the host system.

    This function uses the `apt list --installed` command to inspect installed packages
    and determine whether a CUDA or ROCm driver is present. It extracts the version 
    information and maps it to a specific GPU architecture string.

    Returns:
        tuple: A tuple containing:
            - The detected driver type ('CUDA' or 'ROCM')
            - The corresponding architecture string (e.g., 'CUDA_AMPERE', 'ROCM6')
            - The version number (float for CUDA, string for ROCm)

    """    
    
    try:
        # Run the apt command to get installed packages
        cuda_version_command = [
            "apt", 
            "list", 
            "--installed"
        ]
        apt_result = subprocess.run(cuda_version_command, capture_output=True, text=True)

        if apt_result.returncode != 0:
            print(f"\n{ERROR}Host GPU architecture detection: Failed to execute 'apt list --installed'.{RESET}\n")
            exit(1)

        apt_output = apt_result.stdout

        # Group lines containing CUDA or ROCm into buckets. Every new key starts with an empty list
        lines_by_type = defaultdict(list)

        for line in apt_output.split("\n"):
            if "cuda-" in line:
                lines_by_type["cuda"].append(line)
            elif "rocm" in line:
                lines_by_type["rocm"].append(line)
        
        if lines_by_type["cuda"]:
            cuda_lines = "\n".join(lines_by_type["cuda"])
            match = re.search(r'cuda-(\d+\-\d+(\-\d+)?)', cuda_lines)
            
            if match:
                version_str = match.group(1)  # e.g. '12-3-1'
                parts = version_str.split("-")
                host_cuda_version = float(".".join(parts[:2]))  # e.g. 12.3
                
                if host_cuda_version <= 11.8:
                    return "CUDA", "CUDA_AMPERE", host_cuda_version
                elif 12.8 <= host_cuda_version:
                    return "CUDA", "CUDA_BLACKWELL", host_cuda_version
                elif 12.0 <= host_cuda_version:
                    return "CUDA", "CUDA_ADA", host_cuda_version
                else:
                    print(f"\n{ERROR}Unknown CUDA architecture. {RESET}\n")
                    exit(1)
            else:
                print(f"\n{ERROR}CUDA driver version not found. {RESET}\n")
                exit(1)               
                    
        elif lines_by_type["rocm"]:
            rocm_lines = "\n".join(lines_by_type["rocm"])
            match = re.search(r'rocm-dev/[^\s]+\s+(\d+\.\d+\.\d+)', rocm_lines)
            
            if match:
                version_str = match.group(1)  # e.g. '6.3.3'
                host_rocm_version = int(version_str.split(".")[0])
                return "ROCM", f"ROCM{host_rocm_version}", version_str 
            else:
                print(f"\n{ERROR}ROCm driver version not found. {RESET}\n")
                exit(1)  
        else:
            print(f"\n{ERROR}Neither CUDA nor ROCm were found among the installed APT packages. {RESET}\n")
            exit(1)

    
    except Exception as e:
        print(f"\n{ERROR}Failed to detect host GPU architecture.{RESET}\n")
        exit(1)  


def is_container_active(container_name):
    """Check if the container is active by inspecting its processes.

    Args:
        container_name (str): name of the container.

    Returns:
        boolean: True, if the container is active (number of processes is higher as 2 with a successfull exit code of the docker command).
    """    

    docker_command = f'docker top {container_name} -o pid'
    output, _, exit_code = run_docker_command(docker_command)
    process_count = len(output.splitlines())
    if exit_code == 0 and 2 < process_count:
        return "True"
    else:
        return "False"


def print_existing_container_list(container_list):
    """Print an ordered list with the existing containers.

    Args:
        container_list (list): a list containing the available containers.
    """    
    
    for index, container in enumerate(container_list, start=1):
        print(f"{index}) {container}")


def print_info_header(command):
    """Print an info header depending on the used mlc command.

    Args:
        command (str): mlc command used by the user.
    """ 
    
    if command == "create":
        print(
            "\n" \
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Create a new MLC container \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc create <container_name> <framework_name> <framework_version> -w <workspace_directory> -d <data_directory> -m <models_directory> -s -arch <gpu_architecture> -ng <number of gpus> \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc create pt250 Pytorch 2.5.0 -w /home/$USER/workspace -d /data -m /models\n" 
        ) 
                
    if command == "open":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Open an existing machine learning container  \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc open <container_name> [-s|--script]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc open pt231aime -s\n"
        )
        
    if command == "remove":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Remove an existing and no running machine learning container  \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc remove <container_name> [-s|--script] [-f|--force]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc remove pt231aime -s -f \n"
        )   

    if command == "start":        
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Start an existing machine learning container  \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc start <container_name> [-s|--script] \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc start pt231aime -s\n"
        )

    if command == "stop":        
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Stop an existing machine learning container  \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc stop <container_name> [-s|--script] [-f|--force] \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc stop pt231aime -s -f\n"
        )  

    if command == "update-sys":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Update the system  \
            \n\n    {INFO_HEADER}How to use{RESET}: \
            \n    mlc update-sys [-f|--force]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc update-sys -f \n"
        )  


def run_docker_command(docker_command):
    """Run a shell command and return its output usign subprocess.run().

    Args:
        docker_command (str): docker command to be executed.

    Returns:
        str, str, int: standard output and error file handle and returncode.
    """    
 
    result = subprocess.run(
        docker_command, 
        shell=True, 
        text=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def run_docker_command_popen(command):
    """Run a shell command and return its output using subprocess.Popen().

    Args:
        command (str): docker command to be executed.

    Returns:
        str, int: standard error file handle and returncode.
    """    
    process = subprocess.Popen(
        command, 
        shell=False,
        text=True,
        stderr=subprocess.PIPE,
    )
    stderr = process.communicate()  # Communicate handles interactive input/output
    return stderr, process.returncode

def run_docker_pull_image(docker_command):
    """Pull a docker image and return its output usign subprocess.run().

    Args:
        docker_command (str): docker pull command to be executed.

    """ 
    # Run the command and print output in real-time
    result = subprocess.run(
        docker_command, 
        text=True,
        capture_output=False,  
    )

    returncode = result.returncode

    if returncode == 0:
        print(f"\n{INFO}Docker image pulled successfully.{RESET}")
    else:
        print(f"\n{ERROR}Docker pull image failed. Try mlc create again.{RESET}")
        exit(1)


def set_framework(framework_version_docker_sorted):
    """Display the available frameworks and set the framework by interactive selection.

    Args:
        framework_version_docker_sorted (dict): the content is a dict providing the availables frameworks, versions, gpu architecture and docker images.

    Returns:
        str: selected framework.
    """ 
    
    framework_list = display_frameworks(framework_version_docker_sorted)
    framework_num = get_user_selection(f"{REQUEST}Enter the number of the desired framework: {RESET}", len(framework_list))
    
    return framework_list[framework_num - 1]
    
    
def set_version(framework, version_images):
    """Display the available versions of the preselected framework and set a framework version.

    Args:
        framework (str): name of the preselected framework.
        version_images (list): list including tuples with the format (version, image).
    Returns:
        tuple(str, str): returns the version and the corresponding docker image. 
    """    
                    
    display_versions(framework, version_images)
    version_number = get_user_selection(f"{REQUEST}Enter the number of your version: {RESET}", len(version_images))
    return version_images[version_number - 1]


# ToDo: try to combine select_container() with get_user_selection(prompt, max_value).
def select_container(container_list):
    """Prompts the user to select a container from the list.

    Args:
        container_list (list): a list of containers (running, no running)

    Returns:
        str, int: selected container, position in the list
    """    

    while True:
        try:
            selection = int(input(f"\n{REQUEST}Select the number of the container: {RESET}"))
            container_list_length = len(container_list)
            if 1 <= selection <= container_list_length:
                return container_list[selection - 1], selection
            else:
                print(f"\n{ERROR}Invalid selection. Please choose a valid number.{RESET}")
        except ValueError:
            print(f"\n{ERROR}Invalid input. Please enter a number.{RESET}")


def select_container_to_be_ed(containers):
    """Print a list of existing containers and provide a selected container name and its position in a list.
    
    Options: 
    select_container_to_be_opened/removed/start/stopped
    
    Args:
        containers (list): list of container which can be opened/removed/start/stopped

    Returns:
        str, int: selected container, position in the list
    """

    print_existing_container_list(containers)
    selected_container_name, selected_container_position = select_container(containers)    
    return selected_container_name, selected_container_position


def show_container_stats():  
    """Fetch docker container stats.
    """    
  
    command = [
        "docker",
        "stats",
        "--no-stream",
        "--format",'{{json .}}'
    ]
    process = subprocess.Popen(
        command, 
        shell=False,
        text=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
        )

    stdout_data, stderr_data = process.communicate()

    # If no stdout_data is received
    if not stdout_data:
        process.terminate()
        print(f"\n{ERROR}There are no running containers. Start or open a container to show the stats.{RESET}\n")
        exit(0)
    else:        
        # Print the final processed output
        # Define an output format string with specified widths
        format_string = "{:<30}{:<10}{:<25}{:<10}{:<15}"
        print(f"\n{INFO}Current stats of the running containers:{RESET}")
        titles = ["CONTAINER", "CPU %", "MEM USAGE / LIMIT", "MEM %", "PROCESSES (PIDs)"]
                  
        # Split into individual lines and process them as JSON objects
        output_lines = []
        
        # Split by newlines and parse each line as JSON
        json_lines = stdout_data.split('\n')
        containers_stats = [json.loads(line) for line in json_lines if line]

        # Apply formatting to all containers' info
        output_lines = list(map(format_container_stats, containers_stats))
        print(format_string.format(*titles))
        print("\n".join(format_string.format(*info) for info in output_lines)+"\n")
    
    # Exit after processing one time in non-streaming mode
    process.terminate()        


def short_home_path(provided_path):
    """Replace the home directory with "~" if present

    Args:
        provided_path (str): path to be modify

    Returns:
        str: path with "~" if needed
    """  
    home_directory = os.path.expanduser("~")
    
    if provided_path == "-":
        return "-"
    elif provided_path == home_directory:
        return provided_path
    else:        
        shortened_path = provided_path.replace(home_directory, "~", 1)
        return shortened_path


def show_container_info(**kwargs):
    """Print container info

    Args:
        flags (str): flag arguments provided by the user
    """
    
    # Adapt the filter to the selected flags
    filter_aime_mlc_user =  "label=aime.mlc" if kwargs == {} or kwargs["all_users"] else f"label=aime.mlc={user_name}"
    docker_command_ls = [
        "docker", "container", 
        "ls", 
        "-a", 
        "--filter", filter_aime_mlc_user, 
        "--format", '{{json .}}'    
    ]

    # Initialize Popen to run the docker command with JSON output
    process = subprocess.Popen(
        docker_command_ls, 
        shell=False,
        text=True,
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE    # Capture stderr
    )
    
    # Communicate with the process to get output and errors
    stdout_data, stderr_data = process.communicate()   
       
    # Check for any errors
    if process.returncode != 0:
        print(f"{ERROR}Error:{RESET}\n{stderr_data}")
        exit(1)
    else:
        # If no stdout_data is received
        if not stdout_data:
            process.terminate()
            print(f"\n{ERROR}There are no containers. Create the first one using:{RESET}\n{HINT}mlc create container_name{RESET}\n")
            exit(0)    
                
        stdout_data_stripped = stdout_data.strip()
        
        # Titels  extracted from the kwargs
        kwarg_keys_to_be_deleted = ["command", "all", "all_users"]
        kwarg_titles = {key: key.upper() for key in kwargs if key not in kwarg_keys_to_be_deleted}
        kwarg_titles["all_users"] = "USER"
                
        # Default columns to display
        default_titles_to_display = ["CONTAINER", "FRAMEWORK", "STATUS"]
        
        # Columns when flag --all is set up:
        titles_when_all_is_set = [ "USER", "SIZE", "ARCHITECTURE", "WORKSPACE", "DATA", "MODELS"]

        # Add additional columns based on flags
        if kwargs.get("all"):  
            default_titles_to_display.extend(titles_when_all_is_set)
        else:
            default_titles_to_display.extend(kwarg_titles[key] for key in kwargs if key in kwarg_titles and kwargs[key] is True)        
        
        # Titles to be display on the top of the columns
        titles_to_display = default_titles_to_display        

        columns_transcription = {
            "CONTAINER": "aime.mlc.NAME",
            "FRAMEWORK": "aime.mlc.FRAMEWORK", 
            "STATUS": "Status",
            "USER": "aime.mlc.USER",
            "SIZE": "Size",
            "ARCHITECTURE":"aime.mlc.ARCH",
            "WORKSPACE": "aime.mlc.WORK_MOUNT",
            "DATA": "aime.mlc.DATA_MOUNT",
            "MODELS": "aime.mlc.MODELS_MOUNT"
        }
        # Select the values which can be written with '~' 
        reduce_the_path = ["WORKSPACE", "DATA", "MODELS"]  
              
        # Values which can be written with '~' 
        values_to_be_reduced = [columns_transcription[key] for key in reduce_the_path if key in columns_transcription]
        
        # Split by newlines and parse each line as JSON
        json_lines = stdout_data_stripped.split('\n')

        # List of all fields with info of the available containers
        containers_info = [json.loads(line)for line in json_lines if line]
        
        # Flatten the dicts and apply short_home_path for keys in values_to_be_reduced
        flattened_container_infos = [
            {
                **{key: short_home_path(value) if key in values_to_be_reduced 
                   else f"[{value}]" if key == columns_transcription["CONTAINER"] 
                   else value 
                   for key, value in 
                (pair.split('=', 1) for pair in container_dict["Labels"].split(','))},
                **{key: value for key, value in container_dict.items() if key != "Labels"}
            }
            for container_dict in containers_info
        ]
      
        # Assess the column widths for printing with a correct format
        column_widths = {
            key: max(len(key), *(len(container.get(columns_transcription[key], "")) for container in flattened_container_infos)) 
            for key in titles_to_display
        }
        
        # Build the format string dynamically
        format_string = "".join(f"{{:<{column_widths[col]+2}}}" for col in titles_to_display)

        # Print the titles
        print("")
        print(format_string.format(*(titles_to_display)))
        
        # Print the rows
        for container in flattened_container_infos:
            print(format_string.format(*(container.get(columns_transcription[key], '') for key in titles_to_display if key in columns_transcription)))
        print("")
        
        
def show_frameworks_versions(ml_images_content):
    """Print the available frameworks and versions by mlc create

    Args:
        architecture (str): current gpu architecture
        ml_images_content (dict): dict containing as keys the frameworks and as values tuples (version, docker image)
    """    

    frameworks = list(ml_images_content.keys())
    print(f"{INFO}\nAvailable frameworks and versions:{RESET}")
    for framework in frameworks:
        version_images = ml_images_content[framework]
        print(f"\n{HINT}{framework}:{RESET} \n{', '.join([version[0] for version in version_images])}")
    print(" ")
    exit(0)


def validate_container_name(container_name, command, script=False):
    """Validate the container name provided by the user

    Args:
        container_name (str): name of the container provided by the user
        command (str): mlc command
        script (boolean): script mode (on: True, off: False) .Default: false.

    Raises:
        ValueError: The container name should contain at least one character.
        ValueError: The container name contains invalid characters.
        ValueError: The container name already exists.

    Returns:
        str, str: container name and associated container tag
    """

    _ , available_user_container_tags = existing_user_containers(user_name, command) 
    
    pattern = re.compile(r'^[a-zA-Z0-9_\-#]*$')

    if container_name == "":        
        raise ValueError(f"\n{ERROR}The container name should contain at least one character.{RESET}")    
    elif not pattern.match(container_name):
        invalid_chars = [char for char in container_name if not re.match(r'[a-zA-Z0-9_\-#]', char)]
        invalid_chars_str = ''.join(invalid_chars)
        if script:
            print(f"\n{INPUT}[{container_name}]{RESET} contains {ERROR}invalid{RESET} characters: {ERROR}{invalid_chars_str}{RESET}\n")
            exit(1)
        else:
            raise ValueError(f"\n{INPUT}[{container_name}]{RESET} contains {ERROR}invalid{RESET} characters: {ERROR}{invalid_chars_str}{RESET}\n")
    else:
        # Generate a unique container tag
        provided_container_tag = f"{container_name}._.{user_id}"        
        if provided_container_tag in available_user_container_tags:
            if script:
                print(f'\n{INPUT}[{container_name}]{RESET} {ERROR}already exists. Provide a new container name.{RESET}\n')
                exit(1)
            else:
                raise ValueError(f'\n{INPUT}[{container_name}]{RESET} {ERROR}already exists. Provide a new container name.{RESET}')        
        return container_name, provided_container_tag


def build_docker_run_command(    
        architecture, 
        workspace_dir, 
        workspace, 
        container_tag,
        num_gpus, 
        selected_docker_image, 
        validated_container_name,
        user_name, 
        user_id, 
        group_id
    ):
    """Constructs a 'docker run' command based on the host GPU architecture and user setup.

    This function assembles the appropriate `docker run` command with volume mappings,
    user configurations, GPU-specific options (for CUDA or ROCm), and an embedded
    bash script for setting up the container environment.

    Args:
        architecture (str): GPU architecture type (e.g., 'CUDA', 'ROCM').
        workspace_dir (str): Path on the host to be mounted into the container.
        workspace (str): Path inside the container where the workspace will be mounted.
        container_tag (str): Tag to assign to the running container.
        num_gpus (str): Number of GPUs to allocate (used only for CUDA).
        selected_docker_image (str): Base Docker image to use.
        validated_container_name (str): Display name used in the bash prompt inside the container.
        user_name (str): Username to be created inside the container.
        user_id (int): User ID to assign.
        group_id (int): Group ID to assign.

    Returns:
        list: A list representing the full Docker command to run in subprocess or shell.

    Raises:
        ValueError: If the provided architecture is not supported ('CUDA' or 'ROCM').
    """
    
    # Shared base command
    base_docker_cmd = [
        'docker', 'run',
        '-v', f'{workspace_dir}:{workspace}',
        '-w', workspace,
        '--name', container_tag,
        '--tty',
        '--privileged',
        '--network', 'host',
        '--device', '/dev/snd',
        '--ipc', 'host',
        '--ulimit', 'memlock=-1',
        '--ulimit', 'stack=67108864',
        '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
    ]

    cuda_extras = [
        '--gpus', num_gpus,
        '--device', '/dev/video0',
    ]

    rocm_extras = [
        '-u', 'root',
        '--device', '/dev/kfd',
        '--device', '/dev/dri',
        '--cap-add', 'SYS_PTRACE',
        '--security-opt', 'seccomp=unconfined',
        '--shm-size', '8G'
    ]

    # Shared bash command part
    bash_lines = [
        f"echo \"export PS1='[{validated_container_name}] `whoami`@`hostname`:\\${{PWD#*}}$'\" >> ~/.bashrc;",
        "apt-get update -y > /dev/null;",
        "apt-get install sudo git -q -y > /dev/null;",
        f"addgroup --gid {group_id} {user_name} > /dev/null;",
        f"adduser --uid {user_id} --gid {group_id} {user_name} --disabled-password --gecos aime > /dev/null;",
        f"passwd -d {user_name};",
        f"echo \"{user_name} ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/{user_name}_no_password;",
    ]

    # Add ROCm-specific line if needed
    if 'ROCM' in architecture:
        bash_lines.append(f"echo \"export ROCM_PATH=/opt/rocm\" >> ~/.bashrc;")

    bash_lines.extend([
        f"chmod 440 /etc/sudoers.d/${user_name}_no_password;",
        "exit"
    ])      

    bash_command = ' '.join(bash_lines)

    # Assemble full command
    if 'CUDA' in architecture:
        docker_cmd = base_docker_cmd + cuda_extras + [
            f'{selected_docker_image}', 'bash', '-c', bash_command
        ]
    elif 'ROCM' in architecture:
        docker_cmd = base_docker_cmd + rocm_extras + [
            selected_docker_image, 'bash', '-c', bash_command
        ]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return docker_cmd

def build_docker_create_command(
        user_name, 
        user_id, 
        group_id,   
        architecture,
        selected_docker_image,
        selected_framework,
        selected_version,
        mlc_container_version,
        validated_container_name,
        container_label,
        container_tag,
        workspace,
        workspace_dir, 
        data_dir,
        models_dir,
        dir_to_be_added,
        num_gpus,
        volumes                
    ):
    """Constructs a 'docker create' command customized for a machine learning container environment.

    This function prepares a Docker container creation command with appropriate flags, volume
    mounts, GPU-specific configurations (CUDA or ROCm), and user-specific labels. It embeds a bash
    startup sequence that configures the shell prompt, environment paths, and device permissions.

    Args:
        user_name (str): Name of the user to be created inside the container.
        user_id (int): User ID to assign.
        group_id (int): Group ID to assign.
        architecture (str): Target GPU architecture ('CUDA' or 'ROCM').
        selected_docker_image (str): Base Docker image name.
        selected_framework (str): Deep learning framework (e.g., 'tensorflow', 'pytorch').
        selected_version (str): Version of the framework.
        mlc_container_version (str): Version identifier for the container environment.
        validated_container_name (str): Checked name for bash prompt.
        container_label (str): Label to add to the container.
        container_tag (str): Tag to assign to the container.
        workspace (str): Path inside the container where the workspace will be mounted.
        workspace_dir (str): Host path for the workspace.
        data_dir (str): Host path for the dataset directory.
        models_dir (str): Host path for the models directory.
        dir_to_be_added (str): Directory path to add to the container's PATH.
        num_gpus (str): Number of GPUs to assign (used with CUDA).
        volumes (list): Additional volume mount strings to include.

    Returns:
        list: A list representing the full 'docker create' command.

    Raises:
        ValueError: If the provided architecture is not supported.
    """
    
    # Shared base command
    base_docker_cmd = [
        'docker', 'create',
        '-it',
        '-w', workspace,
        '--name', container_tag,
        '--label', f'{container_label}={user_name}',
        '--label', f'{container_label}.NAME={validated_container_name}',
        '--label', f'{container_label}.USER={user_name}',
        '--label', f'{container_label}.ARCH={architecture}',
        '--label', f'{container_label}.MLC_VERSION={mlc_container_version}',
        '--label', f'{container_label}.WORK_MOUNT={workspace_dir}',
        '--label', f'{container_label}.DATA_MOUNT={data_dir}',
        '--label', f'{container_label}.MODELS_MOUNT={models_dir}',
        '--label', f'{container_label}.FRAMEWORK={selected_framework}-{selected_version}',
        '--label', f'{container_label}.GPUS={num_gpus}',
        '--user', f'{user_id}:{group_id}',
        '--tty',
        '--privileged',
        '--interactive',
        '--network', 'host',
        '--device', '/dev/snd',
        '--ipc', 'host',
        '--ulimit', 'memlock=-1',
        '--ulimit', 'stack=67108864',
        '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
        '--group-add', 'video'
    ]   
    
    # Insert the volumes list at the correct position, after '-it'
    base_docker_cmd[3:3] = volumes    
       
    cuda_extras = [
        '--gpus', num_gpus,
        '--device', '/dev/video0',
        '--group-add', 'sudo'
    ]

    rocm_extras = [
        '--device', '/dev/kfd',
        '--device', '/dev/dri',
        '--cap-add', 'SYS_PTRACE',
        '--security-opt', 'seccomp=unconfined',
        '--shm-size', '8G',
        '--group-add', 'sudo'
    ]
    
    # Shared bash command part
    bash_lines = [
        f"echo \"export PS1='[{validated_container_name}] `whoami`@`hostname`:\\${{PWD#*}}$ '\" >> ~/.bashrc;",
        f"echo \"export PATH='{dir_to_be_added}:$PATH'\" >> ~/.bashrc;"
    ]
    
    # Add ROCm-specific line if needed
    if 'ROCM' in architecture:
        bash_lines.append("sudo chown root:video /dev/kfd; sudo chown -R root:video /dev/dri;")
    
    bash_lines.append("bash")
    
    bash_command = ' '.join(bash_lines)

    # Assemble full command
    if 'CUDA' in architecture:
        docker_cmd = base_docker_cmd + cuda_extras + [
            f'{selected_docker_image}:{container_tag}', 'bash', '-c', bash_command
        ]
    elif 'ROCM' in architecture:
        docker_cmd = base_docker_cmd + rocm_extras + [
            f'{selected_docker_image}:{container_tag}', 'bash', '-c', bash_command
        ]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return docker_cmd

###############################################################################################################################################################################################
def main():
    try: 
        # Arguments parsing
        args = get_flags()
           
        if not args.command:
            print(f"\nUse {INPUT}mlc -h{RESET} or {INPUT}mlc --help{RESET} to get more informations about the AIME MLC tool.\n")
            
   
        if args.command == 'create':
            
            # Set the file with frameworks, versions, gpu architectures and images
            repo_name = "ml_images.repo" 
            
            # Read and save content of ml_images.repo
            repo_file = pathlib.Path(__file__).parent / repo_name
            
            #Get the existing gpu architecture    
            architectures = sorted(get_gpu_architectures(repo_file))
            
            # Get the MLC_ARCH environment variable:
            mlc_repo_env_var = os.environ.get('MLC_ARCH')  
            
            cuda_or_rocm, host_gpu_architecture, host_gpu_driver_version = get_host_gpu_architecture()
         
            # Set the gpu architecture based on a flag, an environment variable or the gpu architecture of the host (default value detected automatically)
            architecture = args.architecture or mlc_repo_env_var or host_gpu_architecture
            available_host_gpu_architectures = [architecture for architecture in architectures if cuda_or_rocm in architecture]
            
            # Check gpu architecture
            if args.script:
                if architecture not in available_host_gpu_architectures:
                    print(f"\n{ERROR}Unknown gpu architecture:{RESET} {INPUT}{architecture}{RESET} \n\n{INFO}Available gpu architectures:{RESET}\n{', '.join(available_host_gpu_architectures)}\n")
                    exit(1)
            else:
                while architecture not in available_host_gpu_architectures:
                    print(f"\n{ERROR}Unknown gpu architecture:{RESET} {INPUT}{architecture}{RESET}")
                    display_gpu_architectures(available_host_gpu_architectures)
                    architecture_number = get_user_selection(f"{REQUEST}Enter the number of the desired architecture: {RESET}", len(available_host_gpu_architectures))
                    architecture = available_host_gpu_architectures[architecture_number - 1]            
            
            # Extract framework, version and docker image from the ml_images.repo file
            framework_version_docker = extract_from_ml_images(repo_file, architecture)
            framework_version_docker_sorted = dict(sorted(framework_version_docker.items()))
            
            # Check if the user requests more info about available gpu architecture, framework and version 
            if args.info:
                print(f"\n{INFO}Available gpu architectures ({INPUT}currently used{RESET}{INFO}):{RESET}\n" + ', '.join(f"{INPUT}{arch}{RESET}" if arch == architecture else arch for arch in available_host_gpu_architectures))
                show_frameworks_versions(framework_version_docker_sorted)
                
            # List existing containers/container_tags of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            # Set the variables to know if the workspace, data and models directories should be asked 
            workspace_dir_be_asked = data_dir_be_asked = models_dir_be_asked = False

            # Print an info header only when the positional arguments (container name, frameworl and version) are not provided 
            if args.container_name is None and args.framework is None and args.version is None:
                print_info_header(args.command)
                workspace_dir_be_asked = data_dir_be_asked = models_dir_be_asked = True            
                        
            if args.script:
                # Set the container name and its validation        
                validated_container_name, validated_container_tag = get_container_name(args.container_name, user_name, args.command, args.script)

                # Set the framework:
                if args.framework is None: 
                    print(f"\n{ERROR}Framework is needed.{RESET}\n")
                    exit(1)
                else:
                    if not framework_version_docker_sorted.get(args.framework):
                        print(f"\n{ERROR}Unknown framework:{RESET} {INPUT}{args.framework}{RESET}\n\n{REQUEST}Available AI frameworks:{RESET}\n{', '.join(framework_version_docker_sorted.keys())}\n")
                        exit(1)
                    else:
                        selected_framework = args.framework
                
                # Set the version:
                version_images = framework_version_docker_sorted[selected_framework]
                
                if args.version is None:
                    print(f"\n{ERROR}Version is needed.{RESET}\n")
                    exit(1)
                else:                     
                    available_versions = [version[0] for version in version_images]
                    if args.version in available_versions:
                        selected_docker_image = get_docker_image(args.version, version_images)
                        selected_version = args.version
                    else:
                        print(f"\n{ERROR}Version is not available:{RESET} {INPUT}{args.version}{RESET}\n")
                        exit(1)                      
            else:                
                if args.framework is None:    
                    while args.framework is None:
                        args.framework = set_framework(framework_version_docker_sorted)
                else:
                    while True:                    
                        if not framework_version_docker_sorted.get(args.framework):
                            print(f"\n{ERROR}Unknown framework:{RESET} {INPUT}{args.framework}{RESET}")
                            args.framework = set_framework(framework_version_docker_sorted) 
                        else:
                            break
                selected_framework = args.framework 
                
                # Set the version:
                version_images = framework_version_docker_sorted[selected_framework] 
                
                if args.version is None:
                    #print(f"\n{ERROR}Version is needed.{RESET}")                    
                    while args.version is None:
                        args.version, selected_docker_image = set_version(selected_framework, version_images)                        
                else:                                   
                    available_versions = [version[0] for version in version_images]
                    while True:
                        if args.version in available_versions:
                            selected_docker_image = get_docker_image(args.version, version_images)
                            break
                        else:
                            print(f"\n{ERROR}Version is not available:{RESET} {INPUT}{args.version}{RESET}")
                            args.version, selected_docker_image = set_version(selected_framework, version_images)           
                selected_version = args.version                       
                        
                # Set the container name and its validation        
                validated_container_name, validated_container_tag = get_container_name(args.container_name, user_name, args.command, args.script)
               
            
            # Select Workspace directory:
            default_workspace_dir = os.path.expanduser('~/workspace') 
            workspace_dir_updated = False
          
            # If the -w option is provided, check the user-provided path
            if args.workspace_dir:
                
                provided_workspace_dir = os.path.expanduser(args.workspace_dir)                                  
                while True:
                    # Check if the provided workspace directory exists:
                    if os.path.isdir(provided_workspace_dir):
                        break
                    else:
                        if args.script:
                            workspace_dir = default_workspace_dir
                            print(f"\n{ERROR}Workspace directory does not exist:{RESET} {INPUT}{provided_workspace_dir}{RESET}\n")
                            exit(0)
                        print(f"\n{ERROR}Workspace directory does not exist:{RESET} {INPUT}{provided_workspace_dir}{RESET}")
                        provided_workspace_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the WORKSPACE directory: {RESET}").strip())
                workspace_dir = provided_workspace_dir
                workspace_dir_updated = True
                workspace_dir_be_asked = False
                
            else:
                workspace_dir = default_workspace_dir
            
            if not args.script:         
                if workspace_dir_be_asked:
                    workspace_message = (
                        f"\n{NEUTRAL}The workspace directory would be mounted by default as /workspace in the container.{RESET}"
                        f"\n{NEUTRAL}It is the directory where your project data should be stored to be accessed inside the container.{RESET}"
                        f"\n{HINT}HINT: It can be set to an existing directory with the option '-w /your_workspace'{RESET}"
                    )
                    print(f"{workspace_message}")
                    
                    # Define a variable to control breaking out of both loops
                    break_inner_loop = False
                    
                    while True:
                        
                        keep_workspace_dir = input(f"\n{REQUEST}Current workspace location:{default_workspace_dir}. Keep it (Y/n)?: {RESET}").strip().lower()

                        if keep_workspace_dir in ["y","yes",""]:
                            workspace_dir = default_workspace_dir
                            break
                        elif keep_workspace_dir in ["n","no"]:
                            while True:
                                provided_workspace_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the WORKSPACE directory: {RESET}").strip())  # Expand '~' to full path
                                # Check if the provided workspace directory exist:
                                if os.path.isdir(provided_workspace_dir):
                                    workspace_dir = provided_workspace_dir
                                    break_inner_loop = True
                                    break
                                else:
                                    print(f"\n{ERROR}Workspace directory does not exist:{RESET} {INPUT}{provided_workspace_dir}{RESET}") 
                            if break_inner_loop:
                                break                           
                        else:
                            print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}")

                else:
                    if not workspace_dir_updated:
                        workspace_dir = default_workspace_dir    
               
            # Select Data directory:     
            default_data_dir = "-"
            data_dir_updated = False

            # Check if the provided data directory exists:
            if args.data_dir:
                provided_data_dir = os.path.expanduser(args.data_dir)
                while True:
                    # Check if the provided data directory exists:
                    if os.path.isdir(provided_data_dir):
                        break
                    else:
                        if args.script:
                            print(f"\n{ERROR}Data directory does not exist:{RESET} {INPUT}{provided_data_dir}{RESET}\n")
                            exit(0)
                        print(f"\n{ERROR}Data directory does not exist:{RESET} {INPUT}{provided_data_dir}{RESET}")
                        provided_data_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the DATA directory: {RESET}").strip())
                data_dir = provided_data_dir
                data_dir_updated = True
                data_dir_be_asked = False
            else:
                data_dir = default_data_dir                            
            
            if not args.script:  
                if data_dir_be_asked:
                    data_message = (
                        f"\n{NEUTRAL}The data directory would be mounted as /data in the container.{RESET}"
                        f"\n{NEUTRAL}It is the directory where data sets, for example, mounted from\nnetwork volumes can be accessed inside the container.{RESET}"
                        f"\n{HINT}HINT: It can be set to an existing directory with the option '-d /your_data_directory'{RESET}"
                    )
                    print(f"{data_message}")

                    # Define a variable to control breaking out of both loops
                    break_inner_loop = False
                    while True:                    
                        provide_data_dir = input(f"\n{REQUEST}Do you want to provide a DATA directory (y/N)?: {RESET}").strip().lower()

                        if provide_data_dir in ["n","no", ""]:
                            break
                        elif provide_data_dir in ["y","yes"]:
                            while True:
                                provided_data_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the DATA directory: {RESET}").strip())  # Expand '~' to full path
                                # Check if the provided data directory exists:
                                if os.path.isdir(provided_data_dir):
                                    data_dir = provided_data_dir
                                    break_inner_loop = True
                                    break
                                else:
                                    print(f"\n{ERROR}Provided directory does not exist:{RESET} {INPUT}{provided_data_dir}{RESET}") 
                            if break_inner_loop:
                                break                           
                        else:
                            print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}")
                            
                else:
                    if not data_dir_updated:
                        data_dir = default_data_dir  

            # Select Models directory:     
            default_models_dir = "-"
            models_dir_updated = False

            # Check if the provided models directory exists:
            if args.models_dir:
                provided_models_dir = os.path.expanduser(args.models_dir)
                while True:
                    # Check if the provided models directory exists:
                    if os.path.isdir(provided_models_dir):
                        break
                    else:
                        if args.script:
                            print(f"\n{ERROR}Models directory does not exist:{RESET} {INPUT}{provided_models_dir}{RESET}\n")
                            exit(0)
                        print(f"\n{ERROR}Models directory does not exist:{RESET} {INPUT}{provided_models_dir}{RESET}")
                        provided_models_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the MODELS directory: {RESET}").strip())
                models_dir = provided_models_dir
                models_dir_updated = True
                models_dir_be_asked = False
            else:
                models_dir = default_models_dir                            
              
            if not args.script:                    
                if models_dir_be_asked:
                    models_message = (
                        f"\n{NEUTRAL}The models directory would be mounted as /models in the container.{RESET}"
                        f"\n{NEUTRAL}It is the directory where weight models are download and saved.{RESET}"
                        f"\n{HINT}HINT: It can be set to an existing directory with the option '-d /your_models_directory'{RESET}"
                    )
                    print(f"{models_message}")

                    # Define a variable to control breaking out of both loops
                    break_inner_loop = False
                    while True:
                        
                        provide_models_dir = input(f"\n{REQUEST}Do you want to provide a MODELS directory (y/N)?: {RESET}").strip().lower()

                        if provide_models_dir in ["n","no", ""]:
                            break
                        elif provide_models_dir in ["y","yes"]:
                            while True:
                                provided_models_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the MODELS directory: {RESET}").strip())  # Expand '~' to full path
                                # Check if the provided models directory exists:
                                if os.path.isdir(provided_models_dir):
                                    models_dir = provided_models_dir
                                    break_inner_loop = True
                                    break
                                else:
                                    print(f"\n{ERROR}Provided directory does not exist:{RESET} {INPUT}{provided_models_dir}{RESET}") 
                            if break_inner_loop:
                                break                           
                        else:
                            print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}")
                            
                else:
                    if not models_dir_updated:
                        models_dir = default_models_dir  
          
            # Print a setup summary 
            set_up_summary = (
                f"\n{INFO_HEADER}{'_'*50}{RESET}"   
                f"\n{INFO_HEADER}Summary of the selected setup:{RESET}"
                f"\nGPU architecture: {INPUT}{architecture}{RESET} (host: {host_gpu_architecture}-{host_gpu_driver_version})"
                f"\nContainer name: {INPUT}{validated_container_name}{RESET}"
                f"\nFramework and Version: {INPUT}{selected_framework} {selected_version}{RESET}"
                f"\nWorkspace directory: {INPUT}{workspace_dir}{RESET}"
                f"\nData directory: {INPUT}{data_dir}{RESET}"
                f"\nModels directory: {INPUT}{models_dir}{RESET}"
                f"\n{INFO_HEADER}{'_'*50}{RESET}"                 
            )
            
            print(f"{set_up_summary}")
           
            # Confirm the user's inputs:
            are_you_sure(validated_container_name, args.command, args.script)
            
            # Generate a unique container tag
            container_tag = validated_container_tag
                       
            # Check if a container with the generated tag already exists
            if container_tag == check_container_exists(container_tag):
                print(f"\n{ERROR}Error:{RESET} \n {INPUT}[{validated_container_name}]{RESET} already exists.{RESET}")
                show_container_info()
                exit(0)
            else:
                print(f"\n{NEUTRAL}The container will be created:{RESET} {INPUT}{validated_container_name}{RESET} ")


            # Pull the required image from aime-hub: 
            print(f"\n{NEUTRAL}Acquiring container image ... {RESET}\n")
            docker_command_pull_image = ['docker', 'pull', selected_docker_image]         
            run_docker_pull_image(docker_command_pull_image)     
        
            print(f"\n{NEUTRAL}Setting up container ... {RESET}")
                         
            container_label = "aime.mlc"
            workspace = "/workspace"
            data = "/data"
            models = "/models"
            dir_to_be_added = f'/home/{user_name}/.local/bin'

            # Generating the Docker command for running
            docker_prepare_container = build_docker_run_command(
                architecture,
                workspace_dir,
                workspace,
                container_tag,
                args.num_gpus,
                selected_docker_image,
                validated_container_name,
                user_name,
                user_id,
                group_id
            )
            
            # ToDo: compare subprocess.Popen with subprocess.run  
            result_run_cmd = subprocess.run(docker_prepare_container, capture_output=True, text=True )

            # Commit the container: saves the current state of the container as a new image.
            # ToDo: compare subprocess.Popen with subprocess.run  
            bash_command_commit = [
                'docker', 'commit', container_tag, f'{selected_docker_image}:{container_tag}'
            ]
            # ToDo: capture possible errors and treat them  
            result_commit = subprocess.run(bash_command_commit, capture_output=True, text=True)
            
            # Remove the container: cleans up the initial container to free up ressources.
            # ToDo: compare subprocess.Popen with subprocess.run  
            result_remove = subprocess.run(['docker', 'rm', container_tag], capture_output=True, text=True)
            
            # Add the workspace volume
            volumes = ['-v', f'{workspace_dir}:{workspace}'] 
            
            # Add the data volume mapping if data_dir is set
            if data_dir != default_data_dir:
                volumes +=  ['-v', f'{data_dir}:{data}']
                
            # Add the models volume mapping if models_dir is set
            if models_dir != default_models_dir:
                volumes +=  ['-v', f'{models_dir}:{models}'] 
                
            docker_create_cmd = build_docker_create_command(
                user_name, 
                user_id, 
                group_id,   
                architecture,
                selected_docker_image,
                selected_framework,
                selected_version,
                mlc_container_version,
                validated_container_name,
                container_label,
                container_tag,
                workspace,
                workspace_dir, 
                data_dir,
                models_dir,
                dir_to_be_added,
                args.num_gpus,
                volumes 
            )
            
            # ToDo: compare subprocess.Popen with subprocess.run  
            result_create_cmd = subprocess.run(docker_create_cmd, capture_output= True, text=True)

            print(f"\n{INPUT}[{validated_container_name}]{RESET} ready.{INFO}\n\nOpen the container with:{RESET}\nmlc open {INPUT}{validated_container_name}{RESET}\n")

                     
        if args.command == 'list':
 
            show_container_info(**vars(args))                    

            
        if args.command == 'open':           
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            if args.container_name:                
                if args.container_name not in available_user_containers:                    
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}\n")
                        exit(1)                        
                    else:                                                
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}")
                        print(f"\n{INFO}Available containers of the current user:{RESET}")
                        selected_container_name, selected_container_position = select_container_to_be_ed(available_user_containers)                                          
                else:                    
                    selected_container_name = args.container_name
                    selected_container_position = available_user_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET}{NEUTRAL} exists and will be opened.{RESET}')     
                                  
            else:              
                if args.script:                
                    print(f"{ERROR}Container name is missing.{RESET}")
                    print_info_header(args.command) 
                    exit(1)                
                else:                                      
                    print_info_header(args.command)
                    print(f"\n{INFO}Available containers of the current user:{RESET}")
                    selected_container_name, selected_container_position = select_container_to_be_ed(available_user_containers) 
                              
            # Obtain container_tag from the selected container name
            selected_container_tag = available_user_container_tags[selected_container_position-1]
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}starting container...{RESET}")
                docker_command = f"docker container start {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command)                
            else:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container already running.{RESET}")
                
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}opening shell to container...{RESET}")
                             
            # Set environment variables to pass to the Docker container
            set_env = f"-e DISPLAY={os.environ.get('DISPLAY')}"
            
            # If the NCCL_P2P_LEVEL environment variable is set, include it in the environment settings
            if 'NCCL_P2P_LEVEL' in os.environ:
                set_env += f" -e NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')}"   

            # Open an interactive shell session in the running container as the current user
            docker_command_open_shell=[
                "docker", "exec", 
                "-it",                                  
                set_env,  
                "--user", f"{user_id}:{group_id}", f"{selected_container_tag}",                   
                "bash"  
            ]
            
            #ToDo: capture possible errors and treat them
            error_mesage, exit_code = run_docker_command_popen(docker_command_open_shell)
            
            if exit_code == 1:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}detached from container, container keeps running.{RESET}")                
            elif exit_code == 0:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container shell closed successfully..{RESET}")  
            
            # Check the status of the opened container     
            active_status = is_container_active(selected_container_tag)

            if active_status == "True":                
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{NEUTRAL} container is active, kept running.{RESET}")                
            else:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{NEUTRAL} container is inactive, stopping container ...{RESET}")
                docker_command_stop_container = f"docker container stop {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command_stop_container)
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{NEUTRAL} container stopped.{RESET}\n")  


        if args.command == 'remove':
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            containers_state = [True if container_tag == check_container_running(container_tag) else False for container_tag in available_user_container_tags]
                        
            no_running_containers, no_running_container_tags, no_running_container_number, running_containers, running_container_tags, running_container_number = filter_running_containers(
                containers_state, 
                available_user_containers, 
                available_user_container_tags
            )
            ask_are_you_sure = True
            if args.container_name:                 
                if no_running_container_number == 0:                    
                    print(f"\n{ERROR}All containers are running.\nIf you want to remove a container, stop it before using:{RESET}{HINT}\nmlc stop container_name{RESET}")
                    show_container_info()
                    exit(0)                    
                if args.container_name in running_containers:                    
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be removed.{RESET}\n")
                        exit(1)                    
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be removed.{RESET}")                        
                        print(f"\n{INFO}The following no running containers of the current user can be removed:{RESET} ")
                        selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
                                          
                elif args.container_name in no_running_containers:                    
                    selected_container_name = args.container_name
                    selected_container_position = no_running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {NEUTRAL}is not running and will be removed.{RESET}')                    
                    if not args.script:                         
                        if not args.force:                            
                            print(f"\n{HINT}Hint: Use the flag -f or --force to avoid be asked.{RESET}")
                        else:                            
                            ask_are_you_sure = False
                    else:
                        ask_are_you_sure = False                        
                else:                    
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.\n")
                        exit(1)                    
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.")     
                    
                        # all containers are running
                        if no_running_container_number == 0:
                            show_container_info()
                            exit(0) 
                                                        
                        while True:
                            print(f"\n{INFO}The following no running containers of the current user can be removed:{RESET} ")
                            selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
                            break                                                               
            else:                  
                                
                # Check that at least 1 container is no running
                if no_running_container_number == 0:                    
                    print(f"\n{ERROR}All containers are running.\nIf you want to remove a container, stop it before using:{RESET}{HINT}\nmlc stop container_name{RESET}")
                    show_container_info()
                    exit(0)                
                if args.script:                
                    print(f"\n{ERROR}Container name is missing.{RESET}")
                    print_info_header(args.command)
                    exit(1)                
                else:                     
                    print_info_header(args.command)                    
                    print(f"\n{INFO}The following no running containers of the current user can be removed:{RESET}")
                    selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
            
            # Obtain container_tag from the selected container name
            selected_container_tag = no_running_container_tags[selected_container_position-1]
            
            if ask_are_you_sure:                
                are_you_sure(selected_container_name, args.command, args.script)
            
            docker_command_get_image = [
                'docker', 
                'container', 
                'ps', 
                '-a', 
                '--filter', f'name={selected_container_tag}', 
                '--filter', 'label=aime.mlc', 
                '--format', '{{.Image}}'
            ]
            process = subprocess.Popen(
                    docker_command_get_image, 
                    shell=False,
                    text=True,
                    stdout=subprocess.PIPE, 
            )
            # Communicate handles interactive input/output
            stdout, _ = process.communicate()  
            container_image = stdout.strip()

            # Delete the container
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}deleting container ...{RESET}")
            docker_command_delete_container = f"docker container rm {selected_container_tag}"
            subprocess.Popen(docker_command_delete_container, shell=True, text=True, stdout=subprocess.PIPE).wait()

            # Delete the container's image
            print(f"\n{NEUTRAL}Deleting related image ...{RESET}")
            docker_command_rm_image = f"docker image rm {container_image}"            
            subprocess.Popen(docker_command_rm_image, shell=True).wait()

            print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container removed.{RESET}\n") 
            
            
        if args.command == 'start':
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            containers_state = [True if container_tag == check_container_running(container_tag) else False for container_tag in available_user_container_tags]
                        
            no_running_containers, no_running_container_tags, no_running_container_number, running_containers, running_container_tags, running_container_number = filter_running_containers(
                containers_state, 
                available_user_containers, 
                available_user_container_tags
            )           
            
            if args.container_name: 
                if no_running_container_number == 0:                    
                    print(
                        f"{ERROR}\nAt the moment all containers are running.\nCreate a new one and start it using:{RESET}\n{HINT}mlc start container_name{RESET}"
                    )
                    show_container_info() 
                    exit(0)
                
                if args.container_name in running_containers:                    
                    if args.script:                         
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be started.{RESET}\n")
                        exit(1)                        
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be started.{RESET}")                        
                        print(f"\n{INFO}The following no running containers of the current user can be started:{RESET} ")
                        selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
                                              
                elif args.container_name in no_running_containers:
                    selected_container_name = args.container_name
                    selected_container_position = no_running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {NEUTRAL}is not running and will be started.{RESET}')                    
                else:                    
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}\n")
                        exit(1)                    
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}")                    
                        if no_running_container_number == 0:                            
                            show_container_info()
                            exit(1)
                        
                        while True:                            
                            print(f"\n{INFO}The following no running containers of the current user can be started:{RESET} ")
                            selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
                            break                                          
            else:             
                if no_running_container_number == 0:
                    print(
                        f"{ERROR}\nAt the moment all containers are running.\nCreate a new one and start it using:{RESET}\n{HINT}mlc start container_name{RESET}"
                    )
                    show_container_info() 
                    exit(0)  

                if args.script:                
                    print(f"\n{ERROR}Container name is missing.{RESET}")
                    print_info_header(args.command)
                    exit(1)                    
                else:                    
                    print_info_header(args.command)                    
                    print(f"\n{INFO}The following no running containers of the current user can be started:{RESET}")
                    selected_container_name, selected_container_position = select_container_to_be_ed(no_running_containers) 
            
            # Obtain container_tag from the selected container name
            selected_container_tag = no_running_container_tags[selected_container_position-1]
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}starting container...{RESET}")
                docker_command_start = [
                    "docker",
                    "container",
                    "start",
                    selected_container_tag
                ]
                process = subprocess.Popen(
                    docker_command_start, 
                    shell=False,
                    text=True,
                    stdout=subprocess.PIPE, 
                )
                # Communicate handles interactive input/output
                stdout, _ = process.communicate()  
                out = stdout.strip()
                exit_code = process.returncode
                if out == selected_container_tag:
                    print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container started.{RESET}\n\n{INFO}To open a shell within the container, use:{RESET} \nmlc open {INPUT}{selected_container_name}{RESET}\n")
                else:
                    print(f"\n{INPUT}[{selected_container_name}]{RESET} {ERROR}error starting container.{RESET}")
            else:                
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container already running.{RESET}\n")
                exit(0)


        if args.command == 'stats':
            
            # ToDo: add the stream mode
            show_container_stats()            
            
        if args.command == 'stop':

            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            containers_state = [True if container_tag == check_container_running(container_tag) else False for container_tag in available_user_container_tags]
                        
            no_running_containers, no_running_container_tags, no_running_container_number, running_containers, running_container_tags, running_container_number = filter_running_containers(
                containers_state, 
                available_user_containers, 
                available_user_container_tags
            )
                       
            ask_are_you_sure = True     
                   
            if args.container_name:                
                if running_container_number == 0:                    
                    print(                        
                        f"{ERROR}\nAll containers are stopped. You cannot stop no running containers.{RESET}"
                    )
                    show_container_info() 
                    exit(0)

                if args.container_name in no_running_containers: 
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists but is not running. Not possible to be stopped.{RESET}\n")
                        exit(1)
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}exists but is not running. Not possible to be stopped.{RESET}")
                        print(f"\n{INFO}The following running containers of the current user can be stopped:{RESET} ")
                        selected_container_name, selected_container_position = select_container_to_be_ed(running_containers) 
                elif args.container_name in running_containers:                    
                    selected_container_name = args.container_name
                    selected_container_position = running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {NEUTRAL}is running and will be stopped.{RESET}')                    
                    if not args.script:                        
                        if not args.force:                            
                            print(f"\n{HINT}Hint: Use the flag -f or --force to avoid be asked.{RESET}")
                        else:                            
                            ask_are_you_sure = False                    
                    else:
                        ask_are_you_sure = False                
                else:                    
                    if args.script:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}\n")
                        exit(1)                        
                    else:                        
                        print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.{RESET}")
                                       
                        while True:
                            print(f"\n{INFO}The following running containers of the current user can be stopped:{RESET} ")
                            selected_container_name, selected_container_position = select_container_to_be_ed(running_containers) 
                            break                     
            else:    
                # Check that at least 1 container is running
                if not running_container_tags:                
                    print(f"\n{ERROR}All containers are stopped. Therefore there are no one to be stopped.{RESET}")
                    show_container_info()
                    exit(0)
                
                if args.script:                
                    print(f"\n{ERROR}Container name is missing.{RESET}")
                    print_info_header(args.command)
                    exit(1)                
                else:                
                    print_info_header(args.command)
                    print(f"\n{INFO}Running containers of the current user:{RESET}")
                    selected_container_name, selected_container_position = select_container_to_be_ed(running_containers) 

            # Obtain container_tag from the selected container name
            selected_container_tag = running_container_tags[selected_container_position-1]   
            
            if ask_are_you_sure:                 
                are_you_sure(selected_container_name, args.command, args.script)

            print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}stopping container ...{RESET}")
            
            # Attempt to stop the container and store the result.
            docker_command_stop = f"docker container stop {selected_container_tag}"
            _, _, _ = run_docker_command(docker_command_stop)
            
            # Print a message indicating the container has been stopped.
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {NEUTRAL}container stopped.{RESET}\n")


        if args.command == 'update-sys':

            # Get the directory of the current script
            mlc_path = os.path.dirname(os.path.abspath(__file__))
            
            # Change the current directory to MLC_PATH
            os.chdir(mlc_path)
            
            # Check if the .git directory exists
            if not os.path.isdir(f"{mlc_path}/.git"):
                print(f"{ERROR}Failed: ML container system not installed as updatable git repo.\nAdd the AIME MLC's location to ~/.bashrc or change to the location of the AIME MLC. {RESET}")
                sys.exit(-1)  
                        
            # Determine if sudo is required for git operations
            sudo = "sudo"
            
            if os.access(f"{mlc_path}/.git", os.W_OK):
                sudo = ""  # No sudo if the git directory is writable
                
            # Fix for "unsafe repository" warning in Git adding the mlc-directory to the list of safe directories
            docker_command_git_config = ["git", "config", "--global", "--add", "safe.directory", mlc_path]
            subprocess.run(docker_command_git_config)

            # Get the current branch name
            docker_command_current_branch = ["git", "symbolic-ref", "HEAD"]
            branch = subprocess.check_output(docker_command_current_branch, universal_newlines=True).strip().split("/")[-1]
            print(f"branch update-sys: {branch}")
            
            if not args.force:
                        
                print(f"\n{HINT}Hint: Use the flag -f or --force to avoid be asked.{RESET}")
                
                print(f"\n{NEUTRAL}This will update the ML container system to the latest version.{RESET}")

                # If sudo is required, ask if the user wants to check for updates
                if sudo == "":
                    reply = input(f"\n{REQUEST}Check for available updates (Y/n)?: {RESET}").strip().lower()
                    if reply not in ["y", "yes", "Y", ""]:
                        exit(0)  

                # Fetch the latest updates from remote repo
                docker_command_git_remote = ["git", "remote", "update"]
                sudo and docker_command_git_remote.insert(0, sudo)
                subprocess.run(docker_command_git_remote)
                
                # Get the update log for commits that are new in the remote repo
                docker_command_git_log = ["git", "log", f"HEAD..origin/{branch}", "--pretty=format:%s"]
                sudo and docker_command_git_log.insert(0, sudo)
                update_log = subprocess.check_output(docker_command_git_log, text=True).strip()
                
                if update_log == "":
                    print(f"\n{NEUTRAL}ML container system is up to date.\n{RESET}")
                    exit(0)  
                
                # Print the update log and prompt the user to confirm update
                print(f"\n{INFO}Update(s) available.\n\nChange Log:{RESET}\n{update_log}")
                reply = input(f"\n{REQUEST}Update ML container system (Y/n)?: {RESET}").strip().lower()
                if reply in ["y", "yes", "Y", ""]:
                    args.update_directly = True  
                else:
                    exit(0)  
                
           # If confirmed, proceed with the update
            try:
                print(f"\n{NEUTRAL}Updating ML container system...{RESET}\n")
                
                # Pull the latest changes from the remote repo
                docker_command_git_pull = ["git", "pull", "origin", branch]
                sudo and docker_command_git_pull.insert(0, sudo)          
                subprocess.run(docker_command_git_pull)
                exit(0)
            except Exception as e:
                print(f"\n{ERROR}Error during update: {e}")
                exit(-1)  

    except KeyboardInterrupt:
        print(f"\n{ERROR}\nRunning process cancelled by the user.{RESET}\n")
        exit(1)
   
             
if __name__ == '__main__':
    main()

    
    
