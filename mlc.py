# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 



import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
import csv
import re



# Set default values for script variables
mlc_container_version=3                         # Version number of AIME MLC setup
mlc_version = "2.0-beta1"                       # Version number of AIME MLC
# Obtain user and group id, user name for different tasks by create, open,...
user_id = os.getuid()
#user_name = subprocess.getoutput("id -un")
#user_name_pwd = pwd.getpwuid(user_id).pw_name
#ToDo: compare both user_name
user_name = os.getlogin()
group_id = os.getgid()      
#print(f"user_name_pwd: {user_name_pwd}")
#print(f"user_name_login: {user_name}")

# Coloring the frontend (ANSI escape codes)

ERROR = "\033[91m"          # Red
INFO = "\033[37m"           # White
INFO_HEADER = "\033[92m"    # Green
REQUEST = "\033[96m"        # Cyan
WARNING = "\033[38;5;208m"  # Orange
INPUT = "\033[38;5;214m"    # Light orange
HINT = "\033[93m"           # Yellow

RESET = "\033[0m"

AIME_LOGO = "\033[38;5;214m"# Light orange           

MAGENTA = "\033[95m"        # Magenta
BLUE = "\033[94m"           # Blue

aime_copyright_claim = f"""{AIME_LOGO}
     ▗▄▄▖   ▄  ▗▖  ▗▖ ▄▄▄▖    ▗▖  ▗▖▗▖    ▗▄▄▄
    ▐▌  ▐▌  █  ▐▛▚▞▜▌         ▐▛▚▞▜▌▐▌   ▐▌   
    ▐▛  ▜▌  █  ▐▌  ▐▌ ▀▀▀     ▐▌  ▐▌▐▌   ▐▌   
    ▐▌  ▐▌  █  ▐▌  ▐▌ ▄▄▄▖    ▐▌  ▐▌▐▙▄▄▖▝▚▄▄▄ 
                                         
              version {mlc_version} 
                 MIT License
    Copyright (c) AIME GmbH and affiliates.                               
        {RESET}"""

def get_flags():
    """_summary_

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(
        #description=f'{AIME_LOGO}Manage machine learning containers (mlc).{RESET}',
        description=f'{aime_copyright_claim} {AIME_LOGO}AIME Machine Learning Container management system.\nEasily install, run and manage Docker containers\nfor Pytorch and Tensorflow deep learning frameworks.{RESET}',
        usage = "mlc [-h] [-v] <command> [-h]",
        formatter_class = argparse.RawTextHelpFormatter  
    )
    
    parser.add_argument(
        '-v', '--version', 
        action = 'version',
        version = f'{INPUT}AIME MLC version: {mlc_version}{RESET}'
        #help = 'Current version of the AIME MLC'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=False, help='Sub-command to execute')

    # Parser for the "create" command
    parser_create = subparsers.add_parser('create', help='Create a new container')
    parser_create.add_argument(
        'container_name', 
        nargs='?', 
        type=str, 
        help='Name of the container'
    )    
    parser_create.add_argument(
        'framework', 
        nargs='?', 
        type=str, 
        help='The framework to use'
    )
    parser_create.add_argument(
        'version', 
        nargs='?', 
        type=str, 
        help='The version of the framework'
    )
    parser_create.add_argument(
        '-w', '--workspace_dir', 
        default=None, 
        type=str, 
        help='Location of the workspace directory. Default: /home/$USER/workspace'
    )
    parser_create.add_argument(
        '-d', '--data_dir', 
        type=str, 
        help='Location of the data directory.'
    )
    parser_create.add_argument(
        '-ng', '--num_gpus', 
        type=str, 
        default='all',
        help='Number of GPUs to be used. Default: all'
    )

    # Parser for the "export" command
    parser_export = subparsers.add_parser('export', help='Export container/s')

    # Parser for the "import" command
    parser_import = subparsers.add_parser('import',  help='Import container/s')
    
    # Parser for the "list" command
    parser_list = subparsers.add_parser('list', help='List of created containers')
    parser_list.add_argument(
        '-a', '--all', 
        action = "store_true", 
        help='Show the full info of the created container/s by all users'
    )
    parser_list.add_argument(
        '-s', '--size', 
        action = "store_true", 
        help='Show the size of every container'
    )
    
    parser_list.add_argument(
        '-d', '--data', 
        action = "store_true", 
        help="Show the data's directories of every container"
    )
    
    parser_list.add_argument(
        '-m', '--models', 
        action = "store_true", 
        help="Show the models' directories of every container"
    )
    
    parser_list.add_argument(
        '-w', '--workspace', 
        action = "store_true", 
        help="Show the workspace's directories of every container"
    )

    # Parser for the "open" command
    parser_open = subparsers.add_parser('open', help='Open an existing container')
    parser_open.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help='Name of the container to be opened'
    )
    
    # Parser for the "remove" command
    parser_remove = subparsers.add_parser('remove', help='Remove a container')
    parser_remove.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help='Name of the container/s to be removed'
    )
    parser_remove.add_argument(
        '-fr', '--force_remove', 
        action = "store_true", 
        help='Force to remove the container without asking the user.'
    ) 
    
    # Parser for the "start" command
    parser_start = subparsers.add_parser('start', help='Start existing container/s')
    parser_start.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, 
        help='Name of the container/s to be started'
    )

    # Parser for the "stats" command
    parser_stats = subparsers.add_parser('stats', help='Show the most important statistics of the running containers')
    parser_stats.add_argument(
        '-s', '--stream', 
        action = "store_true", 
        help='Show the streaming (live) statistics of the running containers'
    )

    # Parser for the "stop" command
    parser_stop = subparsers.add_parser('stop', help='Stop existing container/s')
    parser_stop.add_argument(
        'container_name', 
        nargs = '?', 
        type=str, help='Name of the container/s to be stopped'
    )
    parser_stop.add_argument(
        '-fs', '--force_stop', 
        action = "store_true", 
        help='Force to stop the container without asking the user.'
    ) 
    
    # Parser for the "update-sys" command
    parser_update_sys = subparsers.add_parser('update-sys', help='Update of the system')
    parser_update_sys.add_argument(
        '-fu', '--force_update', 
        action = "store_true", 
        help='Force to update directly without asking user.'
    ) 
            
    # Extract subparser names
    subparser_names = subparsers.choices.keys()
    available_commands = list(subparser_names)

    # Parse arguments
    args = parser.parse_args()
         
    return args, available_commands



################################################################################################################################################
################################################################################################################################################

# ToDo: subtitute this function for the other one (see below), which uses command
def check_container_exists(name):
    """_summary_

    Args:
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    result = subprocess.run(['docker', 'container', 'ps', '-a', '--filter', f'name={name}', '--filter', 'label=aime.mlc', '--format', '{{.Names}}'], capture_output=True, text=True)
    return result.stdout.strip()

def get_docker_image(version, versions_images):
    """_summary_

    Args:
        version (_type_): _description_
        versions_images (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    for tup in versions_images:
        if tup[0] == version:
            return tup[1]
    # Raise an exception if no matching tuple is found              
    raise ValueError("No version available") 

#OPEN######################################################################################################
def run_docker_command(docker_command):
    """Run a shell command and return its output.

    Args:
        docker_command (_type_): _description_

    Returns:
        _type_: _description_
    """    
 
    result = subprocess.run(
        docker_command, 
        shell=True, 
        text=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def run_docker_command1(command):
    process = subprocess.Popen(
        command, 
        shell=False,
        text=True,
        #stdin=subprocess.PIPE, 
        #stdin=None, 
        #stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
    )
    #stdout, stderr = process.communicate()  # Communicate handles interactive input/output
    stderr = process.communicate()  # Communicate handles interactive input/output
    return stderr, process.returncode

    #return stdout.strip(), stderr.strip(), process.returncode



'''
process = subprocess.Popen(
    command, 
    shell=False,
    stdin=subprocess.PIPE, 
    #stdin=None, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True
)
while True:
    user_input = input()
    stdout, stderr = process.communicate(input=user_input)  # Communicate handles interactive input/output

def run_docker_command1(docker_command):
    """
    Run a shell command and return its output.
    Args:
    
    Return:
    
    """ 
    
    result = subprocess.run(docker_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip(), result.stderr.strip(), result.returncode
'''

def check_container_exists(container_name):
    """Check if the container exists.

    Args:
        container_name (_type_): _description_

    Returns:
        _type_: _description_
    """    

    docker_command = f'docker container ps -a --filter=name=^/{container_name}$ --filter=label=aime.mlc --format "{{{{.Names}}}}"'
    output, _, _ = run_docker_command(docker_command)
    return output
#####-------------------------------
def print_existing_container_list(container_list):
    """
    Print an ordered list with the existing containers
    Args:
    
    Return:  
    
    
    """
    for index, container in enumerate(container_list, start=1):
        print(f"{index}) {container}")
        
# ToDo: similar to get_user_selection(prompt, max_value). If possible combine both functions
def select_container(container_list):
    """
    Prompts the user to select a container from the list.
    Args:
    
    Return:      
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
            
def check_container_running(container_name):
    """
    Check if the container is running.
    
    Args:
    
    Return:
    
    """
    docker_command = f'docker container ps --filter=name=^/{container_name}$ --filter=label=aime.mlc --format "{{{{.Names}}}}"'
    output, _, _ = run_docker_command(docker_command)
    return output

def is_container_active(container_name):
    """
    Check if the container is active by inspecting its processes.
    
    Args:
        
    Return:
    
    """
    docker_command = f'docker top {container_name} -o pid'
    output, _, exit_code = run_docker_command(docker_command)
    process_count = len(output.splitlines())
    if exit_code == 0 and 2 < process_count:
        return "True"
    else:
        return "False"
#STOP#########################################################################################



#REMOVE#######################################################################################

def get_container_image(container_tag):
    """
    Get the image of the container using the container tag.
    
    Args:
    
    Return:
    
    """    
   # Get the image associated with the container
    '''
    docker_command_get_image = [
        "docker", 
        "container", 
        "ps", 
        "-a", 
        f"--filter=name={container_tag}",
        "--filter=label=aime.mlc",
        "--format {{{{.Image}}}}"
        ]    
    '''
    docker_command_get_image = [
        'docker', 
        'container', 
        'ps', 
        '-a', 
        '--filter', f'name={container_tag}', 
        '--filter', 'label=aime.mlc', 
        '--format', '{{.Images}}'
        ]
    
    #docker_command = f'docker container ps --filter=name=^/{container_name}$ --filter=label=aime.mlc --format "{{{{.Names}}}}"'
    output, _, _ = run_docker_command(docker_command_get_image)
    return output

#LIST##############################################################################################################




#STATS#############################################################################################################
# Define a function to format the container info

def format_container_stats(container_stats_dict):
    """_summary_

    Args:
        container_stats_dict (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Extract the 'Labels' field
    labels_string = container_stats_dict.get('Labels', {})
    # Retrieve the value for 'aime.mlc.USER'
    container_name = container_stats_dict["Name"].split('._.')[0]
    cpu_usage_perc = container_stats_dict["CPUPerc"]
    memory_usage = container_stats_dict["MemUsage"]
    memory_usage_perc = container_stats_dict["MemPerc"]
    processes_active = container_stats_dict["PIDs"]
    stats_to_be_printed = [f"[{container_name}]", cpu_usage_perc, memory_usage, memory_usage_perc, processes_active]

    # Format the output line
    return stats_to_be_printed #"_".join(info_to_be_printed)


def show_container_stats(stream):  
    """_summary_

    Args:
        stream (_type_): _description_
    """    
    
    
    """Fetch and optionally stream Docker container stats."""
    # Determine the correct Docker command based on the stream flag
    if stream:
        print("\nStreaming containers stats (press Ctrl+C to stop):\n")
        command = [
            "docker",
            "stats",
            "--format",'{{json .}}'
        ]
    else:
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

    try:
        # Print the final processed output
        # Define an output format string with specified widths
        format_string = "{:<30}{:<10}{:<25}{:<10}{:<15}"
        print(f"\n{INFO}Running containers:{RESET}\n")
        titles = ["CONTAINER", "CPU %", "MEM USAGE / LIMIT", "MEM %", "PROCESSES (PIDs)"]
        while True:           
            
            # Break the loop if no stdout_data is received (for non-stream mode)
            if not stdout_data and not stream:
                process.terminate()
                break
            if stdout_data:
            
                # Split into individual lines and process them as JSON objects
                output_lines = []

                # Split by newlines and parse each line as JSON
                json_lines = stdout_data.split('\n') #list

                containers_stats = [json.loads(line) for line in json_lines if line]

                # Apply formatting to all containers' info
                output_lines = list(map(format_container_stats, containers_stats))

                print(format_string.format(*titles))

                print("\n".join(format_string.format(*info) for info in output_lines)+"\n")
            
            # Exit after processing one time in non-streaming mode
            if not stream:
                process.terminate()
                break
        
    except KeyboardInterrupt:
        if stream:
            print("\nTerminating streaming of container stats.")
        process.terminate()
        process.wait()
####################################################CREATE#########################################################

def extract_from_ml_images(filename, filter_cuda_architecture=None):
    """Extract the information from the file corresponding to the supported frameworks, versions, cuda architectures and docker images.

    Args:
        filename (str): name of the file where the framework, version, cuda archicture and docker image name are provided 
        filter_cuda_architecture (str, optional): the cuda architecture, for example, "CUDA_ADA". Defaults to None.

    Returns:
        dict, list: provides a dictionary and a list of the available frameworks.
    """
    frameworks_dict = {}
    headers = ['framework', 'version', 'cuda architecture', 'docker image']
    
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file, fieldnames=headers)
        for row in reader:
            stripped_row = {key: value.strip() for key, value in row.items() }
            framework = stripped_row['framework']
            version = stripped_row['version']
            cuda_architecture = stripped_row['cuda architecture']
            docker_image = stripped_row['docker image']
            
            if cuda_architecture == filter_cuda_architecture or filter_cuda_architecture is None:
                if framework not in frameworks_dict:
                    frameworks_dict[framework] = []
                    frameworks_dict[framework].append((version, docker_image))
                else:
                    frameworks_dict[framework].append((version, docker_image))
    # ToDo: add the possibility to select another CUDA architecture                
    if not frameworks_dict:
        print(f"No frameworks found with CUDA architecture '{filter_cuda_architecture}'.")
        sys.exit(1)
    
    return frameworks_dict


def get_container_name(container_name, user_name, command):
    """_summary_

    Args:
        container_name (_type_): _description_
        user_name (_type_): _description_
        command (_type_): _description_

    Returns:
        _type_: _description_
    """

    available_user_containers, _ = existing_user_containers(user_name, command)  
    
    if container_name is not None:
        return validate_container_name(container_name)
    else:
        while True:                           
            container_name = input(f"\n{REQUEST}Enter a container name (valid characters: a-z, A-Z, 0-9, _,-,#): {RESET}")
            if container_name in available_user_containers:
                print(f'\n{INPUT}[{container_name}]{RESET} {ERROR}already exists. Provide a new container name.{RESET}')
                show_container_info(False)
                continue
            try:
                return validate_container_name(container_name)
            except ValueError as e:
                print(e)

def existing_user_containers(user_name, mlc_command):
    """Provide a list of existing containers created previously by the current user


    Args:
        user_name (_type_): _description_
        mlc_command (_type_): _description_

    Returns:
        _type_: _description_
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
    container_names = [re.match(r"^(.*?)\._\.\w+$", container).group(1) for container in container_tags]

    return container_names, container_tags


def validate_container_name(container_name):
    """_summary_

    Args:
        container_name (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    pattern = re.compile(r'^[a-zA-Z0-9_\-#]*$')
    if not pattern.match(container_name):
        invalid_chars = [char for char in container_name if not re.match(r'[a-zA-Z0-9_\-#]', char)]
        invalid_chars_str = ''.join(invalid_chars)
        raise ValueError(f"The container name {INPUT}{container_name}{RESET} contains {ERROR}invalid{RESET} characters: {ERROR}{invalid_chars_str}{RESET}")
    return container_name

def show_container_info(args_all):
    """_summary_

    Args:
        args_all (_type_): _description_
    """
    # Adapt the filter to the selected flags (no flag or --all)
    filter_aime_mlc_user =  "label=aime.mlc" if args_all else f"label=aime.mlc={user_name}"
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
    stdout_data = str(stdout_data.strip())

    # Check for any errors
    if process.returncode != 0:
        print(f"Error: {stderr_data}")
    else:
        # Split into individual lines and process them as JSON objects
        output_lines = []
        try:
            # Split by newlines and parse each line as JSON
            json_lines = stdout_data.split('\n') #list
            containers_info = [json.loads(line) for line in json_lines if line]

            # Apply formatting to all containers' info
            output_lines = list(map(format_container_info, containers_info))
        
        except json.JSONDecodeError:
            print("Failed to decode JSON output.")
        except KeyError as e:
            print(f"Missing expected key: {e}")
    # Define an output format string with specified widths
    format_string = "{:<30}{:<30}{:<10}{:<20}{:<30}"
    print(f"\n{INFO}Current status of the containers:{RESET}\n")
    titles = ["CONTAINER", "SIZE", "USER", "FRAMEWORK", "STATUS"]
    print(format_string.format(*titles))
    print("\n".join(format_string.format(*info) for info in output_lines)+"\n")

# Define a function to format the container info
def format_container_info(container_info_dict):
    """_summary_

    Args:
        container_info_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Extract the 'Labels' field
    labels_string = container_info_dict.get('Labels', {})
    # Split the labels string into key-value pairs
    labels_dict = dict(pair.split('=', 1) for pair in labels_string.split(','))
    # Retrieve the value for 'aime.mlc.USER'
    container_name = labels_dict["aime.mlc.NAME"]
    size = container_info_dict['Size']
    user_name = labels_dict.get("aime.mlc.USER", "N/A")
    framework = labels_dict["aime.mlc.FRAMEWORK"]
    status = container_info_dict['Status']
    
    info_to_be_printed = [f"[{container_name}]", size, user_name, framework, status]
    return info_to_be_printed 

def get_user_selection(prompt, max_value):
    """
    
    Args:
    
    Return:
    
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

def display_frameworks(frameworks_dict):
    """_summary_

    Args:
        frameworks_dict (_type_): _description_

    Returns:
        _type_: _description_
    """  
    print(f"\n{REQUEST}Select a framework:{RESET}")
    framework_list = list(frameworks_dict.keys())

    for i, framework in enumerate(framework_list, start=1):
        print(f"{i}) {framework}")
    return framework_list

def display_versions(framework, versions):
    """_summary_

    Args:
        framework (_type_): _description_
        versions (_type_): _description_
    """    
    print(f"\n{INFO}Available versions for {framework}:{RESET}")
    for i, (version, _) in enumerate(versions, start=1):
        print(f"{i}) {version}")

# ToDO: the below function is an option to merge the above frameworks and versions functions. Check if it is right?
def display_frameworks_versions(framework_list,  display = "framework"):
    """_summary_

    Args:
        framework_list (_type_): _description_
        display (str, optional): _description_. Defaults to "framework".
    """    
    if display == "framework":
        print(f"\n{REQUEST}Select a framework:{RESET}")
        for i, framework in enumerate(framework_list, start=1):
            print(f"{i}) {framework}")
    elif display == "version":
        print(f"\n{INFO}Available versions for {framework}:{RESET}")
        for i, (version, _) in enumerate(versions, start=1):
            print(f"{i}) {version}")
    else:
        exit(1)
        
def are_you_sure(selected_container_name, command):
    """_summary_

    Args:
        selected_container_name (_type_): _description_
        command (_type_): _description_
    """

  
    if command == "remove":
        
        print(f"\n{WARNING}Caution: After your selection, there is no option to recover the container.{RESET}")
        
        printed_verb = command + "d"
        
    elif command == "stop":
        
        print(f"\n{WARNING}Caution: All running processes of the selected container will be terminated.{RESET}")

        printed_verb = command + "ped"
        
    else:
        exit(1)

    while True:
        are_you_sure_answer = input(f"\n{INPUT}[{selected_container_name}]{RESET} {REQUEST}will be {printed_verb}. Are you sure(y/N)?: {RESET}").strip().lower()
        
        if are_you_sure_answer in ["y", "yes"]:                          

            break
        
        elif are_you_sure_answer in ["n", "no", ""]:
            
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}will not be {printed_verb}.\n{RESET}")
            
            exit(0)
            
        else:
            
            print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}") 


def filter_by_state(state, running_containers, *lists):
    """
    Filters multiple lists based on the provided state (True/False).

    Args:
        state (bool): The state to filter by (True for running, False for not running).
        running_containers (list): A list of boolean values indicating running status.
        *lists: Variable number of lists to filter based on the running_containers.

    Returns:
        list: A list of filtered lists.
    """
    
    return [
        [item for item, running in zip(lst, running_containers) if running == state]
        for lst in lists
    ]

def filter_running_containers(running_containers, *lists):
    """
    Filters multiple lists (e.g., running_containers and running_container_tags) 
    based on the running_containers_state list, using filter_by_state.

    Args:
        running_containers (list): A list of boolean values indicating running status.
        *lists: Variable number of lists to filter based on running_containers.

    Returns:
        tuple: A flattened tuple of no_running and running filtered lists and lengths of the lists.
    """
    
    no_running_results = filter_by_state(False, running_containers, *lists) 
    running_results = filter_by_state(True, running_containers, *lists) 
    
    # Calculate lengths
    no_running_length = len(no_running_results[0])  
    running_length = len(running_results[0])  

    # Return a flattened tuple with no_running followed by running results
    return (*no_running_results, no_running_length, *running_results, running_length)

    
def print_info_header(command):
    """

    Args:
        command (_type_): _description_
    """       
    
    if command == "create":
        print(
            "\n" \
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Create a new container \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc create <container_name> <framework_name> <framework_version> -w /home/$USER/workspace -d /data -ng 1 \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc create pt231aime Pytorch 2.3.1-aime -w /home/user_name/workspace -d /data -ng 1\n" 
        )  
    if command == "export":
        #print("comming soon")
                
        print(
            "\n" \
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Export an existing container/image \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc export <container_name> <name_of_exported_container> <location_of_exported_container_and_image>  \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc export pt231aime pt231_llama /home/user_name/workspace/ \n" 
        )  
        
    if command == "import":
        #print("comming soon")
        
        print(
            "\n" \
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Import an existing container/image \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc import <name_of_imported_container> <name_of_exported_container> <location_of_imported_container_and_image>  \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc import pt231_llama pt231_llama_test /home/user_name/workspace/ \n" 
        )   
        
        
    if command == "list":
        
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    List of created containers  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc list  [-a|--all] [-s|--size] [-w|--workspace] [-d|--data] [-m|--models]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc list -a\n"
        )
        
        
    if command == "open":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Open an existing machine learning container  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc open <container_name>  \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc open pt231aime\n"
        )
        
    if command == "remove":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Remove an existing and no running machine learning container  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc remove <container_name>  [-fr | --force_remove]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc remove pt231aime -fr\n"
        )   

    if command == "start":        
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Start an existing machine learning container  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc start <container_name>  \
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc start pt231aime\n"
        )
        
    if command == "stats":
        print("to be added!!")

    if command == "stop":        
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Stop an existing machine learning container  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc stop <container_name> [-fs | --force_stop]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc stop pt231aime -fs\n"
        )  

    if command == "update-sys":
        print(
            "\n"\
            f"    {INFO_HEADER}Info{RESET}: \
            \n    Update the system  \
            \n\n    {INFO_HEADER}Correct Usage{RESET}: \
            \n    mlc update-sys [-fu | --force_update]\
            \n\n    {INFO_HEADER}Example{RESET}: \
            \n    mlc update-sys -fu\n"
        )  
        
    
        
###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################
def main():
    try: 
        # Needed tasks/info for all features
        repo_name = 'ml_images.repo'
        filter_cuda_architecture = 'CUDA_ADA'    
        
        # Arguments parsing
        args, available_commands = get_flags()
       
        if not args.command:
            print(f"\nUse {INPUT}mlc -h{RESET} or {INPUT}mlc --help{RESET} to get more informations about the AIME MLC tool.\n")
        elif args.command not in available_commands:
            print("Hallo")
        # Read and save content of ml_images.repo
        # OLD: repo_file = os.path.join(os.path.dirname(__file__), repo_name)
        repo_file = Path(__file__).parent / repo_name

        # Extract framework, version and docker image from the ml_images.repo file
        framework_version_docker = extract_from_ml_images(repo_file, filter_cuda_architecture)
        framework_version_docker_sorted = dict(sorted(framework_version_docker.items()))
        
        if args.command == 'create':

            if args.container_name is None and args.framework is None and args.version is None:
                print_info_header(args.command)
                """ 
                print(
                    "\n" \
                    f"    {GREEN}Info{RESET}: \
                    \n    Create a new container. \
                    \n\n    {GREEN}Correct Usage{RESET}: \
                    \n    mlc create <container_name> <framework_name> <framework_version> -w /home/$USER/workspace -d /data -ng 1 \
                    \n\n    {GREEN}Example{RESET}: \
                    \n    mlc create pt231aime Pytorch 2.3.1-aime -w /home/user_name/workspace -d /data -ng 1\n" 
                )                
                """
            # User provides the container name and is validated
            validated_container_name = get_container_name(args.container_name, user_name, args.command)
                        
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            # Generate a unique container tag
            provided_container_tag = f"{validated_container_name}._.{user_id}"
            
            if provided_container_tag in available_user_container_tags:
                print(f"\n{INPUT}[{validated_container_name}]{RESET} {ERROR}already exists.{RESET}")
                show_container_info(False)
                sys.exit(-1)

            # Framework selection:
            if args.framework is None:            
                while args.framework is None:
                    #framework_list = framework_version_docker_sorted.keys()
                    
                    framework_list = display_frameworks(framework_version_docker_sorted)
                    framework_num = get_user_selection(f"{REQUEST}Enter the number of the desired framework: {RESET}", len(framework_list))
                    args.framework = framework_list[framework_num - 1]
            else:    
                #available_frameworks = framework_version_docker_sorted.keys()
                while True:
                    if not framework_version_docker_sorted.get(args.framework):
                        framework_list = display_frameworks(framework_version_docker_sorted)
                        framework_number = get_user_selection(f"{REQUEST}Enter the number of the desired framework: {RESET}", len(framework_list))
                        args.framework = framework_list[framework_number - 1]
                    else:
                        break
                
            selected_framework = args.framework
                
            # Version selection:
            versions_images = framework_version_docker_sorted[selected_framework]
            if args.version is None:
                while args.version is None:  
                    display_versions(selected_framework, versions_images)
                    version_number = get_user_selection(f"{REQUEST}Enter the number of your version: {RESET}", len(versions_images))
                    # Version and docker image selected
                    args.version, selected_docker_image = versions_images[version_number - 1]
                workspace_be_asked= data_dir_be_asked = models_be_asked = True
            else:
                workspace_be_asked= data_dir_be_asked = models_be_asked = False
                available_versions = [version[0] for version in versions_images]
                while True:
                    if args.version in available_versions:
                        selected_docker_image = get_docker_image(args.version, versions_images)
                        break
                    else:
                        display_versions(selected_framework, versions_images)
                        version_number = get_user_selection(f"{REQUEST}Enter the number of your version: {RESET}", len(versions_images))
                        # Version and docker image selected
                        args.version, selected_docker_image = versions_images[version_number - 1]
            
            selected_version = args.version
            
            # Workspace directory selection:
            # Default workspace directory
            default_workspace_dir = os.path.expanduser('~/workspace')
            
            if workspace_be_asked:
                workspace_message = (
                    f"\n{INFO}The workspace directory would be mounted by default as /workspace in the container.{RESET}"
                    f"\n{INFO}It is the directory where your project data should be stored to be accessed inside the container.{RESET}"
                    f"\n{HINT}HINT: It can be set to an existing directory with the option '-w /your_workspace'{RESET}"
                )
                #print(f"\n{INFO}The workspace directory would be mounted by default as /workspace in the container.{RESET}")
                #print(f"{INFO}It is the directory where your project data should be stored to be accessed inside the container.{RESET}")
                #print(f"{HINT}HINT: It can be set to an existing directory with the option '-w /your_workspace'{RESET}")
                print(f"{workspace_message}")
                keep_workspace_dir = input(f"\n{REQUEST}Current workspace location:{default_workspace_dir}. Keep it (Y/n)?: {RESET}").strip()
                # Define a variable to control breaking out of both loops
                break_inner_loop = False
                while True:
                    if keep_workspace_dir in ["y","yes",""]:
                        #print(f'\n{INFO}Workspace directory:{RESET} {INPUT}{default_workspace_dir}{RESET}')
                        workspace_dir = default_workspace_dir
                        break
                    elif keep_workspace_dir in ["n","no"]:
                        while True:
                            provided_workspace_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the workspace directory: {RESET}").strip())  # Expand '~' to full path
                            # Check if the provided workspace directory exist:
                            if os.path.isdir(provided_workspace_dir):
                                print(f'\n{INFO}Workspace directory changed to:{RESET} {INPUT}{provided_workspace_dir}{RESET}')
                                workspace_dir = provided_workspace_dir
                                break_inner_loop = True
                                break
                            else:
                                print(f"\n{ERROR}Workspace directory not existing:{RESET} {INPUT}{provided_workspace_dir}{RESET}") 
                        if break_inner_loop:
                            break                           
                    else:
                        print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}")
                        #break
            else:
                # If the -w option is not provided
                workspace_dir = default_workspace_dir            
          
            if args.workspace_dir:
                # If the -w option is provided, use the user-provided path
                provided_workspace_dir = os.path.expanduser(args.workspace_dir)
                while True:
                    # Check if the provided workspace directory exist:
                    if os.path.isdir(provided_workspace_dir):
                        break
                    else:
                        print(f"\n{ERROR}Workspace directory not existing:{RESET} {INPUT}{provided_workspace_dir}{RESET}")
                        provided_workspace_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the workspace directory: {RESET}").strip())
                workspace_dir = provided_workspace_dir
                print(f'\n{INFO}Workspace directory changed to:{RESET} {INPUT}{provided_workspace_dir}{RESET}')
 
            args.workspace_dir = workspace_dir          
            # ToDO: check the below condition. It looks like the condition will be nevel fullfilled.           
            # Check if the provided workspace directory exist:
            if os.path.isdir(args.workspace_dir):
                print(f"\n{INFO}/workspace will be mounted on:{RESET} {INPUT}{args.workspace_dir}{RESET}")
            else:
                workspace_aborted_message = f"\n{ERROR}Aborted.\n\nWorkspace directory not existing:{RESET} {INPUT}{args.workspace_dir}{RESET}" + workspace_message
                print(f"{workspace_aborted_message}")
                #print(f"\n{ERROR}Aborted.\n\nWorkspace directory not existing:{RESET} {INPUT}{args.workspace_dir}{RESET}")
                #print(f"\n{INFO}The workspace directory would be mounted as /workspace in the container.{RESET}")
                #print(f"\n{INFO}It is the directory where your project data should be stored to be accessed\ninside the container.{RESET}")
                #print(f"\n{INFO}It can be set to an existing directory with the option '-w /your_workspace'{RESET}")
                sys.exit(-1)
            
            # Data directory:            
            if data_dir_be_asked:
                data_message = (
                    f"\n{INFO}The data directory would be mounted as /data in the container.{RESET}"
                    f"\n{INFO}It is the directory where data sets, for example, mounted from\nnetwork volumes can be accessed inside the container.{RESET}"
                    f"\n{HINT}HINT: It can be set to an existing directory with the option '-d /your_data_directory'{RESET}"
                )
                #print(f"\n{INFO}The data directory would be mounted as /data in the container.{RESET}")
                #print(f"\n{INFO}It is the directory where data sets, for example, mounted from\nnetwork volumes can be accessed inside the container.{RESET}")
                #print(f"{HINT}HINT: It can be set to an existing directory with the option '-d /your_data_directory'{RESET}")
                print(f"{data_message}")
                provide_data_dir = input(f"\n{REQUEST}Do you want to provide a data directory (y/N)?: {RESET}").strip()
                # Define a variable to control breaking out of both loops
                break_inner_loop = False
                while True:
                    if provide_data_dir in ["n","no", ""]:
                        print(f'\n{INFO}Data directory:{RESET} {INPUT}-{RESET}')
                        break
                    elif provide_data_dir in ["y","yes"]:
                        while True:
                            provided_data_dir = os.path.expanduser(input(f"\n{REQUEST}Provide the new location of the data directory: {RESET}").strip())  # Expand '~' to full path
                            # Check if the provided data directory exists:
                            if os.path.isdir(provided_data_dir):
                                print(f'\n{INFO}Data directory will be:{RESET} {INPUT}{provided_data_dir}{RESET}')
                                data_dir = provided_data_dir
                                break_inner_loop = True
                                break
                            else:
                                print(f"\n{ERROR}Provided directory does not exist:{RESET} {INPUT}{provided_data_dir}{RESET}") 
                        if break_inner_loop:
                            break                           
                    else:
                        print(f"{ERROR}\nInvalid input. Please use y(yes) or n(no).{RESET}")
                        break
            else:
                # If the -w option is not provided
                print(f"\n{INFO}Data directory:{RESET} {INPUT}-{RESET}")
            
            ###################################################
            # Check if the provided data directory exists:
            if args.data_dir:
                data_dir = os.path.realpath(args.data_dir)
                if os.path.isdir(data_dir):
                    print(f"\n{INFO}/data will be mounted on:{RESET} {INPUT}{data_dir}{RESET}")
                else:
                    print(f"\n{ERROR}Aborted.\n\nData directory not existing: {data_dir}{RESET}")
                    print(f"\n{INFO}The data directory would be mounted as /data in the container.{RESET}")
                    print(f"\n{INFO}It is the directory where data sets, for example, mounted from\nnetwork volumes can be accessed inside the container.{RESET}")
                    sys.exit(-1)           
            
            print(f"\n{INFO}Selected Framework and Version:{RESET} {INPUT}{selected_framework} {selected_version}{RESET}")
            

            #############################################DOCKER TASKS START######################################################################
            # Generate a unique container tag
            container_tag = f"{validated_container_name}._.{user_id}"
                       
            # Check if a container with the generated tag already exists
            if container_tag == check_container_exists(container_tag):
                print(f"\n{ERROR}Error:{RESET} \n {INPUT}[{validated_container_name}]{RESET} already exists.{RESET}")
                show_container_info(False)
                sys.exit(-1)
            else:
                print(f"\n{INFO}The container will be created:{RESET} {INPUT}{validated_container_name}{RESET} ")


            # Pulling the required image from aime-hub: 
            print(f"\n{INFO}Acquiring container image ... {RESET}\n")
            docker_command_pull_image = ['docker', 'pull', selected_docker_image]         
            subprocess.run(docker_command_pull_image)
        
            print(f"\n{INFO}Setting up container ... {RESET}")
            
            # Adding 
            container_label = "aime.mlc"
            workspace = "/workspace"
            data = "/data"
            
            # Run the Docker Container: Starts a new container, installs necessary packages, 
            # and sets up the environment.
            # ToDo(OK): subtitute in docker_run_cmd the os.getlogin() by user_name (see below)
            bash_command_prepare_cmd = (
                f"echo \"export PS1='[{validated_container_name}] `whoami`@`hostname`:\${{PWD#*}}$ '\" >> ~/.bashrc; "
                "apt-get update -y > /dev/null; "
                "apt-get install sudo git -q -y > /dev/null; "
                f"addgroup --gid {group_id} {user_name} > /dev/null; "
                f"adduser --uid {user_id} --gid {group_id} {user_name} --disabled-password --gecos aime > /dev/null; "
                f"passwd -d {user_name}; "
                f"echo \"{user_name} ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/{user_name}_no_password; "
                f"chmod 440 /etc/sudoers.d/${user_name}_no_password; "
                "exit"
            )
                        
            docker_prepare_container = [                
                'docker', 'run', 
                '-v', f'{args.workspace_dir}:{workspace}', 
                '-w', workspace,
                '--name', container_tag, 
                '--tty', 
                '--privileged', 
                '--gpus', args.num_gpus,
                '--network', 'host', 
                '--device', '/dev/video0', 
                '--device', '/dev/snd',
                '--ipc', 'host', 
                '--ulimit', 'memlock=-1', 
                '--ulimit', 'stack=67108864',
                '-v', '/tmp/.X11-unix:/tmp/.X11-unix', 
                selected_docker_image, 
                'bash', '-c',
                bash_command_prepare_cmd
            ]
      
            # ToDo: not use subprocess.run but subprocess.Popen 
            result_run_cmd = subprocess.run(docker_prepare_container, capture_output=True, text=True )

            # Commit the Container: Saves the current state of the container as a new image.
            # ToDo: not use subprocess.run but subprocess.Popen
            #result_commit = subprocess.run(['docker', 'commit', container_tag, f'{selected_docker_image}'], capture_output=True, text=True)#, stdout = subprocess.DEVNULL, stderr =subprocess.DEVNULL)
            bash_command_commit = [
                'docker', 'commit', container_tag, f'{selected_docker_image}:{container_tag}'
            ]
            result_commit = subprocess.run(bash_command_commit, capture_output=True, text=True)#, stdout = subprocess.DEVNULL, stderr =subprocess.DEVNULL)
            
            # ToDo: capture possible errors and treat them  

            # Remove the Container: Cleans up the initial container to free up resources.
            # ToDo: not use subprocess.run but subprocess.Popen
            result_remove = subprocess.run(['docker', 'rm', container_tag], capture_output=True, text=True)

            # Create but not run the final Docker container with labels and user configurations and setting up volume mounts
            #volumes = f'-v {args.workspace_dir}:{workspace}'

            volumes = ['-v', f'{args.workspace_dir}:{workspace}'] 
            # Add the data volume mapping if args.data_dir is set
            if args.data_dir:
                volumes +=  ['-v', f'{args.data_dir}:{data}']
            
            bash_command_create_cmd = (
                f"echo \"export PS1='[{validated_container_name}] `whoami`@`hostname`:\${{PWD#*}}$ '\" >> ~/.bashrc; bash"
            )

            docker_create_cmd = [
                'docker', 'create', 
                '-it', 
                '-w', workspace, 
                '--name', container_tag,
                '--label', f'{container_label}={user_name}', 
                '--label', f'{container_label}.NAME={validated_container_name}',
                '--label', f'{container_label}.USER={user_name}', 
                '--label', f'{container_label}.VERSION={mlc_container_version}',
                '--label', f'{container_label}.WORK_MOUNT={args.workspace_dir}', 
                '--label', f'{container_label}.DATA_MOUNT={args.data_dir}',
                '--label', f'{container_label}.FRAMEWORK={selected_framework}-{selected_version}', 
                '--label', f'{container_label}.GPUS={args.num_gpus}',
                '--user', f'{user_id}:{group_id}', 
                '--tty', 
                '--privileged', 
                '--interactive', 
                '--gpus', args.num_gpus,
                '--network', 'host', 
                '--device', '/dev/video0', 
                '--device', '/dev/snd', 
                '--ipc', 'host',
                '--ulimit', 'memlock=-1', 
                '--ulimit', 'stack=67108864', 
                '-v', '/tmp/.X11-unix:/tmp/.X11-unix', 
                '--group-add', 'video', 
                '--group-add', 'sudo', f'{selected_docker_image}:{container_tag}', 'bash', '-c',
                bash_command_create_cmd
            ]

            # Insert the volumes list at the correct position, after '-it'
            docker_create_cmd[3:3] = volumes
       

            # ToDo: not use subprocess.run but subprocess.Popen
            result_create_cmd = subprocess.run(docker_create_cmd, capture_output= True, text=True)
            
            print(f"\n{INPUT}[{validated_container_name}]{RESET} {INFO}ready. \n\nOpen the container with: mlc open{RESET} {INPUT}{validated_container_name}{RESET}\n")

        if args.command == 'export':
            
            print(f"{INFO}\nNot yet implemented. Coming soon!{RESET}\n")
            
            print_info_header(args.command) 
            
        if args.command == 'import':
            
            print(f"\n{INFO}Not yet implemented. Coming soon!{RESET}\n")
            
            print_info_header(args.command) 
                        
        if args.command == 'list':
                   
            print_info_header(args.command)          
    
            show_container_info(args.all)                    
            
        if args.command == 'open':           
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            if args.container_name:     
                if args.container_name not in available_user_containers:
                    print(f"\n{ERROR}No containers found matching the name{RESET} {INPUT}[{args.container_name}]{RESET} {ERROR}of the current user.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"\n{INFO}Available containers of the current user:{RESET}")
                    print_existing_container_list(available_user_containers)
                    selected_container_name, selected_container_position = select_container(available_user_containers)
                    print(f"\n{INFO}Selected container to be opened:{RESET} {INPUT}{selected_container_name}{RESET}")                            
                else:
                    selected_container_name = args.container_name
                    selected_container_position = available_user_containers.index(args.container_name) + 1
                    print(f'\n{INFO}Provided container name {RESET}{INPUT}[{args.container_name}]{RESET}{INFO} exists and will be opened.{RESET}')                   
            else:
                
                print_info_header(args.command)

                """
                print(
                    "\n" + '_'*100 + "\n" \
                    f"\t{INFO}Info{RESET}: \
                    \n\tOpen an existing machine learning container.  \
                    \n\t{INFO}Correct Usage{RESET}: \
                    \n\tmlc open <container_name>  \
                    \n\t{INFO}Example{RESET}: \
                    \n\tmlc open pt231aime\n" + '_'*100
                )
                """              

                # ToDo: create a function ( the same 4 lines as above)
                print(f"\n{INFO}Available containers of the current user:{RESET}")
                print_existing_container_list(available_user_containers)
                selected_container_name, selected_container_position = select_container(available_user_containers)
                print(f"\n{INFO}Selected container to be opened:{RESET} {INPUT}{selected_container_name}{RESET}")
            
            # Obtain container_tag from the selected container name
            selected_container_tag = available_user_container_tags[selected_container_position-1]
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}starting container...{RESET}")
                docker_command = f"docker container start {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command)
            else:
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container already running.{RESET}")

            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}opening shell to container...{RESET}")
                             
            # Set environment variables to pass to the Docker container
            set_env = f"-e DISPLAY={os.environ.get('DISPLAY')}"
            
            # If the NCCL_P2P_LEVEL environment variable is set, include it in the environment settings
            if 'NCCL_P2P_LEVEL' in os.environ:
                set_env += f" -e NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')}"   

            # Open an interactive shell session in the running container as the current user
            docker_command_open_shell=[
                "docker", "exec", # Docker command
                "-it",            # Flag i: interactive
                                  # Flag t: allocates a pseudo-TTY (terminal), allowing you to interact with the container.
                set_env, # 
                "--user", f"{user_id}:{group_id}", f"{selected_container_tag}", # # User and group IDs and name of the container_tag                  
                "bash"  # Command to execute inside the container
            ]
            error_mesage, exit_code = run_docker_command1(docker_command_open_shell)
            # While the shell is open, the user can run commands, check the environment, or perform any other tasks inside the container.
            # Python script is effectively paused at this point, waiting for the shell session to finish.
            # Now the user ends the session. How the shell session was terminated?
            if exit_code == 1:
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}detached from container, container keeps running.{RESET}")
            else:
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container shell closed with exit code {exit_code}.{RESET}") 
            
            # Check the status of the opened container     
            active_status = is_container_active(selected_container_tag)

            if active_status == "True":
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{INFO}container is active, kept running.{RESET}")
            else:
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{INFO} container is inactive, stopping container ...{RESET}")
                docker_command_stop_container = f"docker container stop {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command_stop_container)
                print(f"\n{INPUT}[{selected_container_name}]{RESET}{INFO} container stopped.{RESET}\n")  

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
                    show_container_info(False)
                    exit(0)
                    
                if args.container_name in running_containers:

                        
                    print(f"\n{ERROR}The provided container{RESET} {INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be removed.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"\n{INFO}Only the following no running containers of the current user can be removed:{RESET} ")
                    print_existing_container_list(no_running_containers)
                    selected_container_name, selected_container_position = select_container(no_running_containers)
                    print(f"\n{INFO}Selected container to be removed:{RESET} {INPUT}{selected_container_name}{RESET}")      
                                          
                elif args.container_name in no_running_containers:
                    selected_container_name = args.container_name
                    selected_container_position = no_running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {INFO}is not running and will be removed.{RESET}')
                    if not args.force_remove:
                        
                        print(f"\n{HINT}Hint: Use the flag -fr or --force_remove to avoid be asked.{RESET}")

                    else:
                        ask_are_you_sure = False
                else:
                    print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.")
                    
                    # all containers are running
                    if no_running_container_number == 0:
                        show_container_info(False)
                        exit(0) 
                    while True:
                        # ToDo: create a function ( the same 4 lines as below)
                        print(f"\n{INFO}Only the following no running containers of the current user can be removed:{RESET} ")
                        print_existing_container_list(no_running_containers)
                        selected_container_name, selected_container_position = select_container(no_running_containers)
                        print(f"\n{INFO}Selected container to be removed:{RESET} {INPUT}{selected_container_name}{RESET}")    
                        break       
                    
                                      
            else:  
                
                print_info_header(args.command) 
                 
                # check that at least 1 container is no running
                if no_running_container_number == 0:
                    print(f"\n{ERROR}All containers are running.\nIf you want to remove a container, stop it before using:{RESET}{HINT}\nmlc stop container_name{RESET}")
                    show_container_info(False)
                    exit(0)
                


                """
                print(
                    "\n" + '_'*100 + "\n" \
                    f"\t{INFO}Info{RESET}: \
                    \n\tRemove an existing and no running machine learning container.  \
                    \n\t{INFO}Correct Usage{RESET}: \
                    \n\tmlc remove <container_name>  [-fr | --force_remove]\
                    \n\t{INFO}Example{RESET}: \
                    \n\tmlc remove pt231aime [-fr | --force_remove]\n" + '_'*100
                 )    
                """    

                # ToDo: create a function ( the same 4 lines as below)
                print(f"\n{INFO}The following no running containers of the current user can be removed:{RESET}")
                print_existing_container_list(no_running_containers)
                #print(f"\n{WARNING}Caution: After your selection, there is no option to recover the container.{RESET}")
                selected_container_name, selected_container_position = select_container(no_running_containers)
                print(f"\n{INFO}Selected container to be removed:{RESET} {INPUT}{selected_container_name}{RESET}")  
            
            # Obtain container_tag from the selected container name
            selected_container_tag = no_running_container_tags[selected_container_position-1]
            
            if ask_are_you_sure:
                are_you_sure(selected_container_name, args.command)
            
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
                    #stderr=subprocess.PIPE,
            )
            stdout, _ = process.communicate()  # Communicate handles interactive input/output
            container_image = stdout.strip()

            # Delete the container
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}deleting container ...{RESET}")
            docker_command_delete_container = f"docker container rm {selected_container_tag}"
            subprocess.Popen(docker_command_delete_container, shell=True, text=True, stdout=subprocess.PIPE).wait()

            # Delete the container's image
            print(f"\n{INFO}Deleting related image ...{RESET}")
            docker_command_rm_image = f"docker image rm {container_image}"            
            subprocess.Popen(docker_command_rm_image, shell=True).wait()

            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container removed.{RESET}\n") 
            
            
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
                    show_container_info(False) 
                    exit(0)
                
                if args.container_name in running_containers:

                    print(f"\n{ERROR}The provided container{RESET} {INPUT}[{args.container_name}]{RESET} {ERROR}exists and is running. Not possible to be started.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"\n{INFO}Only the following no running containers of the current user can be started:{RESET} ")
                    print_existing_container_list(no_running_containers)
                    selected_container_name, selected_container_position = select_container(no_running_containers)
                    print(f"{INFO}Selected container to be started:{RESET} {INPUT}{selected_container_name}{RESET}")                            
                elif args.container_name in no_running_containers:

                    selected_container_name = args.container_name
                    selected_container_position = no_running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {INFO}is not running and will be started.{RESET}')
                    
                else:
                    print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.")
                    if no_running_container_number == 0:
                        show_container_info(False)
                        exit(1)
                    while True:
                        # ToDo: create a function ( the same 4 lines as below)
                        print(f"\n{INFO}Only the following no running containers of the current user can be started:{RESET} ")
                        print_existing_container_list(no_running_containers)
                        selected_container_name, selected_container_position = select_container(no_running_containers)
                        print(f"\n{INFO}Selected container to be started:{RESET} {INPUT}{selected_container_name}{RESET}")    
                        break                      
            else:
                
                print_info_header(args.command)
                
                if no_running_container_number == 0:
                    print(
                        f"{ERROR}\nAt the moment all containers are running.\nCreate a new one and start it using:{RESET}\n{HINT}mlc start container_name{RESET}"
                    )
                    show_container_info(False) 
                    exit(0)
                

                
                """   
                print(
                    "\n" + '_'*100 + "\n" \
                    f"\t{INFO}Info{RESET}: \
                    \n\tStart an existing machine learning container.  \
                    \n\t{INFO}Correct Usage{RESET}: \
                    \n\tmlc start <container_name>  \
                    \n\t{INFO}Example{RESET}: \
                    \n\tmlc start pt231aime\n" + '_'*100
                )   
                """                

                # ToDo: create a function ( the same 4 lines as below)
                print(f"\n{INFO}The following no running containers of the current user can be started:{RESET}")
                print_existing_container_list(no_running_containers)
                selected_container_name, selected_container_position = select_container(no_running_containers)
                print(f"\n{INFO}Selected container to be started:{RESET} {selected_container_name}")  
            
            # Obtain container_tag from the selected container name
            selected_container_tag = no_running_container_tags[selected_container_position-1]
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}starting container...{RESET}")
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
                stdout, _ = process.communicate()  # Communicate handles interactive input/output
                out = stdout.strip()
                exit_code = process.returncode
                if out == selected_container_tag:
                    print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container started.\n\nTo open a shell within the container, use: mlc open{RESET} {INPUT}{selected_container_name}{RESET}\n")
                else:
                    print(f"\n{INPUT}[{selected_container_name}]{RESET} {ERROR}Error Starting Container.{RESET}")
            else:
                print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container already running.{RESET}\n")
                sys.exit(-1)


        if args.command == 'stats':
            
            # Call the function with the correct mode (stream or no-stream)
            show_container_stats(args.stream)            
            
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
                    show_container_info(False) 
                    exit(0)

                if args.container_name in no_running_containers:
 
                    print(f"\n{ERROR}The provided container{RESET} {INPUT}[{args.container_name}]{RESET} {ERROR}exists but is not running. Not possible to be stopped.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"\n{INFO}Only the following running containers of the current user can be stopped:{RESET} ")
                    print_existing_container_list(running_containers)
                    selected_container_name, selected_container_position = select_container(running_containers)
                    print(f"{INFO}Selected container to be stopped:{RESET} {INPUT}{selected_container_name}{RESET}")  
                    
                elif args.container_name in running_containers:
                    
                    selected_container_name = args.container_name
                    selected_container_position = running_containers.index(args.container_name) + 1
                    print(f'\n{INPUT}[{args.container_name}]{RESET} {INFO}is running and will be stopped.{RESET}')
                    
                    if not args.force_stop:
                        
                        print(f"\n{HINT}Hint: Use the flag -fs or --force_stop to avoid be asked.{RESET}")

                    else:
                        ask_are_you_sure = False
                else:
                    
                    print(f"\n{INPUT}[{args.container_name}]{RESET} {ERROR}does not exist.")
                                     
                    while True:
                        # ToDo: create a function ( the same 4 lines as below)
                        print(f"\n{INFO}Only the following running containers of the current user can be stopped:{RESET} ")
                        print_existing_container_list(running_containers)
                        selected_container_name, selected_container_position = select_container(running_containers)
                        print(f"\n{INFO}Selected container to be stopped:{RESET} {INPUT}{selected_container_name}{RESET}")    
                        break 
                        
                    
            else:    
                # check that at least 1 container is running
                if not running_container_tags:
                    print(f"\n{ERROR}All containers are stopped. Therefore there are no one to be stopped.{RESET}")
                    show_container_info(False)
                    exit(0)
                    
                print_info_header(args.command)

                """
                print(
                    "\n" + '_'*100 + "\n" \
                    f"\t{INFO}Info{RESET}: \
                    \n\tStop an existing machine learning container.  \
                    \n\t{INFO}Correct Usage{RESET}: \
                    \n\tmlc stop <container_name> [-fs | --force_stop]\
                    \n\t{INFO}Example{RESET}: \
                    \n\tmlc stop pt231aime [-fs | --force_stop]\n" + '_'*100
                )    
                """
                # ToDo: create a function ( the same 4 lines as below)
                print(f"\n{INFO}Running containers of the current user:{RESET}")
                print_existing_container_list(running_containers)
                selected_container_name, selected_container_position = select_container(running_containers)
            
            # Obtain container_tag from the selected container name
            selected_container_tag = running_container_tags[selected_container_position-1]   
            if ask_are_you_sure: 
                are_you_sure(selected_container_name, args.command)

            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}stopping container ...{RESET}")
            # Attempt to stop the container and store the result.
            docker_command_stop = f"docker container stop {selected_container_tag}"
            _, _, _ = run_docker_command(docker_command_stop)
            # Print a message indicating the container has been stopped.
            print(f"\n{INPUT}[{selected_container_name}]{RESET} {INFO}container stopped.{RESET}\n")

        if args.command == 'update-sys':
            # Set to False initially
            #to_be_updated_directly = False

            # Get the directory of the current script
            mlc_path = os.path.dirname(os.path.abspath(__file__))
            
            # Change the current directory to MLC_PATH
            os.chdir(mlc_path)
            
            # Check if the .git directory exists
            if not os.path.isdir(f"{mlc_path}/.git"):
                print(f"{ERROR}Failed: ML container system not installed as updatable git repo.\nAdd the AIME MLC's location to ~/.bashrc or change to the location of the AIME MLC. {RESET}")
                sys.exit(-1)  # Exit if the directory is not a git repo
                        
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
            
            if not args.force_update:
                        
                print(f"\n{HINT}Hint: Use the flag -fu or --force_update to avoid be asked.{RESET}")
                
                print(f"\n{INFO}This will update the ML container system to the latest version.{RESET}")

                # If sudo is required, ask if the user wants to check for updates
                if sudo == "":
                    reply = input(f"\n{REQUEST}Check for available updates (Y/n)?: {RESET}").strip().lower()
                    if reply not in ["y", "yes", "Y", ""]:
                        sys.exit(0)  # Exit if user does not want to check for updates

                # Fetch the latest updates from remote repo
                docker_command_git_remote = ["git", "remote", "update"]
                sudo and docker_command_git_remote.insert(0, sudo)
                subprocess.run(docker_command_git_remote)
                
                # Get the update log for commits that are new in the remote repo
                docker_command_git_log = ["git", "log", f"HEAD..origin/{branch}", "--pretty=format:%s"]
                sudo and docker_command_git_log.insert(0, sudo)
                update_log = subprocess.check_output(docker_command_git_log, text=True).strip()
                
                if update_log == "":
                    print(f"\n{INFO}ML container system is up to date.\n{RESET}")
                    sys.exit(-1)  # Exit if no updates are available
                
                # Print the update log and prompt the user to confirm update
                print(f"\n{INFO}Update(s) available.\n\nChange Log:\n{update_log}{RESET}")
                reply = input(f"\n{INFO}Update ML container system (Y/n)?: {RESET}").strip().lower()
                if reply in ["y", "yes", "Y", ""]:
                    args.update_directly = True  # Set confirmed to True if user agrees to update
                else:
                    sys.exit(0)  # Exit if user does not want to update
                
           # If confirmed, proceed with the update
            try:
                print(f"\n{INFO}Updating ML container system...{RESET}\n")
                # Pull the latest changes from the remote repo
                docker_command_git_pull = ["git", "pull", "origin", branch]
                sudo and docker_command_git_pull.insert(0, sudo)          
                subprocess.run(docker_command_git_pull)
                sys.exit(1)  # Exit after successful update
            except Exception as e:
                print(f"\n{ERROR}Error during update: {e}")
                sys.exit(-1)  # Exit with an error if update fails

    except KeyboardInterrupt:
        print(f"\n{ERROR}\nRunning process cancelled by the user{RESET}.\n")
   
           
            
if __name__ == '__main__':
    main()

    
    
