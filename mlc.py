# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 



import sys
import os
import subprocess
import argparse
import pwd
import json

import csv
import re



# Set default values for script variables
supported_arch = "CUDA_ADA"     # Supported architecture for the container
mlc_version=3                   # Version number of the machine learning container setup

# Obtain user and group id, user name for different tasks by create, open,...
user_id = os.getuid()
#user_name = subprocess.getoutput("id -un")
#user_name_pwd = pwd.getpwuid(user_id).pw_name
#ToDo: compare both user_name
user_name = os.getlogin()
group_id = os.getgid()      
#print(f"user_name_pwd: {user_name_pwd}")
#print(f"user_name_login: {user_name}")

# ANSI escape codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
ORANGE1 = "\033[38;5;208m"  # Orange color (closer to orange)
ORANGE2 = "\033[38;5;214m"  # Light orange color
RESET = "\033[0m"

#ToDo: add constants
# Definition of command constants:
STOP = 'stop'


def get_flags():
    """
    
    Args:
    
    Return:
        args: 
    """
    parser = argparse.ArgumentParser(description='Manage machine learning containers.')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=False, help='Sub-command to execute')

    # Parser for the "create" command
    parser_create = subparsers.add_parser('create', help='Create a new container')
    parser_create.add_argument('container_name', nargs='?', type=str, help='Name of the container')
    parser_create.add_argument('framework', nargs='?', type=str, help='The framework to use')
    parser_create.add_argument('version', nargs='?', type=str, help='The version of the framework')
    parser_create.add_argument('-w', '--workspace_dir', type=str, default=os.path.expanduser('~/workspace'),
                               help='Location of the workspace directory. Default: /home/$USER/workspace')
    parser_create.add_argument('-d', '--data_dir', type=str, help='Location of the data directory.')
    parser_create.add_argument('-ng', '--num_gpus', type=str, default='all',
                               help='Number of GPUs to be used. Default: all')

    # Parser for the "export" command
    parser_export = subparsers.add_parser('export', help='Export container/s')
    parser_export.add_argument('container_name', nargs = '?', type=str, help='Name of the containers to be exported')
    parser_export.add_argument('destination', nargs = '?', type=str, help='Destination to the export') 

    # Parser for the "import" command
    parser_import = subparsers.add_parser('import',  help='Import container/s')
    parser_import.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be imported')
    parser_import.add_argument('source', nargs = '?', type=str, help='Source from the import')
    
    # Parser for the "list" command
    parser_list = subparsers.add_parser('list', help='List of created ml-containers')
    parser_list.add_argument('-a', '--all', action = "store_true", help='List of the created ml-container/s by all users')

    # Parser for the "open" command
    parser_open = subparsers.add_parser('open', help='Open an existing container')
    parser_open.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be opened')
    
    # Parser for the "remove" command
    parser_remove = subparsers.add_parser('remove', help='Remove a container')
    parser_remove.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be removed')
    parser_remove.add_argument('-fr', '--force_remove', action = "store_true", help='Force to remove the container without asking the user.') 
    
    # Parser for the "start" command
    parser_start = subparsers.add_parser('start', help='Start existing container/s')
    parser_start.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be started')

    # Parser for the "stats" command
    parser_stats = subparsers.add_parser('stats', help='Show the most important statistics of the running ml-containers')
    parser_stats.add_argument('-s', '--stream', action = "store_true", help='Show the streaming (live) statistics of the running ml-containers')

    # Parser for the "stop" command
    parser_stop = subparsers.add_parser('stop', help='Stop existing container/s')
    parser_stop.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be stopped')
    parser_stop.add_argument('-fs', '--force_stop', action = "store_true", help='Force to stop the container without asking the user.') 
    # Parser for the "update-sys" command
    parser_update_sys = subparsers.add_parser('update-sys', help='Update of the system')
    #parser_update_sys.add_argument('container_name', nargs = '?', type=str, help='List of  of the container/s to open')
              
        
    # Extract subparser names
    subparser_names = subparsers.choices.keys()
    available_commands = list(subparser_names)

    # Parse arguments
    args = parser.parse_args()
    
   # If no command is provided, prompt the user to choose one
    while not args.command:
        # ToDo: use the same method: 1) create, 2) open.... instead of a list of the actions
        print(f"{GREEN}Available commands:{RESET} " + ', '.join(available_commands))
        chosen_command = input(f"{CYAN}Please choose a command:{RESET} ").strip()
        if chosen_command in available_commands:
            # Re-parse arguments with the chosen command
            sys.argv.insert(1, chosen_command)
            args = parser.parse_args()
        else:
            print(f"{RED}Invalid command:{RESET} {chosen_command}")
    
    # Convert the Namespace object to a dictionary
    args_dict = vars(args)
        
    return args, args_dict


#############################################################################################################

def extract_from_ml_images(filename, filter_cuda_architecture=None):
    """
    
    Args:
    
    Return:
    
    """
    frameworks_dict = {}
    headers = ['framework', 'version', 'cuda architecture', 'docker image']
    
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file, fieldnames=headers)
        for row in reader:
            stripped_row = {key: value.strip() for key, value in row.items() }
            #print(f"{stripped_row}") DEBUGGING
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
    
    framework_list = list(frameworks_dict.keys())
    #version_list = list()

    return frameworks_dict, framework_list #, version_list

def get_container_name(container_name, user_name, command):
    """
    
    Args:0
    
    Return:
    
    """
    # List existing containers
    available_user_containers, _ = existing_user_containers(user_name, command)  
    
    if container_name is not None:
        return validate_container_name(container_name)
    else:
        while True:                           
            container_name = input(f"{CYAN}Enter a container name: {RESET}")
            if container_name in available_user_containers:
                print(f'Provided container name [{container_name}] already exists. Provide a new container name')
                continue
            try:
                return validate_container_name(container_name)
            except ValueError as e:
                print(e)

def validate_container_name(container_name):
    """
    
    Args:
    
    Return:
    
    """
    
    pattern = re.compile(r'^[a-zA-Z0-9_\-#]*$')
    if not pattern.match(container_name):
        invalid_chars = [char for char in container_name if not re.match(r'[a-zA-Z0-9_\-#]', char)]
        invalid_chars_str = ''.join(invalid_chars)
        raise ValueError(f"The container name {ORANGE2}{container_name}{RESET} contains {RED}invalid{RESET} characters: {RED}{invalid_chars_str}{RESET}")
    return container_name


def display_frameworks(frameworks_dict):
    """
    
    Args:
    
    Return:
    
    """    
    print(f"\n{CYAN}Please, select a framework:{RESET}")
    framework_list = list(frameworks_dict.keys())
    for i, framework in enumerate(framework_list, start=1):
        print(f"{i}) {framework}")
    return framework_list

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
                print(f"{RED}Please enter a number between 1 and {max_value}.{RESET}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
# ToDo: actually not needed this function (only 1 line). Try to embed the list comprenhension within the code (OK)
'''
def get_versions(versions):
    """
    
    Args:
    
    Return:
    
    """      
    available_versions = [version[0] for version in versions]
    
    return available_versions
'''
def display_versions(framework, versions):
    """
    
    Args:
    
    Return:
    
    """    
    print(f"\n{CYAN}Available versions for {framework}:{RESET}")
    for i, (version, _) in enumerate(versions, start=1):
        print(f"{i}) {version}")

# ToDo: subtitute this function for the other one (see below), which uses command
def check_container_exists(name):
    """
    
    Args:
    
    Return:
    
    """ 
    
    result = subprocess.run(['docker', 'container', 'ps', '-a', '--filter', f'name={name}', '--filter', 'label=aime.mlc', '--format', '{{.Names}}'], capture_output=True, text=True)
    return result.stdout.strip()

def get_docker_image(version, versions_images):
    """
    
    Args:
    
    Return:
    
    """ 
    for tup in versions_images:
        if tup[0] == version:
            return tup[1]
    # Raise an exception if no matching tuple is found              
    raise ValueError("No version available") 

###############################################OPEN########################################################
def run_docker_command(docker_command):
    """
    Run a shell command and return its output.
    Args:
    
    Return:
    
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
    """
    Check if the container exists.
    Args:
    
    Return:  
    
    
    """
    docker_command = f'docker container ps -a --filter=name=^/{container_name}$ --filter=label=aime.mlc --format "{{{{.Names}}}}"'
    output, _, _ = run_docker_command(docker_command)
    return output
#####-------------------------------
def existing_user_containers(user_name, mlc_command):
    """
    Provide a list of existing containers created previously by the current user
    Args:
    
    Return:      
    
    """
    # List all containers with the 'aime.mlc' label owned by the current user
    docker_command = f"docker container ps -a --filter=label=aime.mlc.USER={user_name} --format '{{{{.Names}}}}'"
    output, _,_ = run_docker_command(docker_command)
    container_tags = output.splitlines()
    #print(f"Container_tags: {container_tags}")
    # check that at least 1 container has been created previously
    if not container_tags and mlc_command != 'create':
        print(f"{RED}Create at least one container. If not, mlc {mlc_command} does not work!!{RESET}")
        exit(1)

    # Extract base names from full container names
    container_names = [re.match(r"^(.*?)\._\.\w+$", container).group(1) for container in container_tags]

    return container_names, container_tags

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
            selection = int(input(f"{CYAN}Select the number of the container:{RESET}"))
            container_list_length = len(container_list)
            if 1 <= selection <= container_list_length:
                return container_list[selection - 1], selection
            else:
                print(f"{RED}Invalid selection. Please choose a valid number.{RESET}")
        except ValueError:
            print(f"{RED}Invalid input. Please enter a number.{RESET}")
            
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
###############################################STOP#########################################
def filter_running_containers_old(running_containers, *lists):
    """
    Filters multiple lists (running_containers and running_container_tags) based on the running_containers_state list.

    Args:
        running_containers (list): A list of boolean values indicating running status.
        *lists: Variable number of lists to filter based on running_containers.

    Returns:
        tuple: A tuple of filtered running_containers and running_container_tags.
    """
    return tuple(
        [item for item, running in zip(lst, running_containers) if running]
        for lst in lists
    )

def filter_running_containers(running_containers, running_state, *lists):
    """
    Filters multiple lists (running_containers and running_container_tags) based on the running_containers_state list.

    Args:
        running_containers (list): A list of boolean values indicating running status.
        *lists: Variable number of lists to filter based on running_containers.

    Returns:
        tuple: A tuple of filtered running_containers and running_container_tags.
    """
    return tuple(
        [item for item, running in zip(lst, running_containers) if running == running_state]
        for lst in lists
    )


###############################################REMOVE#########################################

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

###############################################LIST#########################################

# Define a function to format the container info
def format_container_info(container_info_dict):
    # Regular expression to capture the container_name, within the container_tag, located before ._. (example: containername._.1001)
    #container_name_pattern = re.compile(r'^([a-zA-Z0-9]+)\._\d{4}$')  
    #print(f"container_info_dict: {container_info_dict}")
    #container_name_tag = container_info_dict['Names']
    #print(f"container_name_tag: {container_name_tag}")
    #user = container_info_dict.get('Labels', {}).get(f'aime.mlc.{user_name}', {})
    # Extract the 'Labels' field
    labels_string = container_info_dict.get('Labels', {})
    # Split the labels string into key-value pairs
    labels_dict = dict(pair.split('=', 1) for pair in labels_string.split(','))
    #print(f"labels_dict: {labels_dict}")
    # Retrieve the value for 'aime.mlc.USER'
    container_name = labels_dict["aime.mlc.NAME"]
    size = container_info_dict['Size']
    user_name = labels_dict.get("aime.mlc.USER", "N/A")
    framework = labels_dict["aime.mlc.FRAMEWORK"]
    status = container_info_dict['Status']
    
    info_to_be_printed = [f"[{container_name}]", size, user_name, framework, status]

    #print(f"info_to_be_printed: {info_to_be_printed}")
    #output = "\t".join(info_to_be_printed)
    #print(f"info_to_be_printed: {output}")
    # Use regular expression to find the container name before ._.
    #match = container_name_pattern.match(container_name_tag)
    #container_name = match.group(1) if match else container_name_tag
    #print(f"container_name: {container_name}")
    # Format the output line
    return info_to_be_printed #"_".join(info_to_be_printed)

def show_container_info(args_all):
    # Adapt the filter to the selected flags (no flag or --all)
    filter_aime_mlc_user =  "label=aime.mlc" if args_all else f"label=aime.mlc={user_name}"
    #ERROR: docker_command_ls = f"docker container ls -a --filter{filter_aime_mlc_user} --format {{json .}}'"
    #print(f"filter_aime_mlc_user: {filter_aime_mlc_user}")
    docker_command_ls = [
        "docker", "container", 
        "ls", 
        "-a", 
        "--filter", filter_aime_mlc_user, 
        "--format", '{{json .}}'    
    ]
    #print(f"docker_command_ls: {docker_command_ls}")
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
            #print(len(json_lines))
            #print(f"type jsonlines[0]: {type(json_lines[0])}")
            #print(f"jsonlines: {json_lines}")
            containers_info = [json.loads(line) for line in json_lines if line]
            #print(f"type containers_info[0]: {type(containers_info[0])}")
            #print(f"containers_info: {containers_info}")
            #print(len(containers_info))

            # Apply formatting to all containers' info
            output_lines = list(map(format_container_info, containers_info))
        
        except json.JSONDecodeError:
            print("Failed to decode JSON output.")
        except KeyError as e:
            print(f"Missing expected key: {e}")
    # Print the final processed output
    # Define an output format string with specified widths
    format_string = "{:<15}{:<30}{:<15}{:<30}{:<30}"
    print(f"\n{GREEN}Available ml-containers are:{RESET}\n")
    titles = ["CONTAINER", "SIZE", "USER", "FRAMEWORK", "STATUS"]
    print(format_string.format(*titles))
    #print("\t".join(titles))
    #print("\n".join(output_lines))
    # Print the container info
    print("\n".join(format_string.format(*info) for info in output_lines)+"\n")
    #print("")
############################################################STATS##################################################
# Define a function to format the container info
def format_container_stats(container_stats_dict):
    # Regular expression to capture the container_name, within the container_tag, located before ._. (example: containername._.1001)
    #container_name_pattern = re.compile(r'^([a-zA-Z0-9]+)\._\d{4}$')  
    #print(f"container_info_dict: {container_info_dict}")
    #container_name_tag = container_info_dict['Names']
    #print(f"container_name_tag: {container_name_tag}")
    #user = container_info_dict.get('Labels', {}).get(f'aime.mlc.{user_name}', {})
    # Extract the 'Labels' field
    labels_string = container_stats_dict.get('Labels', {})
    # Split the labels string into key-value pairs
    #labels_dict = dict(pair.split('=', 1) for pair in labels_string.split(','))
    #print(f"labels_dict: {labels_dict}")
    # Retrieve the value for 'aime.mlc.USER'
    container_name = container_stats_dict["Name"].split('._.')[0]
    cpu_usage_perc = container_stats_dict["CPUPerc"]
    memory_usage = container_stats_dict["MemUsage"]
    memory_usage_perc = container_stats_dict["MemPerc"]
    processes_active = container_stats_dict["PIDs"]
    stats_to_be_printed = [f"[{container_name}]", cpu_usage_perc, memory_usage, memory_usage_perc, processes_active]

    #print(f"info_to_be_printed: {info_to_be_printed}")
    #output = "\t".join(info_to_be_printed)
    #print(f"info_to_be_printed: {output}")
    # Use regular expression to find the container name before ._.
    #match = container_name_pattern.match(container_name_tag)
    #container_name = match.group(1) if match else container_name_tag
    #print(f"container_name: {container_name}")
    # Format the output line
    return stats_to_be_printed #"_".join(info_to_be_printed)


def show_container_stats(stream):  
    
    """Fetch and optionally stream Docker container stats."""
    # Determine the correct Docker command based on the stream flag
    if stream:
        print("\nStreaming ml-containers stats (press Ctrl+C to stop):\n")
        #command = "docker stats --format '{{json .}}'"
        command = [
            "docker",
            "stats",
            "--format",'{{json .}}'
        ]
    else:
        #print("\nFetching ml-containers stats once:\n")
        #command = "docker stats --no-stream --format '{{json .}}'"
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
    #print("Ich bin hier")

    #stdout_data = process.stdout()
    stdout_data, stderr_data = process.communicate()
    print("Ich bin hier")
    try:
        # Print the final processed output
        # Define an output format string with specified widths
        format_string = "{:<15}{:<15}{:<30}{:<15}{:<15}"
        print(f"\n{GREEN}Running ml-containers:{RESET}\n")
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

def show_container_stats1(stream):

    """Fetch and optionally stream Docker container stats."""
    # Determine the correct Docker command based on the stream flag
    if stream:
        print("\nStreaming ml-containers stats (press Ctrl+C to stop):\n")
        #command = "docker stats --format '{{json .}}'"
        command = [
            "docker",
            "stats",
            "--format",'{{json .}}'
        ]
    else:
        #print("\nFetching ml-containers stats once:\n")
        #command = "docker stats --no-stream --format '{{json .}}'"
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
        # Print the header
        #print("CONTAINER\tCPU %\tMEM USAGE")
        format_string = "{:<15}{:<30}{:<15}{:<30}{:<30}"
        print(f"\n{GREEN}Running ml-containers:{RESET}\n")
        titles = ["CONTAINER", "CPU %", "MEM USAGE / LIMIT", "MEM %", "PROCESSES (PIDs)"]
        print(stdout_data)
        while True:
            # Read a line of output (either a single line or continuously)
            #line = stdout_data.readline().strip()
            stdout_data = str(stdout_data.strip())
            
            # Break the loop if no line is received (for non-stream mode)
            if not stdout_data and not stream:
                break
            
            if stdout_data:
                try:
                    # Parse the line as JSON
                    container_stats = json.loads(line)
                    container_name = container_stats["Name"].split('._.')[0]
                    cpu_usage = container_stats["CPUPerc"]
                    memory_usage = container_stats["MemUsage"]
                    # Print the formatted output for each container
                    print(f"{container_name}\t{cpu_usage}\t{memory_usage}")
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line}", file=sys.stderr)

            # Exit after processing one line in non-streaming mode
            if not stream:
                break

    except KeyboardInterrupt:
        if stream:
            print("\nTerminating streaming of container stats.")
        process.terminate()
        process.wait()



###############################################################################################################
    

def main():
    try: 
        # Needed tasks/info for all features
        repo_name = 'ml_images.repo'
        filter_cuda_architecture = 'CUDA_ADA'    
        
        # Arguments parsing
        args, args_dict = get_flags()
        # ToDo: check what is better: args or args_dict
            
        #### DEBUGING: Print each argument separately########
        #print(f"{dir(args)}")
        #for index, arg in enumerate(args_dict):
            
        #    print(f"Argument {index}: {arg}") 
            
        #for key, value in args_dict.items():
        #    print(f"Key({key}): Value({value})")
    
        #####################################################
        # Read and save content of ml_images.repo
        repo_file = os.path.join(os.path.dirname(__file__), repo_name)
        
        # Extract framework, version and docker image from the ml_images.repo file
        framework_version_docker, frameworks = extract_from_ml_images(repo_file, filter_cuda_architecture)
        #print(f"ML_REPO: {framework_version_docker}") #DEBUGGING
              

        if args.command == 'create':

            if args.container_name is None and args.framework is None and args.version is None:
                                
                print(
                    "\n" +\
                    f"{GREEN}Info{RESET}: \
                    \nCreate a new machine learning container. \
                    \n{GREEN}Correct Usage{RESET}: \
                    \nmlc create <container_name> <framework_name> <framework_version> -w /home/$USER/workspace -d /data -ng 1 \
                    \n{GREEN}Example{RESET}: \
                    \nmlc create pt231aime Pytorch 2.3.1-aime\n"
                )                

            # User provides the container name and be validated
            validated_container_name = get_container_name(args.container_name, user_name, args.command)
            
            # Framework part:
            if args.framework is None:            
                while args.framework is None:
                    framework_list = display_frameworks(framework_version_docker)
                    framework_num = get_user_selection(f"{CYAN}Enter the number of your framework: {RESET}", len(framework_list))
                    args.framework = framework_list[framework_num - 1]
            else:    
                available_frameworks = list(framework_version_docker.keys())
                while True:
                    if args.framework in available_frameworks:
                        break
                    else:
                        framework_list = display_frameworks(framework_version_docker)
                        framework_number = get_user_selection(f"{CYAN}Enter the number of your framework: {RESET}", len(framework_list))
                        args.framework = framework_list[framework_number - 1]
                        #print(f"Available frameworks: {', '.join(available_frameworks)}")
                        #selected_framework = input(f"{CYAN}Please select a valid framework: {RESET}")
                
            selected_framework = args.framework
                
            # Version part
            versions_images = framework_version_docker[selected_framework]
            #print(f"Versions and docker images: {versions_images}")
            if args.version is None:
                while args.version is None:  
                    #version_list = framework_version_docker[selected_framework]
                    display_versions(selected_framework, versions_images)
                    version_number = get_user_selection(f"{CYAN}Enter the number of your version: {RESET}", len(versions_images))
                    # Version and docker image selected
                    args.version, selected_docker_image = versions_images[version_number - 1]
            else:
                available_versions = [version[0] for version in versions_images]
                #available_versions = get_versions(versions_images)
                #print(f"{available_versions}")
                #print(f"I am here")
                while True:
                    if args.version in available_versions:
                        selected_docker_image = get_docker_image(args.version, versions_images)
                        break
                    else:
                        display_versions(selected_framework, versions_images)
                        version_number = get_user_selection(f"{CYAN}Enter the number of your version: {RESET}", len(versions_images))
                        # Version and docker image selected
                        args.version, selected_docker_image = versions_images[version_number - 1]
            
            selected_version = args.version
            
            # Check if the provided workspace directory exist:
            if os.path.isdir(args.workspace_dir):
                print(f"\n/workspace will mount on '{args.workspace_dir}' - OK")
            else:
                print(f"\n{RED}ERROR!!: Aborted.\n\nWorkspace directory not existing: {args.workspace_dir}{RESET}")
                print("\nThe workspace directory would be mounted as /workspace in the container.")
                print("It is the directory where your project data should be stored to be accessed\ninside the container.")
                print("\nIt can be set to an existing directory with the option '-w /your_workspace'\n")
                sys.exit(-4)
            
            # Check if the provided data directory exist:
            if args.data_dir:
                data_mount = os.path.realpath(args.data_dir)
                if os.path.isdir(data_mount):
                    print(f"/data will mount on '{data_mount}' - OK")
                else:
                    print(f"\n{RED}ERROR!!: Aborted.\n\nData directory not existing: {data_mount}{RESET}")
                    print("\nThe data directory would be mounted as /data in the container.")
                    print("It is the directory where data sets, for example, mounted from\nnetwork volumes can be accessed inside the container.")
                    sys.exit(-4)
            
            print(f"\nYou have selected: {ORANGE2}{selected_framework} {selected_version}{RESET}")
            print(f"The mlc container called {ORANGE2}{validated_container_name}{RESET} with the framework {ORANGE2}{selected_framework}{RESET} and version {ORANGE2}{selected_version}{RESET} with docker image {selected_docker_image} will be created!!")
            

            
            #############################################DOCKER TASKS START######################################################################
            # Generate a unique container tag
            container_tag = f"{validated_container_name}._.{user_id}"
            
            #print(container_tag)
            
            # Check if a container with the generated tag already exists
            if container_tag == check_container_exists(container_tag):
                print(f"\n{RED}Error:{RESET} \nContainer [{ORANGE2}{validated_container_name}{RESET}] already exists.")
                sys.exit(-1)
            else:
                print(f"{validated_container_name} will be created!!!")

            # Pulling the required image from aime-hub: 
            print("\nAcquiring container image ... \n")
            docker_command_pull_image = ['docker', 'pull', selected_docker_image]         
            subprocess.run(docker_command_pull_image)
        
            print("\nSetting up container ... \n")
            
            # Adding 
            container_label = "aime.mlc"
            workspace = "/workspace"
            data = "/data"
            
            # Run the Docker Container: Starts a new container, installs necessary packages, 
            # and sets up the environment.
            #print("docker_run_cmd will be created")
            # ToDo(OK): subtitute in docker_run_cmd the os.getlogin() by user_name (see below)
            docker_run_cmd = [                
                'docker', 'run', '-v', f'{args.workspace_dir}:{workspace}', '-w', workspace,
                '--name', container_tag, '--tty', '--privileged', '--gpus', args.num_gpus,
                '--network', 'host', 
                '--device', '/dev/video0', 
                '--device', '/dev/snd',
                '--ipc', 'host', 
                '--ulimit', 'memlock=-1', 
                '--ulimit', 'stack=67108864',
                '-v', '/tmp/.X11-unix:/tmp/.X11-unix', 
                selected_docker_image, 
                'bash', '-c',
                f'echo "export PS1=\'[{validated_container_name}] `whoami`@`hostname`:${{PWD#*}}$ \'" >> ~/.bashrc; '
                f'apt-get update -y > /dev/null; apt-get install sudo git -q -y > /dev/null; '
                f'addgroup --gid {group_id} {user_name} > /dev/null; '
                f'adduser --uid {user_id} --gid {group_id} {user_name} --disabled-password --gecos aime > /dev/null; '
                f'passwd -d {user_name}; echo "{user_name} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/{user_name}_no_password; '
                f'chmod 440 /etc/sudoers.d/{user_name}_no_password; exit'
            ]
            #print("subprocess docker_run_cmd will starten")       
            # ToDo: not use subprocess.run but subprocess.Popen 
            result_run_cmd = subprocess.run(docker_run_cmd, capture_output=True, text=True )
            #print(f"STDOUT run_cmd: {result_run_cmd.stdout}")
            #print(f"STDERR run_cmd: {result_run_cmd.stderr}")
            
            #print("subprocess docker_run_cmd finished")

            # Commit the Container: Saves the current state of the container as a new image.
            # ToDo: not use subprocess.run but subprocess.Popen
            #result_commit = subprocess.run(['docker', 'commit', container_tag, f'{selected_docker_image}'], capture_output=True, text=True)#, stdout = subprocess.DEVNULL, stderr =subprocess.DEVNULL)
            
            result_commit = subprocess.run(['docker', 'commit', container_tag, f'{selected_docker_image}:{container_tag}'], capture_output=True, text=True)#, stdout = subprocess.DEVNULL, stderr =subprocess.DEVNULL)
            #print(f"STDOUT commit: {result_commit.stdout}")
            #print(f"STDERR commit: {result_commit.stderr}")
            
            # ToDo: capture possible errors and treat them  
            #print("subprocess commit finished")

            # Remove the Container: Cleans up the initial container to free up resources.
            # ToDo: not use subprocess.run but subprocess.Popen
            result_remove = subprocess.run(['docker', 'rm', container_tag], capture_output=True, text=True)
            #print(f"STDOUT remove: {result_remove.stdout}")
            #print(f"STDERR remove: {result_remove.stderr}")
            #print("subprocess rm finished")

            # Create but not run the final Docker container with labels and user configurations and setting up volume mounts
            #volumes = f'-v {args.workspace_dir}:{workspace}'
            ''' OLD STUFF
            args.data_dir = '/data'
            volumes = ' \'-v\',' + f' {args.workspace_dir}:{workspace}'
            #volumes = '-v' + f'{args.workspace_dir}:{workspace}'
            if args.data_dir:
                volumes += ',' + '\'-v\','+ f' {args.data_dir}:{data}'
                #volumes += f'-v {args.data_dir}:{data}'
            print("subprocess commit finished")
            print(f"volumes: {volumes}")
            '''    
            volumes = ['-v', f'{args.workspace_dir}:{workspace}'] 
            # Add the data volume mapping if args.data_dir is set
            if args.data_dir:
                #volumes.extend(['-v', f'{args.data_dir}:{data}'])
                volumes +=  ['-v', f'{args.data_dir}:{data}']
            
            docker_create_cmd = [
                'docker', 'create', 
                '-it', 
                '-w', workspace, 
                '--name', container_tag,
                '--label', f'{container_label}={user_name}', 
                '--label', f'{container_label}.NAME={validated_container_name}',
                '--label', f'{container_label}.USER={user_name}', 
                '--label', f'{container_label}.VERSION={mlc_version}',
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
                f'echo "export PS1=\'[{validated_container_name}] `whoami`@`hostname`:${{PWD#*}}$ \'" >> ~/.bashrc; bash'
            ]
            #'--group-add', 'sudo', f'{selected_docker_image}:{container_tag}', 'bash', '-c',
            #'--group-add', 'sudo', f'{selected_docker_image}', 'bash', '-c',

            # Insert the volumes list at the correct position, after '-it'
            docker_create_cmd[3:3] = volumes
            
            #print(f"{docker_create_cmd}")
            #print("docker_create_cmd finished")
            # ToDo: not use subprocess.run but subprocess.Popen
            result_create_cmd = subprocess.run(docker_create_cmd, capture_output= True, text=True)
            #print(f"STDOUT create_cmd: {result_create_cmd.stdout}")
            #print(f"STDERR create_cmd: {result_create_cmd.stderr}")
            
            print(f"\n[{validated_container_name}] ready. Open the container with: mlc open {validated_container_name}\n")
        
            # docker ps --format "{{.Label \"aime.mlc.NAME\"}}"



            
        if args.command == 'open':           
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            if args.container_name:     
                if args.container_name not in available_user_containers:
                    print(f"No containers found matching the name '{args.container_name}' for the current user.")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"{GREEN}Available containers of the current user:{RESET}")
                    print_existing_container_list(available_user_containers)
                    selected_container_name, selected_container_position = select_container(available_user_containers)
                    print(f"{GREEN}Selected container to be opened:{RESET} {selected_container_name}")                            
                else:
                    selected_container_name = args.container_name
                    selected_container_position = available_user_containers.index(args.container_name) + 1
                    print(f'Provided container name [{args.container_name}] exists and will be opened.')                   
            else:
                print(
                    "\n" +\
                    f"{GREEN}Info{RESET}: \
                    \nOpen an existing machine learning container.  \
                    \n{GREEN}Correct Usage{RESET}: \
                    \nmlc open <container_name>  \
                    \n{GREEN}Example{RESET}: \
                    \nmlc open pt231aime\n"
                )

                # ToDo: create a function ( the same 4 lines as above)
                print(f"{GREEN}Available containers of the current user:{RESET}")
                print_existing_container_list(available_user_containers)
                selected_container_name, selected_container_position = select_container(available_user_containers)
                print(f"{GREEN}Selected container to be opened:{RESET} {selected_container_name}")
            
            # Obtain container_tag from the selected container name
            selected_container_tag = available_user_container_tags[selected_container_position-1]
            print(f"Container_tag: {selected_container_tag}")
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):
                print(f"[{selected_container_name}] starting container")
                docker_command = f"docker container start {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command)
            else:
                print(f"[{selected_container_name}] container already running")

            print(f"[{selected_container_name}] opening shell to container")
            
            '''
            # Set environment variables for Docker exec
            set_env = ["-e", f"DISPLAY={subprocess.getoutput('echo $DISPLAY')}"]

            # Check and add NCCL_P2P_LEVEL if it's set in the environment
            nccl_p2p_level = subprocess.getoutput("echo $NCCL_P2P_LEVEL")
            if nccl_p2p_level:
                #set_env.extend(["-e", f"NCCL_P2P_LEVEL={nccl_p2p_level}"])
                set_env.append(["-e", f"NCCL_P2P_LEVEL={nccl_p2p_level}"])


            # Open an interactive shell in the running container
            result = subprocess.Popen(
                ["docker", "exec", "-it"] + set_env + ["--user", f"{user_id}:{group_id}", {selected_container_tag}, "/bin/bash"]
            )
            exit_code = result.returncode
            '''
                            
            # Set environment variables to pass to the Docker container
            set_env = f"-e DISPLAY={os.environ.get('DISPLAY')}"
            
            # If the NCCL_P2P_LEVEL environment variable is set, include it in the environment settings
            if 'NCCL_P2P_LEVEL' in os.environ:
                set_env += f" -e NCCL_P2P_LEVEL={os.environ.get('NCCL_P2P_LEVEL')}"   
            print(f"SET_ENV value: {set_env}")
            str(5)
            "5"
            # Open an interactive shell session in the running container as the current user
            #docker_command = f"docker exec -it {set_env} --user {user_id}:{group_id} {selected_container_tag} /bin/bash"
            #'''
            docker_command_open_shell=[
                "docker", "exec", # Docker command
                "-it",   # Flag i: interactive
                            # Flag t: allocates a pseudo-TTY (terminal), allowing you to interact with the container.
                set_env, # 
                "--user", 
                f"{user_id}:{group_id}",  # User and group IDs
                f"{selected_container_tag}",  # Name of the container_tag
                "bash"  # Command to execute inside the container
            ]
            #'''

            print(f"Docker command: {docker_command_open_shell}")
            #_, error_mesage, exit_code = run_docker_command1(docker_command)

            error_mesage, exit_code = run_docker_command1(docker_command_open_shell)
            # While the shell is open, the user can run commands, check the environment, or perform any other tasks inside the container.
            # Python script is effectively paused at this point, waiting for the shell session to finish.
            #print(f"stdout: {stdout}")
            print(f'Error_mesage: {error_mesage}')
            print(f"Exit code: {exit_code}")
            # Now the user ends the session. How the shell session was terminated?
            if exit_code == 1:
                print(f"[{selected_container_name}] detached from container, container keeps running")
            else:
                print(f"[{selected_container_name}] container shell closed with exit code {exit_code}") 
            
            # Check the status of the opened container     
            active_status = is_container_active(selected_container_tag)

            if active_status == "True":
                print(f"[{selected_container_name}] container is active, kept running.")
            else:
                print(f"[{selected_container_name}] container is inactive, stopping container ...")
                docker_command_stop_container = f"docker container stop {selected_container_tag}"
                _, _, _ = run_docker_command(docker_command_stop_container)
                print(f"[{selected_container_name}] container stopped.\n")  

        if args.command == 'remove':
            
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            containers_state = [True if container_tag == check_container_running(container_tag) else False for container_tag in available_user_container_tags]
            # Automate filtering            
            no_running_containers, no_running_container_tags = filter_running_containers(
                containers_state,
                False, 
                available_user_containers, 
                available_user_container_tags
            )
            running_containers, running_container_tags = filter_running_containers(
                containers_state,
                True, 
                available_user_containers, 
                available_user_container_tags
            )
            available_container_number = len(available_user_containers)
            running_container_number = len(running_containers)
            no_running_container_number = len(no_running_containers)

            if no_running_container_number == 0:
                print(
                    f"{RED}\nERROR!!: There are only running containers ({running_container_number}) or not containers at all. \n \
                    You cannot remove running containers. Here an overview of running/not-running containers."
                )
                show_container_info(False) 
                exit(1)
            print(f"available user containers: {available_user_containers}")
            print(f"running containers: {running_containers}")      
            print(f"running container_tags: {running_container_tags}")
            print(f"container_states: {containers_state}")
            

            print(f"NO running containers: {no_running_containers}")      
            print(f"NO running container_tags: {no_running_container_tags}")            
            # check that at least 1 container is running
            #if not running_container_tags:
            #    print(f"{RED}WARNING!!: No containers are running and therefore there are no one to be stopped.{RESET}")
            #    exit(1)
            
            if args.container_name: 
                if args.container_name in running_containers:
                    print(f"{RED}The provided container [{args.container_name}] exists and is running. Not possible to be removed.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"{GREEN}Only the following no running containers of the current user can be removed:{RESET} ")
                    print_existing_container_list(no_running_containers)
                    #print(f"{RED}WARNING!!: All running processes of the selected container will be terminated and you will NOT ask again. Be sure about your decision.{RESET}")
                    selected_container_name, selected_container_position = select_container(no_running_containers)
                    print(f"{GREEN}Selected container to be removed:{RESET} {selected_container_name}")                            
                elif args.container_name in no_running_containers:
                    selected_container_name = args.container_name
                    selected_container_position = no_running_containers.index(args.container_name) + 1
                    print(f'\n[{args.container_name}] is not running and will be removed.')
                    if not args.force_remove:
                        
                        print(f"\n\nHint: use the flag -fr or --force_remove to avoid be asked.")

                        while True:
                            
                            user_input = input(f"\n\nAre you sure that you want to remove the container [{args.container_name}]? (y/n): ").strip().lower()

                            if user_input in ["y", "yes"]:                          
                                print(f"{RED}WARNING!!: All running processes of the selected container will be terminated.{RESET}")
                                break
                            elif user_input in ["n", "no"]:
                                exit(1)
                            else:
                                print(f"{RED}\nInvalid input. Please use y(yes) or n(no).{RESET}")  
                else:
                    print(f"{RED}ERROR!!: [{args.container_name}] does not exist. {RESET}")
                    while True:
                        # ToDo: create a function ( the same 4 lines as below)
                        print(f"{GREEN}Only the following no-running containers of the current user can be removed:{RESET} ")
                        print_existing_container_list(no_running_containers)
                        #print(f"{RED}WARNING!!: All running processes of the selected container will be terminated and you will NOT ask again. Be sure about your decision.{RESET}")
                        selected_container_name, selected_container_position = select_container(no_running_containers)
                        print(f"{GREEN}Selected container to be removed:{RESET} {selected_container_name}")    
                        break                      
            else:    
                print(
                    "\n" +\
                    f"{GREEN}Info{RESET}: \
                    \nRemove an existing and not running machine learning container.  \
                    \n{GREEN}Correct Usage{RESET}: \
                    \nmlc remove <container_name>  \
                    \n{GREEN}Example{RESET}: \
                    \nmlc remove pt231aime\n"
                )    

                # ToDo: create a function ( the same 4 lines as below)
                print(f"{GREEN}The following not running containers of the current user can be removed:{RESET}")
                print_existing_container_list(no_running_containers)
                #print(f"{RED}WARNING!!: All running processes of the selected container will be terminated.{RESET}")
                selected_container_name, selected_container_position = select_container(no_running_containers)
                print(f"{GREEN}Selected container to be removed:{RESET} {selected_container_name}")  
            
            # Obtain container_tag from the selected container name
            selected_container_tag = no_running_container_tags[selected_container_position-1]
            print("Estoy aqui")
            '''
            # Get the image associated with the container
            docker_command_get_image = [
                "docker", 
                "container", 
                "ps", 
                "-a", 
                f"filter=name={selected_container_tag}",
                "--filter=label=aime.mlc",
                "--format {{{{.Image}}}}"
                ]
            
            out_message, error_mesage, exit_code = run_docker_command(docker_command_get_image)
            '''
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
            #exit_code = process.returncode
            
            #container_image = get_container_image(selected_container_tag)
            print(f"container_image: {container_image}")
            # Delete the container
            print("\nDeleting container ...")
            docker_command_delete_container = f"docker container rm {selected_container_tag}"
            subprocess.Popen(docker_command_delete_container, shell=True).wait()
            
            # Delete the container's image
            print("Deleting container image ...")
            docker_command_rm_image = f"docker image rm {container_image}"            
            subprocess.Popen(docker_command_rm_image, shell=True).wait()

            print(f"\n[{selected_container_name}] container removed.\n") 
            
            
        if args.command == 'start':
           
            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            
            if args.container_name:     
                if args.container_name not in available_user_containers:
                    print(f"No containers found matching the name '{args.container_name}' for the current user.")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"{GREEN}Available containers of the current user:{RESET}")
                    print_existing_container_list(available_user_containers)
                    selected_container_name, selected_container_position = select_container(available_user_containers)
                    print(f"{GREEN}Selected container to be started:{RESET} {selected_container_name}")                            
                else:
                    selected_container_name = args.container_name
                    selected_container_position = available_user_containers.index(args.container_name) + 1
                    #print(f'Provided container name [{args.container_name}] exists and will be started.')                   
            else:
                print(
                    "\n" +\
                    f"{GREEN}Info{RESET}: \
                    \nStart an existing machine learning container.  \
                    \n{GREEN}Correct Usage{RESET}: \
                    \nmlc start <container_name>  \
                    \n{GREEN}Example{RESET}: \
                    \nmlc start pt231aime\n"
                )

                # ToDo: create a function ( the same 4 lines as above)
                print(f"{GREEN}Available containers of the current user:{RESET}")
                print_existing_container_list(available_user_containers)
                selected_container_name, selected_container_position = select_container(available_user_containers)
                print(f"{GREEN}Selected container to be started:{RESET} {selected_container_name}")
            
            # Obtain container_tag from the selected container name
            selected_container_tag = available_user_container_tags[selected_container_position-1]
            #print(f"Container_tag: {selected_container_tag}")
            
            # Start the existing selected container:
            if selected_container_tag != check_container_running(selected_container_tag):
                print(f"[{selected_container_name}] starting container...")
                #docker_command = f"docker container start {selected_container_tag}"
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
                    #stderr=subprocess.PIPE,
                )
                stdout, _ = process.communicate()  # Communicate handles interactive input/output
                out = stdout.strip()
                exit_code = process.returncode
                #print(f"output: {out}")
                #print(f"selected_container_tag: {selected_container_tag}")
                #print(f"exit_code: {exit_code}")
                #_, _, _ = run_docker_command(docker_command)
                if out == selected_container_tag:
                    print(f"[{selected_container_name}] container started.\n\nTo open a shell to container use: mlc open {selected_container_name}")
                else:
                    print(f"{RED}ERROR!![{selected_container_name}] Error Starting Container.{RESET}")
            else:
                print(f"[{selected_container_name}] container already running")
                sys.exit(-1)


        if args.command == 'list':            
 
            show_container_info(args.all)
            '''
            # Adapt the filter to the selected flags (no flag or --all)
            filter_aime_mlc_user =  "label=aime.mlc" if args.all else f"label=aime.mlc={user_name}"
            #ERROR: docker_command_ls = f"docker container ls -a --filter{filter_aime_mlc_user} --format {{json .}}'"
            #print(f"filter_aime_mlc_user: {filter_aime_mlc_user}")
            docker_command_ls = [
                "docker", "container", 
                "ls", 
                "-a", 
                "--filter", filter_aime_mlc_user, 
                "--format", '{{json .}}'    
            ]
            #print(f"docker_command_ls: {docker_command_ls}")
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
                    #print(len(json_lines))
                    #print(f"type jsonlines[0]: {type(json_lines[0])}")
                    #print(f"jsonlines: {json_lines}")
                    containers_info = [json.loads(line) for line in json_lines if line]
                    #print(f"type containers_info[0]: {type(containers_info[0])}")
                    #print(f"containers_info: {containers_info}")
                    #print(len(containers_info))

                    # Apply formatting to all containers' info
                    output_lines = list(map(format_container_info, containers_info))
              
                except json.JSONDecodeError:
                    print("Failed to decode JSON output.")
                except KeyError as e:
                    print(f"Missing expected key: {e}")
            # Print the final processed output
            # Define an output format string with specified widths
            format_string = "{:<15}{:<30}{:<15}{:<30}{:<30}"
            print(f"\n{GREEN}Available ml-containers are:{RESET}\n")
            titles = ["CONTAINER", "SIZE", "USER", "FRAMEWORK", "STATUS"]
            print(format_string.format(*titles))
            #print("\t".join(titles))
            #print("\n".join(output_lines))
            # Print the container info
            print("\n".join(format_string.format(*info) for info in output_lines)+"\n")
            #print("")
            '''

        if args.command == 'stats':
            
            # Call the function with the correct mode (stream or no-stream)
            show_container_stats(args.stream)
            
            
            '''
            # Print a message to the console indicating the running ml-containers
            print("\nRunning ml-containers are:\n")

            # Run the 'docker container stats' command to get container name, CPU usage, and memory usage
            # The `--format` uses default placeholders for the full container name, CPU percentage, and memory usage
            docker_command = [
                "docker", "container", "stats",
                "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
                "--no-stream"
            ]

            # Use subprocess to execute the command and capture the output
            process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()

            # Check if the command returned an error
            if error:
                print(f"Error: {error.decode()}")
            else:
                # Convert the output from bytes to string
                output_str = output.decode()
                print(str(output_str))
                # Split the output by lines
                lines = output_str.splitlines()
                print(lines)
                # Initialize a list to hold the formatted output
                formatted_output = []

                # Process each line
                for line in lines:
                    # Split each line by tab to separate fields (container name, CPU %, memory usage)
                    parts = line.split('\t')

                    # If there are at least 3 parts (name, CPU %, memory usage), process the name
                    if len(parts) >= 3:
                        # Split the container name by "._." and take the first part
                        container_name = parts[0].split('._.')[0]

                        # Rebuild the line with the modified container name
                        formatted_output.append(f"[{container_name}]\t{parts[1]}\t{parts[2]}")

                # Join the formatted output back into a string
                output_str = "\n".join(formatted_output)

                # Replace "[78]" with "CONTAINER" in the output for consistency
                output_str = output_str.replace("[78]   ", "CONTAINER")

                # Print the formatted output
                print(output_str)

            # Print an empty line for better readability
            print("")
            '''
        if args.command == 'stop':

            # List existing containers of the current user
            available_user_containers, available_user_container_tags = existing_user_containers(user_name, args.command)
            containers_state = [True if container_tag == check_container_running(container_tag) else False for container_tag in available_user_container_tags]
            # Automate filtering
            running_containers, running_container_tags = filter_running_containers(
                containers_state, 
                True,
                available_user_containers, 
                available_user_container_tags
            )        

            # check that at least 1 container is running
            if not running_container_tags:
                print(f"{RED}WARNING!!: No containers are running and therefore there are no one to be stopped.{RESET}")
                exit(1)
            
            if args.container_name: 
                if args.container_name not in running_containers:
                    print("{RED}The provided container [{args.container_name}] is not running.{RESET}")
                    # ToDo: create a function ( the same 4 lines as below)
                    print(f"{GREEN}Running containers of the current user:{RESET}")
                    print_existing_container_list(running_containers)
                    print(f"{RED}WARNING!!: All running processes of the selected container will be terminated and you will NOT ask again. Be sure about your decision.{RESET}")
                    selected_container_name, selected_container_position = select_container(running_containers)
                    print(f"{GREEN}Selected container to be stopped:{RESET} {selected_container_name}")                            
                else:
                    selected_container_name = args.container_name
                    selected_container_position = running_containers.index(args.container_name) + 1
                    print(f'\n[{args.container_name}] is running and will be stopped.')
                    if not args.force_stop:
                        
                        print(f"\n\nHint: use the flag -fs or --force_stop to avoid be asked.")

                        while True:
                            
                            user_input = input(f"\n\nAre you sure that you want to stop the container [{args.container_name}]? (y/n): ").strip().lower()

                            if user_input in ["y", "yes"]:                          
                                print(f"{RED}WARNING!!: All running processes of the selected container will be terminated.{RESET}")
                                break
                            elif user_input in ["n", "no"]:
                                exit(1)
                            else:
                                print(f"{RED}\nInvalid input. Please use y(yes) or n(no).{RESET}")                                                    
            else:    
                print(
                    "\n" +\
                    f"{GREEN}Info{RESET}: \
                    \nStop an existing machine learning container.  \
                    \n{GREEN}Correct Usage{RESET}: \
                    \nmlc stop <container_name>  \
                    \n{GREEN}Example{RESET}: \
                    \nmlc stop pt231aime\n"
                )    

                # ToDo: create a function ( the same 4 lines as below)
                print(f"{GREEN}Running containers of the current user:{RESET}")
                print_existing_container_list(running_containers)
                print(f"{RED}WARNING!!: All running processes of the selected container will be terminated.{RESET}")
                selected_container_name, selected_container_position = select_container(running_containers)
                print(f"{GREEN}Selected container to be stopped:{RESET} {selected_container_name}")  
            
            # Obtain container_tag from the selected container name
            selected_container_tag = running_container_tags[selected_container_position-1]
            
            print(f"\n[{selected_container_name}] Stopping container ...")
            # Attempt to stop the container and store the result.
            docker_command_stop = f"docker container stop {selected_container_tag}"
            _, _, _ = run_docker_command(docker_command_stop)
            # Print a message indicating the container has been stopped.
            print(f"[{selected_container_name}] container stopped.\n")
      

    except KeyboardInterrupt:
        print(f"\n{RED}WARNING!!: Running process cancelled by the user{RESET}")
   
           
            
if __name__ == '__main__':
    main()

    
    
