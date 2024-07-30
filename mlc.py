# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 



import sys
import os
import subprocess
import argparse

import csv
import re



# Set default values for script variables
supported_arch = "CUDA_ADA"     # Supported architecture for the container
mlc_version=3                   # Version number of the machine learning container setup


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
    #parser_list.add_argument('container_name', nargs = '?', type=str, help='List of the created ml-container/s')
    
    # Parser for the "open" command
    parser_open = subparsers.add_parser('open', help='Open an existing container')
    parser_open.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be opened')
    
    # Parser for the "remove" command
    parser_remove = subparsers.add_parser('remove', help='Remove a container')
    parser_remove.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be removed')
    
    # Parser for the "start" command
    parser_start = subparsers.add_parser('start', help='Start existing container/s')
    parser_start.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be started')

    # Parser for the "stats" command
    parser_stats = subparsers.add_parser('stats', help='Show the most important statistics of the running ml-containers')
    #parser_stats.add_argument('container_name', nargs = '?', type=str, help='Show the most important statistics of the running ml-containers')

    # Parser for the "stop" command
    parser_stop = subparsers.add_parser('stop', help='Stop existing container/s')
    parser_stop.add_argument('container_name', nargs = '?', type=str, help='Name of the container/s to be stopped')

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
        print(f"{GREEN}Available commands:{RESET} " + ', '.join(available_commands))
        chosen_command = input("Please choose a command: ").strip()
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

def get_container_name(container_name):
    """
    
    Args:0
    
    Return:
    
    """
    if container_name is not None:
        return validate_container_name(container_name)
    else:
        while True:
            container_name = input(f"{CYAN}Enter a container name: {RESET}")
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
                print(f"Please enter a number between 1 and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_versions(versions):
    """
    
    Args:
    
    Return:
    
    """      
    available_versions = [version[0] for version in versions]
    
    return available_versions

def display_versions(framework, versions):
    """
    
    Args:
    
    Return:
    
    """    
    print(f"\n{CYAN}Available versions for {framework}:{RESET}")
    for i, (version, _) in enumerate(versions, start=1):
        print(f"{i}) {version}")


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
    

###############################################################################################################
    

def main():
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
    
    # Obtain user and group id for different tasks by create, open,...
    user_id = os.getuid()
    group_id = os.getgid()        
    
    if args.command == 'create':
   
        if args.container_name is None and args.framework is None and args.version is None:
            print(
                "\n" +\
                f"{GREEN}Info{RESET}: \
                \nCreate a new machine learning container. \
                \n{GREEN}Correct Usage{RESET}: \
                \n<container_name> <framework_name> <framework_version> -w /home/$USER/workspace -d /data -ng 1 \
                \n{GREEN}Example{RESET}: \
                \npt231aime Pytorch 2.3.1-aime\n"
            )

        # User provides the container name
        validated_container_name = get_container_name(args.container_name)
        
        # ToDo (check if the container_name exist due to an existing container, no check container_tag but container_name):
        #while validated_container_name exist:
        #    user_container_name = input("The provided container name alread exist: Please, introduce a new container name:")
        
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
            available_versions = get_versions(versions_images)
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
         
        print(f"\nYou have selected: {ORANGE2}{selected_framework} {selected_version}{RESET}")
        print(f"The mlc container called {ORANGE2}{validated_container_name}{RESET} with the framework {ORANGE2}{selected_framework}{RESET} and version {ORANGE2}{selected_version}{RESET} with docker image {selected_docker_image} will be created!!")
        
        # Check if the provided workspace directory exist:
        if os.path.isdir(args.workspace_dir):
            print(f"\n/workspace will mount on '{args.workspace_dir}' - OK")
        else:
            print(f"\nAborted.\n\nWorkspace directory not existing: {args.workspace_dir}")
            print("\nThe workspace directory will be mounted as /workspace in the container.")
            print("It is the directory where your project data should be stored to be accessed\ninside the container.")
            print("\nIt can be set to an existing directory with the option '-w=/your/workspace'\n")
            sys.exit(-4)
        
        # Check if the provided data directory exist:
        if args.data_dir:
            data_mount = os.path.realpath(args.data_dir)
            if os.path.isdir(data_mount):
                print(f"/data will mount on '{data_mount}' - OK")
            else:
                print(f"\nAborted.\n\nData directory not existing: {data_mount}")
                print("\nThe data directory will be mounted as /data in the container.")
                print("It is the directory where data sets for example mounted from\nnetwork volumes can be accessed inside the container.")
                sys.exit(-4)
        
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
        subprocess.run(['docker', 'pull', selected_docker_image])
    
        print("\nSetting up container ... \n")
        
        # Adding 
        container_label = "aime.mlc"
        workspace = "/workspace"
        data = "/data"
        
        # Run the Docker Container: Starts a new container, installs necessary packages, 
        # and sets up the environment.
        #print("docker_run_cmd will be created")
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
            f'addgroup --gid {group_id} {os.getlogin()} > /dev/null; '
            f'adduser --uid {user_id} --gid {group_id} {os.getlogin()} --disabled-password --gecos aime > /dev/null; '
            f'passwd -d {os.getlogin()}; echo "{os.getlogin()} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/{os.getlogin()}_no_password; '
            f'chmod 440 /etc/sudoers.d/{os.getlogin()}_no_password; exit'
        ]
        #print("subprocess docker_run_cmd will starten")        
        result_run_cmd = subprocess.run(docker_run_cmd, capture_output=True, text=True )
        #print(f"STDOUT run_cmd: {result_run_cmd.stdout}")
        #print(f"STDERR run_cmd: {result_run_cmd.stderr}")
        
        #print("subprocess docker_run_cmd finished")

        # Commit the Container: Saves the current state of the container as a new image.
        result_commit = subprocess.run(['docker', 'commit', container_tag, f'{selected_docker_image}:{container_tag}'], capture_output=True, text=True)#, stdout = subprocess.DEVNULL, stderr =subprocess.DEVNULL)
        #print(f"STDOUT commit: {result_commit.stdout}")
        #print(f"STDERR commit: {result_commit.stderr}")
        
        # ToDo: capture possible errors and treat them  
        #print("subprocess commit finished")

        # Remove the Container: Cleans up the initial container to free up resources.
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
        'docker', 'create', '-it', '-w', workspace, '--name', container_tag,
        '--label', f'{container_label}={os.getlogin()}', '--label', f'{container_label}.NAME={validated_container_name}',
        '--label', f'{container_label}.USER={os.getlogin()}', '--label', f'{container_label}.VERSION={mlc_version}',
        '--label', f'{container_label}.WORK_MOUNT={args.workspace_dir}', '--label', f'{container_label}.DATA_MOUNT={args.data_dir}',
        '--label', f'{container_label}.FRAMEWORK={selected_framework}-{selected_version}', '--label', f'{container_label}.GPUS={args.num_gpus}',
        '--user', f'{user_id}:{group_id}', '--tty', '--privileged', '--interactive', '--gpus', args.num_gpus,
        '--network', 'host', '--device', '/dev/video0', '--device', '/dev/snd', '--ipc', 'host',
        '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864', '-v', '/tmp/.X11-unix:/tmp/.X11-unix', 
        '--group-add', 'video', '--group-add', 'sudo', f'{selected_docker_image}:{container_tag}', 'bash', '-c',
        f'echo "export PS1=\'[{validated_container_name}] `whoami`@`hostname`:${{PWD#*}}$ \'" >> ~/.bashrc; bash'
        ]
        # Insert the volumes list at the correct position, after '-it'
        docker_create_cmd[3:3] = volumes
        
    #print(f"{docker_create_cmd}")
    #print("docker_create_cmd finished")

    result_create_cmd = subprocess.run(docker_create_cmd, capture_output= True, text=True)
    #print(f"STDOUT create_cmd: {result_create_cmd.stdout}")
    #print(f"STDERR create_cmd: {result_create_cmd.stderr}")
    
    print(f"\n[{validated_container_name}] ready. Open the container with: mlc open {validated_container_name}\n")
       
        
    if args.command == 'open':
        if not args.container_name:
            print(
                "\n" +\
                f"{GREEN}Info{RESET}: \
                \Open a created machine learning container. \
                \n{GREEN}Correct Usage{RESET}: \
                \nmlc open <container_name>\
                \n{GREEN}Example{RESET}: \
                \npt231aime\n"
            )
            args.container_name = input("Please provide a container name: ").strip()            
   
           
            
if __name__ == '__main__':
    main()

    
    
