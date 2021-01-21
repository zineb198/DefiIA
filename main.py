# coding: utf-8

import project

# Create Project Manager
projectManager = project.ProjectManager(execute_command=True)


# Start the instace
print("\n########## Start Instance ##########")
projectManager.instance_init()

# Update data
print("\n########## Create directories ##########")
projectManager.execute_code_ssh("'mkdir script'")

print("\n########## Send data and unzip it ##########")
projectManager.update_data("data.zip")

# Update last version of code
print("\n########## Send code ##########")
projectManager.update_code()

print("\n########## Install requirements ##########")
projectManager.execute_code_ssh(" 'pip install -r script/requirements.txt' ")

# Execute Job
print("\n########## Execute script ##########")
projectManager.execute_python_script("cleaning.py")
projectManager.execute_python_script("embedding.py")
projectManager.execute_python_script("classification.py")

# Collect job output
print("\n########## Get Result back ##########")
projectManager.collect_results()

## Finalize instance
print("\n########## Stop Instance ##########")
projectManager.instance_end()
