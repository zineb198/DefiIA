# coding: utf-8

import project

# Create Project Manager
projectManager = project.ProjectManager(execute_command=True)


# Start the instace
print("\n########## Start Instance ##########")
projectManager.instance_init()


# Update data
print("\n########## Create directories ##########")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_folder + "'")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_data_cleaned + "'")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_results + "'")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_model + "'")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_model + '/word2vec' + "'")
#projectManager.execute_code_ssh("'mkdir " + projectManager.remote_code + "'")



print("\n########## Send data and unzip it ##########")
#projectManager.update_data("data.zip")


# Update last version of code
print("\n########## Send code ##########")
#projectManager.update_code()


# Execute Job
print("\n########## Execute script ##########")
#projectManager.execute_python_script("cleaning.py")
#projectManager.execute_python_script("embedding.py")
projectManager.execute_python_script("classification.py")


# Collect job output
print("\n########## Get Result back ##########")
projectManager.collect_results()


## Finalize instance
print("\n########## Stop Instance ##########")
projectManager.instance_end()
