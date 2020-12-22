# coding: utf-8

import instances
import os
from yaml import load


class ProjectManager:

    def __init__(self, print_command=True, execute_command=False):

        self.instance = instances.InstancesManager(print_command=print_command, execute_command=execute_command)


        self.local_folder = '/Users/cecile/Documents/INSA/DefiIA/'  #TODO
        self.local_data = self.local_folder + "/data"
        self.local_code = self.local_folder + "/script"
        self.local_results = self.local_folder + "/results"

        self.remote_folder = "/home/cecile"  #TODO
        self.remote_data = self.remote_folder + "/data"
        self.remote_data_cleaned = self.remote_data + "/cleaned"
        self.remote_code = self.remote_folder + "/script"
        self.remote_model = self.remote_folder + "/models"
        self.remote_results = self.remote_folder + "/results"

        self.container_folder = "/tmp/TP"  #not use here
        self.container_data = self.container_folder + "/data"
        self.container_code = self.container_folder + "/script"
        self.container_model = self.container_folder + "/models"
        self.container_results = self.container_folder + "/results"

    def set_image_container_names(self, image_name, container_name):
        self.image_name = image_name
        self.container_name = container_name

    def instance_init(self):
        """

        :return:
        """

        self.instance.instance_init()

    def update_data(self, zip_file, container=None):
        """

        :return:
        """

        self.instance.scp(direction='up', src_folder=self.local_folder+"/"+zip_file, dst_folder=self.remote_folder,
                          recurse=False, python_filter=False)

        if container is not None:
            command = "'unzip -o " + self.container_folder + "/" + zip_file + " -d " + self.container_folder + "'"
            self.execute_code_ssh_container(command)
        else:
            command = "'unzip -o " + self.remote_folder + "/" + zip_file + " -d " + self.remote_folder + " | pv -l >/dev/null'"
            self.instance.ssh_command(command)

    def update_code(self):
        """

        :return:
        """

        self.instance.scp(direction='up',
                          src_folder=self.local_code,
                          dst_folder=self.remote_code,
                          recurse=False,
                          python_filter=True)

    def execute_code_ssh(self, command):
        """

        :return:
        """

        ssh_command = command

        self.instance.ssh_command(ssh_command)

    def execute_code_ssh_container(self, command):
        """

        :return:
        """

        ssh_command = command

        self.instance.ssh_command_container(ssh_command, self.container_name)

    def execute_python_script(self, script_name, args=None):
        """

        :return:
        """

        command = "'python " + self.remote_code + "/" + script_name #/opt/conda/bin/

                  #+ " --data_dir " \
                  #+ self.remote_data + " --results_dir " + self.remote_results + " --model_dir " + self.remote_model

        if not(args is None):
            for k,v in args:
                command +=  " --" + k + " " + v
        command += "'"
        ssh_command = command

        self.instance.ssh_command(ssh_command)


    def execute_python_script_container(self, script_name, args=None):
        """

        :return:
        """

        command = "'python " + self.container_code + "/" + script_name + " --data_dir " \
                  + self.container_data + " --results_dir " + self.container_results + " --model_dir " + self.container_model

        if not(args is None):
            for k,v in args:
                command +=  " --" + k + " " + v
        command += "'"
        ssh_command = command


        self.instance.ssh_command_container(ssh_command, self.container_name)

    def manage_container(self, action):
        """

        :return:
        """
        if action =="run":
            command = "'sudo docker run -t -d --gpus all --name "+self.container_name+" -v " + self.remote_folder +":"+self.container_folder +" " + self.image_name +"'"
        elif action == "stop":
            command = "'sudo docker stop "+ self.container_name +"'"
        elif action == "remove":
            command = "'sudo docker rm "+ self.container_name +"'"

        else:
            raise ValueError("'action' parameter should be 'run', 'stop' or 'remove' ")

        ssh_command = command

        self.instance.ssh_command(ssh_command)

    def collect_results(self):
        """

        :return:
        """
        self.instance.scp(direction='down',
                          src_folder=self.remote_results,
                          dst_folder=self.local_results,
                          python_filter=False)

        #self.instance.scp(direction='down',
        #                  src_folder=self.remote_results,
        #                  dst_folder=self.local_results,
        #                  recurse=True,
        #                  python_filter=False,
        #                  pickle_filter=True,
        #                  csv_filter=False)

        #self.instance.scp(direction='down',
        #                  src_folder=self.remote_results,
        #                  dst_folder=self.local_results,
        #                  recurse=True,
        #                  python_filter=False,
        #                  pickle_filter=False,
        #                  csv_filter=True)

    def instance_end(self):
        """

        :return:
        """

        self.instance.instance_final()
