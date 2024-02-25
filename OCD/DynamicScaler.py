import time, math
from kubernetes import client, config
import pandas as pd
import numpy as np
import requests
import os
import aiohttp
import asyncio
import re


# Initialize the Kubernetes API client
config.load_kube_config()
kube_client = client.AppsV1Api()
autoscaling_client = client.AutoscalingV1Api()
v1 = client.CoreV1Api()
MODEL_EXEC_TIME = {
    "BERT": 0.127,
    "GPT": 0.3,
    "LLAMA": 0.035,
    "WHISPER": 0.073,
    "WIDERESNET": 1.2
}
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Pipeline:
    def __init__(self, namespace, kube_client, v1_client):
        self.namespace = namespace
        self.kube_client = kube_client
        self.v1_client = v1_client
        self.pipelines = self.update_pipeline_stage_pods()
        self.model_configs = pd.read_csv("model_config.csv",names=["model","stages","inference_time"], header= 0)
        self.pipeline_tolerance = 3
        self.pipeline_change_record = pd.DataFrame(columns=["pipeline","last_change_time"])
        self.pipeline_change_interval = 2*60
        self.openfaas_operator = OpenfaasOperator(self.namespace)
        self.proxy_service = "http://172.169.8.253:31116/proxy"

    def update_pipeline_stage_pods(self):
        # get all pod in namespace, get model name (pod_name.split('-') 0,1 is model name)
        pods = v1.list_namespaced_pod(namespace=self.namespace)
        pipeline_record = pd.DataFrame(columns=["model","pipeline","stages","pod_name","pod_ip","deployment_name"])
        for pod in pods.items:
            # pipeline and stages from labels
            stages = pod.metadata.labels['stages']
            # pipeline = pod.metadata.labels['pipeline']
            pipeline = f"{stages}-stages"
            model = "-".join(pod.metadata.name.split('-')[0:2])
            pod_name = pod.metadata.name
            pod_ip = pod.status.pod_ip
            deployment_name = pod.metadata.labels['faas_function']
            pipeline_record = pd.concat([pipeline_record, pd.DataFrame({"model":model,"pipeline":pipeline,"stages":stages,"pod_name":pod_name,"pod_ip":pod_ip,"deployment_name":deployment_name},index=[0])])
        return pipeline_record

    
    def patch_deployments(self, deployment):
        if deployment.spec.replicas < 1:
            self.openfaas_operator.delete_function(deployment.metadata.name)
            print(f"Delete function {deployment.metadata.name} in patch_deployments, deployment.spec.replicas < 1")
            return
        maxtries = 3
        for _ in range(maxtries):
            try:
                self.kube_client.patch_namespaced_deployment(name=deployment.metadata.name, namespace=self.namespace, body=deployment)
                print(f"Deployment '{deployment.metadata.name}' updated successfully.")
                break
            except Exception as e:
               
                print("Conflict detected, retrying... in patch_deployments")
                time.sleep(1)

    def create_new_pod_of_deployment(self, model, deployment_name, to_stages):
        deployment = self.kube_client.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
        deployment.spec.replicas += 1
        model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // to_stages
        # container env update model_nums
        containers = []
        for container in deployment.spec.template.spec.containers:
            env_vars = {env.name: env for env in container.env}
            env_vars['model_nums'] = client.V1EnvVar(name='model_nums', value=str(model_nums))
            env_vars['pipeline'] = client.V1EnvVar(name='pipeline', value=f"{to_stages}-stages")
            env_vars['stages'] = client.V1EnvVar(name='stages', value=str(to_stages))
            container.env = list(env_vars.values())
            containers.append(container)
        deployment.spec.template.spec.containers = containers
        self.patch_deployments(deployment)
        # wait for new pod created
        time.sleep(1)
        self.update_pipeline_stage_pods()

    # if stage = 4, model_max_stage = 32, then only 0,4,8,12,16,20,24,28 will keep
    def calculate_stages_to_pod(self, model, stage):
        stages = ["submod-0"]
        no_in_stages = []
        stage = int(stage)
        model_max_stage = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max()
        model_nums = model_max_stage // (stage)        
        for i in range(1, model_max_stage):
            if i%(model_nums) == 0:
                stages.append(f'submod-{i}')
            else:
                no_in_stages.append(f'submod-{i}')
        return stages, no_in_stages
    
    def delete_function_notin_pipeline(self, model, pipeline, pod_ip_should_be_deleted,pod_name_need_to_delete):
        pipeline_record = self.pipelines.loc[(self.pipelines['model'] == model) & (self.pipelines['pipeline'] == pipeline)]
        for pod_name in pod_name_need_to_delete:
            deployment_name = pipeline_record.loc[pipeline_record['pod_name'] == pod_name]['deployment_name'].values
            if len(deployment_name) < 1:
                continue
            deployment_name = deployment_name[0]
            deployment = self.kube_client.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
            if deployment.spec.replicas == 1:
                self.openfaas_operator.delete_function(deployment_name)
            else:
                deployment.spec.replicas -= 1
                self.kube_client.patch_namespaced_deployment(name=deployment_name, namespace=self.namespace, body=deployment)

    def check_pipeline_compelete(self, model, stages):
        self.pipelines = self.update_pipeline_stage_pods()
        pipeline = f"{stages}-stages"
        stages_should_exist, no_in_stages = self.calculate_stages_to_pod(model, stages)
        pipeline_record = self.pipelines.loc[(self.pipelines['model'] == model) & (self.pipelines['pipeline'] == pipeline)]
        function_name_suffix = "latency-64"
        pod_name_prefix = [f"{model}-{i}-{function_name_suffix}" for i in stages_should_exist]
        deployment_name_exist = pipeline_record.loc[pipeline_record['deployment_name'].str.startswith(tuple(pod_name_prefix))]['deployment_name'].tolist()
        # find which deployment not exist
        function_lost = []
        if len(deployment_name_exist) != len(stages_should_exist):
            deployment_name_should_exist = [f"{model}-{i}-{function_name_suffix}" for i in stages_should_exist]
            deployment_name_not_exist = list(set(deployment_name_should_exist) - set(deployment_name_exist))
            function_lost = deployment_name_not_exist

        if len(function_lost) > 0:
            for function_name in function_lost:
                model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // stages
                self.openfaas_operator.creat_function(function_name, f"--env pipeline={pipeline} --env stages={stages} --env model_nums={model_nums}")
            # update pipeline
            self.update_pipeline_stage_pods()
            return 1, "recreating"
        
        return 0, "complete"
    def sure_pipeline_complete(self, model, stages):
        while True:
            code, status = self.check_pipeline_compelete(model, stages)
            if code == 0:
                print(f"Pipeline {stages}-stages of {model} complete")
                return
            else:
                print(f"Wait for pipeline complete....")
                time.sleep(3)
                continue

    # return 0, None: pod all ready; 1: function lost, Lost_functions; 2: pod not ready, not_ready_functions
    def check_pod_in_pipeline_ready(self, model,  stages):
        self.pipelines = self.update_pipeline_stage_pods()
        pipeline = f"{stages}-stages"
        stages_should_exist, no_in_stages = self.calculate_stages_to_pod(model, stages)
        pipeline_record = self.pipelines.loc[(self.pipelines['model'] == model) & (self.pipelines['pipeline'] == pipeline)]
        if pipeline_record.empty:
            return 3, None
        function_name_suffix = '-'.join(pipeline_record["deployment_name"].values[0].split('-')[-2:])
        pod_name_prefix = [f"{model}-{i}" for i in stages_should_exist]
        deployment_name_exist = pipeline_record.loc[pipeline_record['deployment_name'].str.startswith(tuple(pod_name_prefix))]['deployment_name'].tolist()
        deployment_name_exist = list(set(deployment_name_exist))
        # find which deployment not exist
        function_lost = []
        if len(deployment_name_exist) < len(stages_should_exist):
            deployment_name_should_exist = [f"{model}-{i}-{function_name_suffix}" for i in stages_should_exist]
            deployment_name_not_exist = list(set(deployment_name_should_exist) - set(deployment_name_exist))
            function_lost = deployment_name_not_exist
            model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // stages
            self.openfaas_operator.create_list_functions(function_lost, f"--env pipeline={pipeline} --env stages={stages} --env model_nums={model_nums}")
            return 1, function_lost

        else:
            # check pod ready status using kubernetes.
            not_ready_functions = []
            for deployment_name in deployment_name_exist:
                deployment = self.kube_client.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
                if deployment.status.ready_replicas != deployment.spec.replicas:
                    not_ready_functions.append(deployment_name)
                # if deployment containers ready == false
                for container in deployment.status.conditions:
                    if container.type == "Ready" and container.status == "False":
                        not_ready_functions.append(deployment_name)                    
            if len(not_ready_functions) > 0:
                # if status is CrashLoopBackOff, delete pod
                # pods = list deployment pods 
                pods = self.v1_client.list_namespaced_pod(namespace=self.namespace)
                for pod in pods.items:
                    if pod.metadata.name.startswith(tuple(not_ready_functions)):
                        # if pod.status is pending 
                        if pod.status.phase == "Pending":
                            continue
                        reason = pod.status.container_statuses[0].state.waiting.reason if pod.status.container_statuses[0].state.waiting else None
                        if reason == "CrashLoopBackOff":
                            self.delete_pod_by_name(pod.metadata.name)
                return 2, not_ready_functions
            return 0, None
    

    def check_pod_is_ready(self, pod_names):
        # check pod ready status using kubernetes.
        not_ready_functions = []
        for pod_name in pod_names:
            pod = self.v1_client.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            if pod.status.phase != "Running":
                not_ready_functions.append(pod_name)
        if len(not_ready_functions) > 0:
            return False
        return True
    
        
    def delete_pod_by_name(self, pod_name):
        try:
            self.v1_client.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
            print(f"Pod '{pod_name}' deleted successfully.")
        except Exception as e:
            print("Conflict detected, retrying... in delete_pod_by_name")
            time.sleep(1)


    # change pipeline stages
    # model = opt-66b, from_pipeline 64, to_pipeline 16, i.e. model_nums = 4
    async def change_pipeline(self, model, from_pipeline, to_stages, number_of_pipeline_to_change=0):
        # if pipeline change intervale less than self.pipeline_change_interval, do nothing
        self.pipelines = self.update_pipeline_stage_pods()
        from_pipeline = f"{from_pipeline}-stages"
        to_stages = int(to_stages)
        if not self.pipeline_change_record.loc[self.pipeline_change_record['pipeline'] == from_pipeline].empty:
            last_change_time = self.pipeline_change_record.loc[self.pipeline_change_record['pipeline'] == from_pipeline]['last_change_time']
            if time.time() - last_change_time < self.pipeline_change_interval:
                return
        else:
            if len(self.pipeline_change_record) < 1:
                self.pipeline_change_record = pd.DataFrame({"pipeline":from_pipeline,"last_change_time":time.time()}, index=[0])
            else:
                self.pipeline_change_record = pd.concat([self.pipeline_change_record, pd.DataFrame({"pipeline":from_pipeline,"last_change_time":time.time()},index=[0])])
        # if pipeline stage decrease, merge function then delete function.
        # if from_pipeline is not ready, wait for it to be ready
        
        
        pipeline_record = self.pipelines.loc[(self.pipelines['model'] == model) & (self.pipelines['pipeline'] == from_pipeline)]
        # pipeline_replicas  = len of pod_name contains submod-0
        pipeline_replicas = len([pod_name for pod_name in pipeline_record['pod_name'].tolist() if '-0-' in pod_name])
        if pipeline_record.empty:
            return
        if number_of_pipeline_to_change == 0 or number_of_pipeline_to_change > pipeline_replicas:
            # means change all pipeline
            pipeline_record = pipeline_record
        else:
            # each deployment random choose number_of_pipeline_to_change pods
            deployment_names = pipeline_record['deployment_name'].tolist()
            deployment_names = list(set(deployment_names))
            pod_names_to_change = []
            for deployment_name in deployment_names:
                deployment_record = pipeline_record.loc[pipeline_record['deployment_name'] == deployment_name]
                pod_names_to_change.extend(deployment_record.sample(n=number_of_pipeline_to_change)['pod_name'].tolist())
            pipeline_record = pipeline_record.loc[pipeline_record['pod_name'].isin(pod_names_to_change)]

        
        # submod-0
        pod_need_to_change_ids, pod_ip_should_be_deleted = self.calculate_stages_to_pod(model, to_stages)
        # opt-66b-submod-0
        pod_need_to_change_suffix = [f"{model}-{i}-" for i in pod_need_to_change_ids]
        pod_ip_need_to_change = pipeline_record.loc[pipeline_record['pod_name'].str.startswith(tuple(pod_need_to_change_suffix))]['pod_ip'].tolist()
        # exclude ip is None
        pod_ip_need_to_change = [pod_ip for pod_ip in pod_ip_need_to_change if pod_ip is not None]
        pod_name_need_to_delete = pipeline_record.loc[~pipeline_record['pod_name'].str.startswith(tuple(pod_need_to_change_suffix))]['pod_name'].tolist()



        # while True:
        #     # check pipeline ready
        #     status, not_ready_functions = self.check_pod_in_pipeline_ready(model, from_pipeline, to_stages)
        #     if status == 0:
        #         break
        #     elif status == 1:
        #         # create new function
        #         model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // to_stages
        #         for function_name in not_ready_functions:
        #             self.openfaas_operator.creat_function(function_name, f"--env pipeline={from_pipeline} --env stages={to_stages} --env model_nums={model_nums}")
        #             deployment = self.kube_client.read_namespaced_deployment(name=function_name, namespace=self.namespace)
        #             deployment.spec.replicas = 1
        #             self.patch_deployments(deployment)
        #         print(f"Create new function {not_ready_functions}")
        #     elif status == 2:
        #         # wait for pod ready
        #         print(f"Wait for pod ready {not_ready_functions}")
        #         await asyncio.sleep(5)
        #         continue
        #     elif status == 3:
        #         return




        for pod_ip in pod_ip_need_to_change:
            res, code = await self.call_function_merge_pipeline(model, pod_ip, to_stages)
            print(f"Change pipeline of {model} to {to_stages} stages {pod_ip} {code} {res}")
            if code != 200:
                function_name = self.get_deployment_from_ip(pod_ip, pipeline_record)
                self.create_new_pod_of_deployment(model, function_name, to_stages)

                pod_ip_should_be_deleted.append(pod_ip)
                pod_name_need_to_delete.append(self.get_pod_name_from_ip(pod_ip, pipeline_record))
                pod_ip_need_to_change.remove(pod_ip)
                print(f"Create new pod of {function_name} {to_stages} stages")
        self.patch_pod_env_in_pod(model, from_pipeline, to_stages, pod_ip_need_to_change,pipeline_record)
        
        self.sure_pipeline_complete(model, to_stages)

        while True:
            status, not_ready_functions = self.check_pod_in_pipeline_ready(model, to_stages)
            if status == 0:
                break
            else:
                print(f"Wait for pipeline ready....")
                await asyncio.sleep(5)
                continue
        print(f"new Pipeline {to_stages}-stages ready, delete old pipeline {from_pipeline}")
        self.delete_function_notin_pipeline(model, from_pipeline, pod_ip_should_be_deleted,pod_name_need_to_delete)

        # update env
        self.update_pipeline_stage_pods()

        # make sure all pipeline completed.
        current_stages = list(set(self.pipelines.loc[self.pipelines['model'] == model]['stages'].tolist()))
        for stages in current_stages:
            stages = int(stages)
            self.sure_pipeline_complete(model, stages)



    # DEPRECATED
    def create_pipeline(self, model, stages, pod_ip_need_to_change):
        # create new pipeline, stages
        for pod_ip in pod_ip_need_to_change:
            function_name   = self.get_deployment_from_ip(pod_ip, self.pipelines)
            if not function_name:
                continue
            model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // stages
            self.openfaas_operator.creat_function(function_name, f"--env pipeline={stages}-stages --env stages={stages} --env model_nums={model_nums}")
            
        # wait for function created
        time.sleep(5)
        self.update_pipeline_stage_pods()

    #dont use!
    def patch_pod_env_in_deployment(self, model, from_pipeline, to_stages, pod_ip_need_to_change):
        to_pipeline = f"{to_stages}-stages"
        for pod_ip in pod_ip_need_to_change:
            deployment_name = self.get_deployment_from_ip(pod_ip)

            # Pause Deployment Rollouts
            

            max_retries = 3
            for _ in range(max_retries):
                try:
                    deployment = self.kube_client.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
                    all_containers = []
                    for container in deployment.spec.template.spec.containers:
                        env_vars = {env.name: env for env in container.env}
                        # if env_vars.get('pipeline') and env_vars['pipeline'].value == from_pipeline:
                        env_vars['stages'] = client.V1EnvVar(name='stages', value=str(to_stages))
                        env_vars['pipeline'] = client.V1EnvVar(name='pipeline', value=f"{to_pipeline}")
                        model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // to_stages
                        env_vars['model_nums'] = client.V1EnvVar(name='model_nums', value=str(model_nums))
                        container.env = list(env_vars.values())
                        
                        all_containers.append(container)
                    deployment.spec.template.spec.containers = all_containers
                    # Patch the deployment
                    self.kube_client.patch_namespaced_deployment(name=deployment_name, namespace=self.namespace, body=deployment)
                    print(f"Deployment '{deployment_name}' updated successfully.")
                    break
                except Exception as e:

                    print("Conflict detected, retrying... in patch_pod_env_in_deployment")
                    time.sleep(1)


    def patch_pod_env_in_pod(self, model, from_pipeline, to_stages, pod_ip_need_to_change, pipeline_record):
        to_pipeline = f"{to_stages}-stages"
        for pod_ip in pod_ip_need_to_change:
            pod_name = self.get_pod_name_from_ip(pod_ip, pipeline_record)
            # Pause Deployment Rollouts
            max_retries = 6
            for _ in range(max_retries):
                try:
                    pod = self.v1_client.read_namespaced_pod(name=pod_name, namespace=self.namespace)
                    # patch pipeline and stages in labels
                    pod.metadata.labels['pipeline'] = to_pipeline
                    pod.metadata.labels['stages'] = str(to_stages)
                    # Patch the pod
                    self.v1_client.patch_namespaced_pod(name=pod_name, namespace=self.namespace, body=pod)
                    print(f"Pod '{pod_name}' updated successfully.")
                    break
                except client.exceptions.ApiException as e:

                    print("Conflict detected, retrying... in patch_pod_env_in_pod")
                    time.sleep(1)


    def label_all_pod_in_namespace(self):
        pods = self.v1_client.list_namespaced_pod(namespace=self.namespace)
        for pod in pods.items:
            pod_name = pod.metadata.name
            pod.metadata.labels['pipeline'] = "32-stages"
            pod.metadata.labels['stages'] = "32"
            self.v1_client.patch_namespaced_pod(name=pod_name, namespace=self.namespace, body=pod)
            print(f"Pod '{pod_name}' updated successfully.")

    
    def get_deployment_from_ip(self, pod_ip, pipeline_records):
        deployment_name = pipeline_records.loc[pipeline_records['pod_ip'] == pod_ip]['deployment_name'].values
        return deployment_name[0] if len(deployment_name) > 0 else None

    def get_pod_name_from_ip(self, pod_ip, pipline_record):
        return pipline_record.loc[pipline_record['pod_ip'] == pod_ip]['pod_name'].values[0]

    async def call_function_merge_pipeline(self, model, pod_ip, stages):
        model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // stages
        post_data = {
            "pod_ip": pod_ip,
            "port": 5000,
            "url": f"merge/{model_nums}"
        }
        url = f"{self.proxy_service}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=post_data) as response:
                content = await response.read()
                return content, response.status
    
    def call_function_change_pipeline(self, model, pod_ip, stages):
        model_nums = self.model_configs.loc[self.model_configs['model'] == model]['stages'].max() // stages
        post_data = {
            "pod_ip": pod_ip,
            "port": 5000,
            "url": f"change/{model_nums}"
        }
        url = f"{self.proxy_service}"
        response = requests.post(url, json=post_data)
        return response.content, response.status_code

    # return new pipelines with new stages
    def calculate_stage_from_cv(self, model, cv):
        # get current pipelines and stages of model
        model_stages = self.model_configs.loc[self.model_configs['model'] == model]['stages']
        min_stage = model_stages.min()
        max_stage = model_stages.max()

        model_pipeline = self.pipelines.loc[self.pipelines['model'] == model]
        current_stages = model_pipeline['stages']
        # cv values range from 0 to 8, calculate new stages using cv, bigger cv, more stages. using easeoutback function
        cv = cv / 16
        desired_stages = self.easeOutBack(cv, min_stage, max_stage)
        # if gap between desired and current is less than Pipeline tolerance, do nothing
        std_dev = np.std([abs(stage - desired_stages) for stage in current_stages])
        if std_dev > self.pipeline_tolerance:
            # change the biggest gap pipeline to desired stage
            biggest_gap = 0
            pipeline_to_change = None
            for index, row in model_pipeline.iterrows():
                gap = abs(desired_stages - row['stages'])
                if gap > biggest_gap:
                    biggest_gap = gap
                    pipeline_to_change = row['pipeline']
            return pipeline_to_change, desired_stages
        return None, None
    
    def easeOutBack(t, b=2, c=18):
        return c * ((math.sin(t * (2 - t) * math.pi / 2)) ** 1) + b
    

class OpenfaasOperator:
    def __init__(self, namespace):
        # self.benchmark_address = '/home/yanying/workspace/dybranch/benchmark_merge/GPT-LATENCY-64_merge'
        self.benchmark_address = '/home/pengshijie/dybranch/benchmark_merge/GPT-LATENCY-64_merge'
        self.gateway = 'http://172.169.8.253:31112'
        self.namespace = namespace
        self.kube_client = client.AppsV1Api()
        self.v1_client = client.CoreV1Api()

    def patch_deployments(self, deployment):
        maxtries = 3
        for _ in range(maxtries):
            try:
                self.kube_client.patch_namespaced_deployment(name=deployment.metadata.name, namespace=self.namespace, body=deployment)
                print(f"Deployment '{deployment.metadata.name}' updated successfully.")
                break
            except Exception as e:
               
                print("Conflict detected, retrying...in patch_deployments")
                time.sleep(1)

    def creat_function(self, function_name, config = ""):
        # if deployment exist and replicas ==0, scale up using configs 
        try:
            deployment = kube_client.read_namespaced_deployment(name=function_name, namespace=self.namespace)
            if deployment.spec.replicas == 0:
                print(f"delete function {function_name}, since replicas == 0")
                self.delete_function(function_name)
                time.sleep(5)
            else:
                print(f"Function {function_name} exist, scale up")
                deployment.spec.replicas += 1
                containers = []
                configs = config.split("--env")
                env_config = {}
                for c in configs:
                    if c == "":
                        continue
                    key, value = c.split("=")
                    env_config[key.strip()] = value.strip()
                for container in deployment.spec.template.spec.containers:
                    env_vars = {env.name: env for env in container.env}
                    for key, value in env_config.items():
                        env_vars[key] = client.V1EnvVar(name=key, value=value)
                    container.env = list(env_vars.values())
                    containers.append(container)
                deployment.spec.template.spec.containers = containers
                self.patch_deployments(deployment)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                print(f"Function {function_name} not exist, create new function")
                cmd = f"cd {self.benchmark_address} && faas-cli deploy -f config.yml --gateway={self.gateway} --filter {function_name} {config} -n {self.namespace}"
                os.system(cmd)
            else:
                raise e
        except Exception as e:
            print(f"Error in creat_function {function_name} {e}")
            return                

    # set regex in config
    def create_functions_regex(self, config = ""):
        cmd = f"cd {self.benchmark_address} && faas-cli deploy -f config.yml --gateway={self.gateway} {config} -n {self.namespace}"
        os.system(cmd)

    def delete_function(self, function_name):
        cmd = f"cd {self.benchmark_address} &&  faas-cli remove -f config.yml --gateway={self.gateway} --filter '{function_name}' -n {self.namespace}"
        os.system(cmd)
    
    def create_list_functions(self,function_lists, config=""):
        for function_name in function_lists:
            self.creat_function(function_name, config)
        return
    

    
class Scaler:
    def __init__(self, namespace, kube_client, v1_client):
        self.model_configs = pd.read_csv("model_config.csv",names=["model","stages","inference_time"])
        self.MIN_REPLICAS = 1
        self.MAX_REPLICAS = 4
        self.QUEUE_TIME = 0.3
        self.namespace = namespace
        self.kube_client = kube_client
        self.v1_client = v1_client
        self.pipeline_operator = Pipeline(namespace, self.kube_client, self.v1_client)
        self.openfaas_operator = OpenfaasOperator(self.namespace)
        self.scale_interval = 5*60
        self.max_stages = 32

    def theoretical_max_tp(self, model_name):
        theoretical_max_tp = 0
        for pipeline in self.pipelines:
            if '-0-' in pipeline['pod_name']:
                stage = pipeline['stages'][0]
                exec_time = self.model_configs.loc[(self.model_configs['model'] == model_name) & (self.model_configs['stages'] == stage)]['inference_time']
                theoretical_max_tp += 1 / exec_time
        
        theoretical_max_tp = theoretical_max_tp * (1 + self.QUEUE_TIME)
        return theoretical_max_tp
    
    def theoretical_latency(self, model_name):
        theoretical_latency = []
        for pipeline in self.pipelines:
            if '-0-' in pipeline['pod_name']:
                stage = pipeline['stages'][0]
                exec_time = self.model_configs.loc[(self.model_configs['model'] == model_name) & (self.model_configs['stages'] == stage)]['inference_time']
                theoretical_latency.append((exec_time * stage) + 0.01 * (stage-1))
        return sum(theoretical_latency)/len(theoretical_latency)


    # qps increase, using more replicas to handle more requests
    def calculate_replicas_by_rps(self,model, rps):
        max_tp = self.theoretical_max_tp(model)
        desired_replicas = math.ceil(rps / max_tp)
        desired_replicas = max(self.MIN_REPLICAS, min(self.MAX_REPLICAS, desired_replicas))
        print(f"Model: {model}, RPS: {rps}, Replicas: {desired_replicas}")
        return desired_replicas

    def get_current_replicas(self, model):
        deployments = self.kube_client.list_namespaced_deployment(namespace=self.namespace)
        # get deployment start with {model}-{submod-0}
        for deployment in deployments.items:
            deployment_name = deployment.metadata.name
            if deployment_name.startswith(f"{model}-submod-0"):
                return deployment.spec.replicas
        return None
    
    def scale_up_using_fined_grain(self, model, desired_replicas):
        model_stages = self.model_configs.loc[self.model_configs['model'] == model]['stages']
        fined_grained_stages = model_stages.values.astype(int).max()
        max_stage = min(self.max_stages, fined_grained_stages)
        model_nums = fined_grained_stages // max_stage
        subfix = "latency-64"
        deployment_annotations = f"{max_stage}-stages"
        deployment_ids, no_in_ids = self.pipeline_operator.calculate_stages_to_pod(model, max_stage)
        # if deployment_ids is exist, scale up, or create new function
        function_names = [f"{model}-{i}-{subfix}" for i in deployment_ids]

        already_exist_deployment = [deployment.metadata.name for deployment in self.kube_client.list_namespaced_deployment(namespace=self.namespace).items if deployment.metadata.name.startswith(model)]
        function_to_create = list(set(function_names) - set(already_exist_deployment))
        current_replicas = len([pod_name for pod_name in self.pipeline_operator.pipelines['pod_name'].tolist() if '-0-' in pod_name])
        add_replicas = desired_replicas - current_replicas

        for deployment in already_exist_deployment:
            # if deployment_name start with function_names, scale up
            deployment = self.kube_client.read_namespaced_deployment(name=deployment, namespace=self.namespace)
            # get current replicas
            replicas = deployment.spec.replicas
            # get desired replicas
            if desired_replicas > replicas:
                deployment.spec.replicas = desired_replicas
                # container env update model_nums
                contaners = []
                for container in deployment.spec.template.spec.containers:
                    env_vars = {env.name: env for env in container.env}
                    env_vars['model_nums'] = client.V1EnvVar(name='model_nums', value=str(desired_replicas))
                    container.env = list(env_vars.values())
                    contaners.append(container)
                deployment.spec.template.spec.containers = contaners
                self.patch_deployments(deployment)


        if len(function_to_create) != 0:
            faas_cli_env = f"--env pipeline={deployment_annotations} --env stages={max_stage} --env model_nums={model_nums}"
            self.openfaas_operator.create_list_functions(function_lists=function_to_create, config=faas_cli_env)
            self.directly_scale_function_using_current_stage(function_to_create, add_replicas)
            return


    def directly_scale_function_using_current_stage(self, function_names, scale):
        # scale using kubernetes 
        for fname in function_names:
            deployment = self.kube_client.read_namespaced_deployment(name=fname, namespace=self.namespace)
            deployment.spec.replicas = scale
            self.patch_deployments(deployment)


    def patch_deployments(self, deployment):
        maxtries = 3
        for _ in range(maxtries):
            try:
                self.kube_client.patch_namespaced_deployment(name=deployment.metadata.name, namespace=self.namespace, body=deployment)
                print(f"Deployment '{deployment.metadata.name}' updated successfully.")
                break
            except Exception as e:
               
                print("Conflict detected, retrying...in patch_deployments")
                time.sleep(1)

    # scale down model to desired_replicas, first scale down pipeline
    def scale_down_pipeline(self, model, to_scale, pipeline):
        self.pipeline_operator.update_pipeline_stage_pods()
        # get current pipeline
        pipeline = f"{pipeline}-stages"
        pipeline_record = self.pipeline_operator.pipelines.loc[(self.pipeline_operator.pipelines['model'] == model) & (self.pipeline_operator.pipelines['pipeline'] == pipeline)]
        # get deployment_name
        deployment_names = pipeline_record['deployment_name'].tolist()
        deployment_names = list(set(deployment_names))
        pod_names = pipeline_record['pod_name'].tolist()
        # pipeline_replicas = numbers of pod in pod_names contains submod-0
        pipeline_replicas = len([pod_name for pod_name in pod_names if '-0-' in pod_name])
        replicas_reduce = pipeline_replicas - to_scale
        if replicas_reduce <= 0:
            return False

        # pod_names_to_delete = each deployment random choose replicas_reduce pods
        pod_names_to_delete = []
        for deployment_name in deployment_names:
            deployment_record = pipeline_record.loc[pipeline_record['deployment_name'] == deployment_name]
            replicas_reduce = deployment_record.shape[0] if deployment_record.shape[0] < replicas_reduce else replicas_reduce
            pod_names_to_delete.extend(deployment_record.sample(n=replicas_reduce)['pod_name'].tolist())
        # delete pod
        for pod_name in pod_names_to_delete:
            deployment_name = self.pipeline_operator.pipelines.loc[self.pipeline_operator.pipelines['pod_name'] == pod_name]['deployment_name'].values[0]
            deployment = self.kube_client.read_namespaced_deployment(name=deployment_name, namespace=self.namespace)
            current_replicas = deployment.spec.replicas
            if current_replicas > 1:
                self.delete_pod_by_name(pod_name)
                deployment.spec.replicas -= 1
                self.patch_deployments(deployment)
            else:
                self.openfaas_operator.delete_function(deployment_name)
        # update pipeline
        self.pipeline_operator.update_pipeline_stage_pods()
        # delete function not in pipeline


    def delete_pod_by_name(self, pod_name):
        try:
            self.v1_client.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
            print(f"Pod '{pod_name}' deleted successfully.")
        except Exception as e:
            print("Conflict detected, retrying... in delete_pod_by_name")
            time.sleep(1)
    
    def delete_all_deployment(self, model):
        deployments = self.kube_client.list_namespaced_deployment(namespace=self.namespace)
        for deployment in deployments.items:
            deployment_name = deployment.metadata.name
            if deployment_name.startswith(model):
                print(f"Delete function {deployment_name}")
                self.openfaas_operator.delete_function(deployment_name)
                print(f"Deployment '{deployment_name}' deleted successfully.")


# test
namespace = "cdgp"
# pipeline_operator = Pipeline(namespace, kube_client, v1)
# pipeline_operator.change_pipeline("opt-66b", "32", 16)
# pipeline_operator.label_all_pod_in_namespace()


async def main():

    dashengScaler = Scaler(namespace, kube_client, v1)
    # dashengScaler.scale_down_pipeline("opt-66b", 16, "32-stages")
    # dashengScaler.delete_all_deployment("opt-66b")

    # # time.sleep(15)
    # print("start testing pipeline scale up from scratch")
    # dashengScaler.scale_up_using_fined_grain("opt-66b", 2)

    # # # wait for pipeline ready
    # dashengScaler.pipeline_operator.sure_pipeline_complete("opt-66b", 32)

    # print("start testing pipeline change from 32-stages to 16 stages")
    await dashengScaler.pipeline_operator.change_pipeline("opt-66b", "32", 16)

    # print("start testing scale up using fined grain, create a 32-stages pipeline")
    # dashengScaler.scale_up_using_fined_grain("opt-66b", 2)

    # print("start testing one of pipeline change from 16-stages to 8 stages")
    # await dashengScaler.pipeline_operator.change_pipeline("opt-66b", "16", 8)

    # print("scale down one of pipeline to zero")
    # pipelines = dashengScaler.pipeline_operator.pipelines
    # stages = list(set(pipelines.loc[pipelines['model'] == "opt-66b"]['stages'].tolist()))
    # for stage in stages:
    #     stage = int(stage)
    #     dashengScaler.pipeline_operator.sure_pipeline_complete("opt-66b", stage)
    # dashengScaler.scale_down_pipeline("opt-66b", 0, "32")


# Run the event loop
asyncio.run(main())


