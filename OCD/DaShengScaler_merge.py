import time
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import re
import subprocess, math, os

from kubernetes.client import (
    V1HorizontalPodAutoscaler,
    V1HorizontalPodAutoscalerSpec,
    V1CrossVersionObjectReference,
)

# Initialize Prometheus client
prometheus_client = PrometheusConnect(
    url="http://172.169.8.253:31113/", disable_ssl=False
)


# Load the Kubernetes configuration
config.load_kube_config()

# Initialize the Kubernetes API client
kube_client = client.AppsV1Api()
autoscaling_client = client.AutoscalingV1Api()
v1 = client.CoreV1Api()

# Constants
TARGET_RPS = 25  # Requests Per Second for each pod
MAX_QUEUE_TIME = 4  # Maximum Queue Time in Seconds
MAX_REPLICAS = 4  # 8  # Maximum number of replicas for each deployment
SCALE_UP_COOLDOWN = 6 * 60  # 10 minutes
DEFAULT_NUM_REPLICAS = 2  # Default number of replicas for non-ME
MIN_REPLICAS = 2  # Minimum number of replicas for ME
MODEL_EXEC_TIME = {
    "BERT": 0.127,
    "GPT": 0.095 * 3,
    "LLAMA": 0.035,
    "WIDERESNET": 0.649 + 0.6,
    "WHISPER": 0.073,
    "LLAMA2KK70": 1.5,
}
scale_records = {}


def easeOutBack(t, b=2, c=7):
    return c * ((math.sin(t * (2 - t) * math.pi / 2)) ** 1) + b


def calculate_replicas(rps, b=2, c=7):
    normalized_rps = min(MIN_REPLICAS, rps / (8 * 12))
    replicas = easeOutBack(normalized_rps, b=b, c=c)
    replicas = max(1, min(MAX_REPLICAS, replicas))
    print("Replicas: ", replicas)
    return int(replicas)


def calculate_replicas_by_model(model_name, rps, pipeline):
    model_exec_time = MODEL_EXEC_TIME.get(model_name, 0.1) + 0.01 * int(pipeline)
    theoretical_max_rps_per_replica = 1 / model_exec_time
    desired_replicas = math.ceil(rps / theoretical_max_rps_per_replica)
    desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))
    print(f"Model: {model_name}, RPS: {rps}, Replicas: {desired_replicas}")
    return desired_replicas


import pandas as pd

data = {
    "model": ["Model1", "Model1", "Model1","Model1","Model1", "Model2", "Model2", "Model2"],
    "stage": [2, 4, 8,16,32, 2, 4, 8],
    "qps": [100, 150, 120, 80, 110, 90],
    "exec_time": [10, 15, 12, 8, 11, 9],
    "replicas":[0,0,0,0,0,0]
}
stage_profile_pd = pd.DataFrame(data)

def get_current_stage_num(deployment_name, namespace):
    deployment_df= get_deployment_info(deployment_name, namespace)
    deployment_prefix = deployment_name.split("-submod-0")[0]
    pipeline = deployment_name.split("-")[-1]
    powers_of_two = [2 ** i for i in reversed(range(int(math.log(pipeline, 2)) + 1))]
    for x in powers_of_two:
        conditions_1 = (stage_profile_pd["model"] == deployment_prefix) & (stage_profile_pd["stage"] == x)
        conditions_2 = (deployment_df["deployment_name"] == re.sub(r"-(\d+)-", f"-{x}-", deployment_name))
        stage_profile_pd.loc[conditions_1, "replicas"]=deployment_df[conditions_2,"replicas"].default(0).values[0]
        conditions_3 = (deployment_df["deployment_name"] == re.sub(r"-(\d+)-", f"-{x*2}-", deployment_name))
        stage_profile_pd.loc[conditions_1, "replicas"]-=stage_profile_pd.loc[conditions_3, "replicas"].default(0).values[0]
    return stage_profile_pd

def get_cv(deployment_name, namespace):
    # stddev_over_time(sum(irate(gateway_function_invocation_started{function_name="opt-66b-submod-0-latency-64.cdgp"}[60s]))by (function_name) [10m:1s])/avg_over_time(sum(irate(gateway_function_invocation_started{function_name="opt-66b-submod-0-latency-64.cdgp"}[60s]))by (function_name) [10m:1s])
    cv_query = f'stddev_over_time(sum(irate(gateway_function_invocation_started{{function_name=~"{deployment_name}.{namespace}"}}[1m])) by (function_name) [10m:1s])/avg_over_time(sum(irate(gateway_function_invocation_started{{function_name=~"{deployment_name}.{namespace}"}}[1m])) by (function_name) [10m:1s])'
    result = prometheus_client.custom_query(query=cv_query)
    request_df = pd.DataFrame()
    for r in result:
        df = pd.DataFrame([r["value"]], columns=["time", "cv"])
        request_df = pd.concat([request_df, df])
    if request_df.empty:
        print(f"Query {cv_query} empty")
        return 0
    request_df["cv"] = pd.to_numeric(request_df["cv"], errors="coerce")
    request_df["cv"].fillna(0, inplace=True)
    cv = float(request_df["cv"].mean())
    return cv


def get_current_qps(deployment_name, namespace):
    rps_query = f'sum(irate(gateway_function_invocation_started{{function_name=~"{deployment_name}.{namespace}"}}[1m])) by (function_name)'
    result = prometheus_client.custom_query(query=rps_query)
    request_df = pd.DataFrame()
    for r in result:
        df = pd.DataFrame([r["value"]], columns=["time", "rps"])
        request_df = pd.concat([request_df, df])
    if request_df.empty:
        print(f"Query {rps_query} empty")
        return 0
    rps = float(request_df["rps"].sum())
    return rps


def get_avg_qps(deployment_name, namespace):
    # avg_over_time(sum(rate(gateway_function_invocation_started{function_name="opt-66b-submod-0-latency-64.cdgp"}[60s]))[10m:1s])
    rps_query = f'avg_over_time(sum(irate(gateway_function_invocation_started{{function_name=~"{deployment_name}.{namespace}"}}[1m])) by (function_name) [10m:1s]) '
    result = prometheus_client.custom_query(query=rps_query)
    request_df = pd.DataFrame()
    for r in result:
        df = pd.DataFrame([r["value"]], columns=["time", "rps"])
        request_df = pd.concat([request_df, df])
    if request_df.empty:
        print(f"Query {rps_query} empty")
        return 0
    request_df["rps"] = pd.to_numeric(request_df["rps"], errors="coerce")
    avg_rps = float(request_df["rps"].mean())
    return avg_rps


def get_lantency():
    latency = 1
    return latency


def get_throughput(deployment_name, namespace):
    # avg_over_time(sum(rate(gateway_function_invocation_total{function_name="opt-66b-submod-0-latency-64.cdgp",code="200"}[1m]))by (function_name) [5s:1s])
    throughput_query = f'avg_over_time(sum(irate(gateway_function_invocation_total{{function_name=~"{deployment_name}.{namespace}",code="200"}}[1m])) by (function_name) [5s:1s] '
    result = prometheus_client.custom_query(query=throughput_query)
    request_df = pd.DataFrame()
    for r in result:
        df = pd.DataFrame([r["value"]], columns=["time", "throughput"])
        request_df = pd.concat([request_df, df])
    if request_df.empty:
        print(f"Query {throughput_query} empty")
        return 0
    request_df["throughput"] = pd.to_numeric(request_df["throughput"], errors="coerce")
    throughput = float(request_df["throughput"].mean())
    return throughput


async def change_pipeline(namespace, scale_num, stage):
    pod_info_df = get_pod_info()

    await send_merge_requset(namespace, scale_num, stage)
    pass



def get_deployment_info(deployment_name, namespace):
    deployment_prefix = deployment_name.split("-submod-0")[0]
    deployments = kube_client.list_namespaced_deployment(namespace)
    matching_deployments = [
        deployment
        for deployment in deployments.items
        if deployment.metadata.name.startswith(deployment_prefix)
    ]
    return matching_deployments

def get_pod_info(deployment_name, namespace):
    matching_deployments=get_deployment_info(deployment_name, namespace)
    pod_info_list = []
    for deployment in matching_deployments:
        deployment_name = deployment.metadata.name
        pods = v1.list_namespaced_pod(
            namespace, label_selector=f"faas_function={deployment_name}"
        )
        for pod in pods.items:
            pod_name = pod.metadata.name
            pod_ip = pod.status.pod_ip
            pod_info_list.append(
                {
                    "deployment_name": deployment_name,
                    "pod_name": pod_name,
                    "pod_IP": pod_ip,
                }
            )
    pod_info_df = pd.DataFrame(pod_info_list)
    return pod_info_df


async def send_request(url, deployment_name, pod_name):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    print(f"merge {pod_name} success")
                else:
                    print(f"Failed to merge {pod_name} :{response}")
    except Exception as e:
        print(
            f"Error sending GET request to {url} for {deployment_name} - {pod_name}: {str(e)}"
        )


async def send_merge_requset(pod_info_df, namespace, pipe_num, stage):
    requests_sent_count = {}
    tasks = []
    for index, row in pod_info_df.iterrows():
        deployment_name = row["deployment_name"]
        pod_name = row["pod_name"]
        pipeline = deployment_name.split("-")[-1]
        pod_ip = row["pod_IP"]
        model_num = int(int(pipeline) / int(stage))
        current_request_count = requests_sent_count.get(deployment_name, 0)
        if current_request_count <= pipe_num:
            url = f"http://172.169.8.253:31112/function/{deployment_name}.{namespace}/merge/{model_num}"
            # url = f"http://{pod_ip}:8080/merge/{model_num}"
            tasks.append(send_request(url, deployment_name, pod_name))
            requests_sent_count[deployment_name] = current_request_count + 1
        else:
            print(
                f"Maximum requests {pipe_num} already sent for {deployment_name}. Skipping additional requests."
            )
        await asyncio.gather(*tasks)
    return True


async def remove_dead_pod(pod_name, namespace):
    try:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        if not pod.spec.containers[0].resources.limits.get("nvidia.com/gpu"):
            return
        if (
            not pod.status.container_statuses
            or not pod.status.container_statuses[0].state.running
        ):
            os.system(f"kubectl delete pod {pod.metadata.name} -n {namespace}")
    except:
        pass


async def check_pod_ready(namespace):
    while True:
        pod_status = {}
        pods = v1.list_namespaced_pod(namespace).items
        for pod in pods:
            if pod.status.container_statuses:
                pod_status[pod.metadata.name] = pod.status.container_statuses[0].ready
            else:
                pod_status[pod.metadata.name] = False

        pod_ready = all(status for status in pod_status.values())
        # delete the pod of status "CrashLoopBackOff"
        if not pod_ready:
            for pod in pod_status:
                if not pod_status[pod]:
                    await remove_dead_pod(pod, namespace)
        else:
            return pod_ready
        await asyncio.sleep(2)


async def scale_up_stage_pipeline(deployment_name, namespace, stage, scale_up_num):
    current_scale = kube_client.read_namespaced_deployment_scale(
        deployment_name, namespace
    )
    # opt-66b-submod-0-latency-64
    module_name = deployment_name.split("-submod-0")[0]
    pipeline = deployment_name.split("-")[-1]
    pod_info_df1 = get_pod_info()
    scale_up_No=np.linspace(0, pipeline, num=stage + 1)[0:-1]
    for i in scale_up_No:
        deployment_name_scale = re.sub(r"-(\d+)-", f"-{i}-", deployment_name)
        command = f"kubectl scale deployment {deployment_name_scale} --namespace={namespace} --replicas {current_scale.spec.replicas+scale_up_num}"
        subprocess.check_output(command)
    scale_records[module_name] = time.time()
    await check_pod_ready(namespace)
    pod_info_df2 = get_pod_info()
    new_pods_df = pod_info_df2[~pod_info_df2.isin(pod_info_df1.to_dict("list")).all(1)]
    await send_merge_requset(new_pods_df, namespace, scale_up_num, stage)
    
    return


def dasheng_scale_up(
    model,
    namespace,
    avg_qps,
    throughput,
):
    # fine_grained_model
    model_data = stage_profile_pd[stage_profile_pd["model"] == model]
    max_stage_row = model_data.loc[model_data["stage"].idxmax()]
    max_stage = max_stage_row["stage"]
    max_stage_qps = max_stage_row["qps"]
    scale_up_num = math.ceil((avg_qps - throughput) / max_stage_qps)
    scale_up_stage_pipeline(model, namespace, max_stage, scale_up_num)
    return


def dasheng_scale_out(model, avg_qps, throughput, namespace):
    # fine_grained_model
    desired_pipelines = math.floor(avg_qps / throughput)
    cur_cv = get_cv(model, namespace, avg_qps)
    stage_profile_pd=get_current_stage_num()
    # todo cv to pipeline
    change_pipeline(stage_profile_pd,model)
    return


async def dasheng_scale(namespace):
    # Get all deployments in the namespace
    deployments = kube_client.list_namespaced_deployment(namespace)
    for deployment in deployments.items:
        deployment_name = deployment.metadata.name
        if "-submod-0" not in deployment_name:
            continue
        module_name = deployment_name.split("-submod-0")[0]
        # wrk_name = deployment_name.split('-')[-1].upper()
        model_name = deployment_name.split("-")[0].upper()
        pipeline = deployment_name.split("-")[-1]
        cur_qps = get_current_qps(deployment_name, namespace)
        throughput = get_throughput(deployment_name)
        if cur_qps > throughput:
            dasheng_scale_up(deployment_name, cur_qps, throughput)
        else:
            avg_qps = get_avg_qps(deployment_name, namespace, namespace)
            dasheng_scale_out(deployment_name, avg_qps, throughput, namespace)


async def scale_deployment(deployment_name, namespace, num_replicas):
    # Check current number of replicas
    try:
        current_deployment = kube_client.read_namespaced_deployment(
            deployment_name, namespace
        )
        current_replicas = current_deployment.spec.replicas
        if current_replicas == num_replicas:
            # print(f"No need to scale {deployment_name}. Current replicas: {current_replicas}")
            return
    except ApiException as e:
        print(f"Failed to get deployment {deployment_name}: {e}")
        return

    # Scale up a specific deployment
    try:
        # Construct the command
        cmd = [
            "kubectl",
            "scale",
            "deployment",
            deployment_name,
            "--replicas",
            str(num_replicas),
            "--namespace",
            namespace,
        ]

        # Run the command
        subprocess.check_output(cmd)
        print(
            f"Successfully scaled deployment {deployment_name} to {num_replicas} replicas"
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to scale deployment {deployment_name}: {e}")


def create_hpa(api_instance, namespace, deployment_name):
    # Check if HPA already exists
    try:
        existing_hpa = api_instance.read_namespaced_horizontal_pod_autoscaler(
            deployment_name, namespace
        )
        # print(f"HPA for {deployment_name} already exists.")
        return
    except ApiException as e:
        if (
            e.status != 404
        ):  # If the error is something other than 'Not Found', re-raise the exception
            print(f"Error when checking if HPA exists: {e}")
            raise

    # Define the target resource
    target = V1CrossVersionObjectReference(
        api_version="apps/v1", kind="Deployment", name=deployment_name
    )

    # Define the HPA spec
    hpa_spec = V1HorizontalPodAutoscalerSpec(
        scale_target_ref=target,
        min_replicas=1,
        max_replicas=10,
        target_cpu_utilization_percentage=50,
    )

    # Define the HPA
    hpa = V1HorizontalPodAutoscaler(
        api_version="autoscaling/v1",
        kind="HorizontalPodAutoscaler",
        metadata={
            "name": deployment_name,
            "namespace": namespace,
        },
        spec=hpa_spec,
    )

    # Create the HPA
    api_instance.create_namespaced_horizontal_pod_autoscaler(namespace, hpa)
    print(f"HPA for {deployment_name} created.")


async def sdag_scale(namespace):
    # Get all deployments in the namespace
    deployments = kube_client.list_namespaced_deployment(namespace)
    for deployment in deployments.items:
        deployment_name = deployment.metadata.name

        # Check if this deployment is a '-submod-0' one
        if "-submod-0" not in deployment_name:
            continue

        # Extract the module name from the deployment_name (assumes format is 'module-submod-0-...')
        module_name = deployment_name.split("-submod-0")[0]
        # wrk_name = deployment_name.split('-')[-1].upper()
        model_name = deployment_name.split("-")[0].upper()
        pipeline = deployment_name.split("-")[-1]

        # Get the requests per second for the '-submod-0' function
        rps_query = f'sum(irate(gateway_function_invocation_started{{function_name="{deployment_name}.{namespace}"}}[1m]))'
        result = prometheus_client.custom_query(query=rps_query)
        request_df = pd.DataFrame()
        for r in result:
            df = pd.DataFrame([r["value"]], columns=["time", "rps"])
            request_df = pd.concat([request_df, df])

        if request_df.empty:
            print(f"Skipping scale-up for {deployment_name} due to no requests")
            print(f"Query: {rps_query}")
            continue

        rps = float(request_df["rps"].sum())
        print(f"RPS for {deployment_name} is {rps}")
        current_scale = kube_client.read_namespaced_deployment_scale(
            deployment_name, namespace
        )
        desired_replicas = calculate_replicas_by_model(model_name, rps, pipeline)
        if current_scale.spec.replicas == desired_replicas:
            print(f"Skipping scale-up for {deployment_name}")
            continue
        # if rps == 0:
        # continue

        if desired_replicas < current_scale.status.replicas:
            # Check if we've scaled this deployment in the last 10 minutes
            last_scale_time = scale_records.get(module_name)
            if last_scale_time and time.time() - last_scale_time < SCALE_UP_COOLDOWN:
                # It's been less than 10 minutes since we last scaled this deployment, so skip this cycle
                print(f"Skipping scale-down for {deployment_name} due to cooldown")
                return

        print(f"Desired replicas for {module_name} functions is {desired_replicas}")

        # Scale all functions for the same module to the desired number of replicas
        for d in deployments.items:
            if module_name in d.metadata.name:
                await scale_deployment(d.metadata.name, namespace, desired_replicas)
                scale_records[module_name] = time.time()


async def rps_scale(namespace):
    deployments = kube_client.list_namespaced_deployment(namespace)
    for deployment in deployments.items:
        deployment_name = deployment.metadata.name
        # Get the requests per second for the '-submod-0' function
        rps_query = f'sum(irate(gateway_function_invocation_started{{function_name="{deployment_name}.{namespace}"}}[1m]))'
        result = prometheus_client.custom_query(query=rps_query)
        request_df = pd.DataFrame()
        for r in result:
            df = pd.DataFrame([r["value"]], columns=["time", "rps"])
            request_df = pd.concat([request_df, df])

        if request_df.empty:
            print(f"Skipping scale-up for {deployment_name} due to no requests")
            print(f"Query: {rps_query}")
            continue

        rps = float(request_df["rps"].sum())
        print(f"RPS for {deployment_name} is {rps}")

        # Calculate desired number of replicas based on request rate and max queue time
        desired_replicas = calculate_replicas(rps, 1)
        current_scale = kube_client.read_namespaced_deployment_scale(
            deployment_name, namespace
        )

        if desired_replicas < current_scale.status.replicas:
            # Check if we've scaled this deployment in the last 10 minutes
            last_scale_time = scale_records.get(deployment_name)
            if last_scale_time and time.time() - last_scale_time < SCALE_UP_COOLDOWN:
                # It's been less than 10 minutes since we last scaled this deployment, so skip this cycle
                print(f"Skipping scale-down for {deployment_name} due to cooldown")
                return

        if desired_replicas == current_scale.status.replicas or rps == 0:
            print(
                f"Skipping scale-up for {deployment_name} due to already at desired replicas"
            )
            continue

        print(f"Desired replicas for {deployment_name} functions is {desired_replicas}")

        # Scale all functions for the same module to the desired number of replicas
        for d in deployments.items:
            await scale_deployment(d.metadata.name, namespace, desired_replicas)
            scale_records[deployment_name] = time.time()


async def delete_pods(namespace):
    # delete pod if pod is crashed
    status_skip = ["Running", "Pending"]
    try:
        pods = v1.list_namespaced_pod(namespace)
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return

    for pod in pods.items:
        if not pod.status.container_statuses:
            return
        for container_status in pod.status.container_statuses:
            if (
                container_status.state.waiting
                and container_status.state.waiting.reason == "CrashLoopBackOff"
            ):
                print(f"Pod {pod.metadata.name} is not running. Deleting...")
                try:
                    v1.delete_namespaced_pod(pod.metadata.name, namespace)
                    print(f"Pod {pod.metadata.name} deleted.")
                except ApiException as e:
                    print(
                        f"Exception when calling CoreV1Api->delete_namespaced_pod: {e}"
                    )


async def check_and_scale(namespace):
    # Retrieve all deployments in the namespace
    deployments = kube_client.list_namespaced_deployment(namespace)
    await delete_pods(namespace)
    if not deployments.items:
        print(f"No deployments found in namespace {namespace}")
        return

    if "-whole" in deployments.items[0].metadata.name:
        await rps_scale(namespace)
        return

    if "-me-" not in deployments.items[0].metadata.name:
        # scale to default replicas
        for deployment in deployments.items:
            deployment_name = deployment.metadata.name
            await scale_deployment(deployment_name, namespace, DEFAULT_NUM_REPLICAS)
        return
    # return
    await sdag_scale(namespace)


# Main loop
while True:
    asyncio.run(check_and_scale("cdgp"))
    time.sleep(3)
