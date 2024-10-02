import time
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import asyncio
import pandas as pd
import math
import subprocess

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
# Constants

SCALE_UP_COOLDOWN = 1 * 60 #10s # 10 minutes
DEFAULT_NUM_REPLICAS = 1  # Default number of replicas for each deployment
MAX_REPLICAS = 3  # Maximum number of replicas for each deployment
lantency_map = {
    "mnist": 0.11,
    "lstm": 0.07,
    "vggnet11": 1.49,
    "mobilenet": 0.93,
    "shufflenet": 0.2,
    "resnet18": 2.61,
    "resnet34": 4.30,
    "resnet50": 7.48,
}
MODEL_EXEC_TIME = {
    "BERT": 0.127,
    "OPT": 0.3,
    "LLAMA": 0.035,
    "WHISPER": 0.073,
    "WIDERESNET": 1.2
}
# Scale record dictionary
scale_records = {}


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


def extract_profiling(csv_file, input_value):
    df = pd.read_csv(csv_file)
    input_parts = input_value.split("-")
    model_name = "-".join(input_parts[:-4])
    graph_id = int(input_parts[-3])
    oap_method = input_parts[-2].upper()
    oap_params = input_parts[-1]

    filtered_df = df[
        (df["model_name"] == model_name)
        & (df["graph_id"] == graph_id)
        & (df["oap_method"].str.split("_").str[0] == oap_method)
        & (df["oap_params"].str.split("-").str[-1] == oap_params)
        & (df["device_type"].str.startswith("cuda"))
    ]
    print(f"find:{model_name} {graph_id} {oap_method} {oap_params}")
    if len(filtered_df) == 0:
        return None
    else:
        return (
            filtered_df.iloc[0]["batch_size"],
            filtered_df.iloc[0]["load_time"],
            filtered_df.iloc[0]["exec_time"],
            filtered_df.iloc[0]["warmup_time"],
            filtered_df.iloc[0]["input_size"],
        )


async def Tetris_scale(namespace):
    # Get all deployments in the namespace
    deployments = kube_client.list_namespaced_deployment(namespace)
    for deployment in deployments.items:
        deployment_name = deployment.metadata.name
        # print(deployment_name)
        # Check if this deployment is a '-submod-0' one
        # if "-submod-0" not in deployment_name:
        #     continue

        # Extract the module name from the deployment name (assumes format is 'module-submod-0-...')
        # module_name = deployment_name.split("-submod-0")[0]

        # Get the requests per second for the '-submod-0' function
        rps_query = f'sum(irate(gateway_function_invocation_started{{function_name="{deployment_name}.{namespace}"}}[1m]))'
        result = prometheus_client.custom_query(query=rps_query)
        # print(result)
        request_df = pd.DataFrame()
        for r in result:
            df = pd.DataFrame([r["value"]], columns=["time", "rps"])
            request_df = pd.concat([request_df, df])

        if request_df.empty:
            print(f"{deployment_name} no requests")
            # print(f"Query: {rps_query}")
            continue

        rps = float(request_df["rps"].sum())
        print(f"RPS for {deployment_name} is {rps}")

        # DTS：
        #
        # while R > 0 do
        # R ← R− (b_i*p_i)/l_i;
        # R = rps residual RPS
        # b_i = 1  the maximum batch size for that instance to process requests
        # (no batch queue exists for an inference thread)
        # p_i = 1  the number of concurrent inference threads
        # l_i = warmup_time  the inference latency under previous configurations

        # subgraph_profiling_csv_path = (
        #     "/home/pengshijie/sdag/profiling/subgraph_profiling.csv"
        # )
        # try:
        #     (
        #         batch_size,
        #         load_time,
        #         exec_time,
        #         warmup_time,
        #         input_size,
        #     ) = extract_profiling(subgraph_profiling_csv_path, deployment_name)
        # except:
        #     print(
        #         f"Skipping scale-up for {deployment_name} due to not obtaining its execution time."
        #     )
        #     continue
        # lantency = warmup_time
        batch_size = 1
        # lantency = lantency_map[deployment_name]
        lantency = MODEL_EXEC_TIME[deployment_name.split("-")[0].upper()]
        # current_replicas = deployment.spec.replicas
        desired_replicas = math.ceil(rps / ((batch_size * 1.0) / lantency))
        current_replicas = deployment.spec.replicas
        desired_replicas = min(desired_replicas, MAX_REPLICAS)
        current_scale = kube_client.read_namespaced_deployment_scale(
            deployment_name, namespace
        )
        if rps == 0:
            # continue
            desired_replicas = math.ceil(current_scale.spec.replicas / 2)
            print(
                f"Scale for {deployment_name} to half replicas {desired_replicas} due to zero RPS"
            )
        if desired_replicas == 0:
            desired_replicas = 1

        # print(
        #     f"deployment:{deployment_name}:\n exec_time={exec_time} \t rps={rps} \ndesired_replicas = {desired_replicas} \t current_replicas = {current_scale.spec.replicas}\n"
        # )
        if desired_replicas == current_scale.spec.replicas:
            print(
                f"Skipping scale-up for {deployment_name} due to already at desired replicas"
            )
            continue

        if desired_replicas < current_scale.spec.replicas:
            # Check if we've scaled this deployment in the last 10 minutes
            last_scale_time = scale_records.get(deployment_name)
            if last_scale_time and time.time() - last_scale_time < SCALE_UP_COOLDOWN:
                # It's been less than 10 minutes since we last scaled this deployment, so skip this cycle
                print(f"{deployment_name} cooldown")
                continue
        print(f"Desired replicas for {deployment_name} functions is {desired_replicas}")
        # Scale all functions for the same module to the desired number of replicas
        # for d in deployments.items:
        #     if deployment_name in d.metadata.name:
        await scale_deployment(deployment_name, namespace, desired_replicas)
        scale_records[deployment_name] = time.time()


async def check_and_scale(namespace):
    # Retrieve all deployments in the namespace
    try:
        deployments = kube_client.list_namespaced_deployment(namespace)
        if not deployments.items:
            # print(f"No deployments found in namespace {namespace}")
            return
        await Tetris_scale(namespace)
    except Exception as e :
        print("Error",e)
        return


# Main loop
while True:
    asyncio.run(check_and_scale("cdgp"))
    time.sleep(5)
