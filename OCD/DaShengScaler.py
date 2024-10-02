import time
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import asyncio
import pandas as pd
import subprocess, math, os

from kubernetes.client import V1HorizontalPodAutoscaler, V1HorizontalPodAutoscalerSpec, V1CrossVersionObjectReference

# Initialize Prometheus client
prometheus_client = PrometheusConnect(url="http://172.169.8.253:31113/", disable_ssl=False) 


# Load the Kubernetes configuration
config.load_kube_config()

# Initialize the Kubernetes API client
kube_client = client.AppsV1Api()
autoscaling_client = client.AutoscalingV1Api()
v1 = client.CoreV1Api()

# Constants
TARGET_RPS = 25  # Requests Per Second for each pod
MAX_QUEUE_TIME = 4  # Maximum Queue Time in Seconds
MAX_REPLICAS = 3# 8  # Maximum number of replicas for each deployment
SCALE_UP_COOLDOWN = 6 * 60  # 10 minutes
DEFAULT_NUM_REPLICAS = 1 # Default number of replicas for non-ME
MIN_REPLICAS = 1  # Minimum number of replicas for ME
MODEL_EXEC_TIME = {
    'BERT': 0.127 ,
    'GPT': 0.095*3,
    'LLAMA': 0.7,
    'WIDERESNET': 0.649 + 0.6,
    'WHISPER':0.073,
    'LLAMA2KK70': 1.5
}
scale_records = {}

def easeOutBack(t, b=2, c=7):
    return c * ((math.sin(t * (2 - t) * math.pi / 2)) ** 1) + b

def calculate_replicas(rps, b=2,c=7):
    normalized_rps = min(MIN_REPLICAS, rps / (8 * 12))
    replicas = easeOutBack(normalized_rps, b=b,c=c)
    replicas = max(1, min(MAX_REPLICAS, replicas))
    print("Replicas: ", replicas)
    return int(replicas)


def calculate_replicas_by_model(model_name, rps, pipeline):
    model_exec_time = MODEL_EXEC_TIME.get(model_name, 0.1) + 0.01* int(pipeline)
    theoretical_max_rps_per_replica = 1 / model_exec_time
    desired_replicas = math.ceil(rps / theoretical_max_rps_per_replica)
    desired_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, desired_replicas))
    print(f"Model: {model_name}, RPS: {rps}, Replicas: {desired_replicas}")
    return desired_replicas




async def scale_deployment(deployment_name, namespace, num_replicas):
    # Check current number of replicas
    try:
        current_deployment = kube_client.read_namespaced_deployment(deployment_name, namespace)
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
        cmd = ["kubectl", "scale", "deployment", deployment_name, "--replicas", str(num_replicas), "--namespace", namespace]
        
        # Run the command
        subprocess.check_output(cmd)
        print(f"Successfully scaled deployment {deployment_name} to {num_replicas} replicas")
    except subprocess.CalledProcessError as e:
        print(f"Failed to scale deployment {deployment_name}: {e}")


def create_hpa(api_instance, namespace, deployment_name):
    # Check if HPA already exists
    try:
        existing_hpa = api_instance.read_namespaced_horizontal_pod_autoscaler(deployment_name, namespace)
        # print(f"HPA for {deployment_name} already exists.")
        return
    except ApiException as e:
        if e.status != 404:  # If the error is something other than 'Not Found', re-raise the exception
            print(f"Error when checking if HPA exists: {e}")
            raise

    # Define the target resource
    target = V1CrossVersionObjectReference(
        api_version="apps/v1",
        kind="Deployment",
        name=deployment_name
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
        spec=hpa_spec
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
        if '-submod-0' not in deployment_name:
            continue

        # Extract the module name from the deployment name (assumes format is 'module-submod-0-...')
        module_name = deployment_name.split('-submod-0')[0]
        # wrk_name = deployment_name.split('-')[-1].upper()
        model_name = deployment_name.split('-')[0].upper()
        pipeline = deployment_name.split('-')[-1]

        # Get the requests per second for the '-submod-0' function
        rps_query = f'sum(irate(gateway_function_invocation_started{{function_name="{deployment_name}.{namespace}"}}[1m]))'
        result = prometheus_client.custom_query(query=rps_query)
        request_df = pd.DataFrame()
        for r in result:
            df = pd.DataFrame([r['value']], columns=['time', 'rps'])
            request_df = pd.concat([request_df, df])

        if request_df.empty:
            print(f"Skipping scale-up for {deployment_name} due to no requests")
            print(f"Query: {rps_query}")
            continue

        rps = float(request_df['rps'].sum())
        print(f"RPS for {deployment_name} is {rps}")
        current_scale = kube_client.read_namespaced_deployment_scale(deployment_name, namespace)
        desired_replicas = calculate_replicas_by_model(model_name,rps,pipeline)
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
            df = pd.DataFrame([r['value']], columns=['time', 'rps'])
            request_df = pd.concat([request_df, df])

        if request_df.empty:
            print(f"Skipping scale-up for {deployment_name} due to no requests")
            print(f"Query: {rps_query}")
            continue

        rps = float(request_df['rps'].sum())
        print(f"RPS for {deployment_name} is {rps}")

        # Calculate desired number of replicas based on request rate and max queue time
        desired_replicas = calculate_replicas(rps,1)
        current_scale = kube_client.read_namespaced_deployment_scale(deployment_name, namespace)

        if desired_replicas < current_scale.status.replicas:
        # Check if we've scaled this deployment in the last 10 minutes
            last_scale_time = scale_records.get(deployment_name)
            if last_scale_time and time.time() - last_scale_time < SCALE_UP_COOLDOWN:
                # It's been less than 10 minutes since we last scaled this deployment, so skip this cycle
                print(f"Skipping scale-down for {deployment_name} due to cooldown")
                return
            
        if desired_replicas == current_scale.status.replicas or rps == 0:
            print(f"Skipping scale-up for {deployment_name} due to already at desired replicas")
            continue

        print(f"Desired replicas for {deployment_name} functions is {desired_replicas}")

        # Scale all functions for the same module to the desired number of replicas
        for d in deployments.items:
            await scale_deployment(d.metadata.name, namespace, desired_replicas)
            scale_records[deployment_name] = time.time()

async def delete_pods(namespace):
# delete pod if pod is crashed
    status_skip = ['Running', 'Pending']
    try:
        pods = v1.list_namespaced_pod(namespace)
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}")
        return

    for pod in pods.items:
        if not pod.status.container_statuses:
            return
        for container_status in pod.status.container_statuses:
            if container_status.state.waiting and container_status.state.waiting.reason == 'CrashLoopBackOff':
                print(f"Pod {pod.metadata.name} is not running. Deleting...")
                try:
                    v1.delete_namespaced_pod(pod.metadata.name, namespace)
                    print(f"Pod {pod.metadata.name} deleted.")
                except ApiException as e:
                    print(f"Exception when calling CoreV1Api->delete_namespaced_pod: {e}")

async def check_and_scale(namespace):
    # Retrieve all deployments in the namespace
    deployments = kube_client.list_namespaced_deployment(namespace)
    await delete_pods(namespace)
    if not deployments.items:
        print(f"No deployments found in namespace {namespace}")
        return

    if '-whole' in deployments.items[0].metadata.name:
        await rps_scale(namespace)
        return

    if '-me-' not in deployments.items[0].metadata.name:
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
