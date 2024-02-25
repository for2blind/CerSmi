from kubernetes import client, config
import re
from perfering import GPUMetric
from flask import Flask, request
import json
import ssl, os
import base64
import  pandas as pd
from Kuberinit import KubernetesInstance


os.chdir(os.path.dirname(os.path.abspath(__file__)))

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain('server.crt', 'server.key')

from prometheus_api_client import PrometheusConnect
prometheus = PrometheusConnect(url="http://172.169.8.253:30500")
cluster_config={
"c108": {
        "master_ip": "172.169.8.253",
        "kubeconfig_path": "config_c1",
        "prometheus_port": 31113,
        "price": 0.019
    }
}
nodes=['cc224', 'cc229','cc230','cc234', 'cc225', 'cc240',   'cc241',
       'cc238', 'cc239']
# node_list_1 = ['cc224']
node_list_1 = ['cc224', 'cc229']
node_list_2 = ['cc224', 'cc229','cc230','cc234']
node_list_3 = ['cc224', 'cc229','cc230','cc234','cc225', 'cc240']


kinstance = KubernetesInstance(cluster_config['c108']['kubeconfig_path'], cluster_config['c108']['master_ip'])
priority_scores = {}
# Load the Kubernetes configuration
config.load_kube_config()
# Initialize the Kubernetes API client
api = client.CoreV1Api()
v1 = client.AppsV1Api()
scheduler_name = "odc_scheduler"
gpu_metric = GPUMetric()
# function_to_gpu_map,model_to_gpu_map= kinstance.get_pod_to_gpu_map()
reserved_cpu = 0.3
namespace = "cdgp"
GPU_QUOTA = 1.15
memory_units = {
    'K': 1024 ** 2,
    'M': 1024 ** 1,
    'G': 1,
}

def init_mappings():
    node_list = api.list_node().items
    server_to_pods = {}
    server_gpu_map = {}
    for node in node_list:
        node_name = node.metadata.name
        server_to_pods[node_name] = []
        gpu_allocatable = int(node.status.allocatable.get("nvidia.com/gpu", 0))
        server_gpu_map[node_name] = gpu_allocatable
        if gpu_allocatable > 0 and node_name not in priority_scores:  # Initialize priority score if not already present
            priority_scores[node_name] = 100  # Choose an initial value for the score
    pod_list = api.list_namespaced_pod(namespace=namespace).items
    for pod in pod_list:
        if pod.spec.scheduler_name != scheduler_name:
            continue
        node_name = pod.spec.node_name
        if node_name is None:
            continue
        server_to_pods[node_name].append(pod.metadata.name)
        gpu_requested = int(pod.spec.containers[0].resources.requests.get("nvidia.com/gpu", 0))
        if gpu_requested > 0:
            server_gpu_map[node_name] -= gpu_requested
    return server_to_pods, server_gpu_map



# get real time memory spare
def standar_mem(memory_str):
    memory_match = re.match(r'(\d+)([KMGT]i?)?', memory_str)
    memory_value = int(memory_match.group(1))
    memory_unit = memory_match.group(2).replace('i', '')

    # Convert the memory allocation to bytes using the appropriate multiplier
    if memory_unit:
        memory_multiplier = memory_units.get(memory_unit, 1)
        memory_allocatable = memory_value / memory_multiplier
    else:
        memory_allocatable = memory_value / 1024 ** 3
    return memory_allocatable

def get_function_gpu_require(function):
    function_model_size_df = pd.read_csv('../benchmark/function_model_size.csv')
    if '-whole' not in function:
        gpu_quota = 1.4
    else:
        gpu_quota = GPU_QUOTA
    require = function_model_size_df[function_model_size_df['function_name_config'] == function]['size'].values[0] * gpu_quota
    
    return min(43, require)


def schedule(pod):
    # Get function from pod's labels
    function = pod['metadata']['labels'].get('faas_function')
    if not function:
        return None, None
    # Get required GPU for the function
    gpu_require = get_function_gpu_require(function)
    if not gpu_require:
        return None, None
    # Get used GPUs for the function
    function_to_gpu_map, model_to_gpu_map = kinstance.get_pod_to_gpu_map()
    model_name= '-'.join(function.split('-')[0:2])
    used_gpus_for_model = model_to_gpu_map.get(model_name, [])
    used_gpus_for_function = function_to_gpu_map.get(function,[])
    # Get real-time GPU metrics
    spare_GM = gpu_metric.hard_get_gpu(prometheus)
    spare_GM = spare_GM[spare_GM.memory_spare > gpu_require]
    spare_GM = spare_GM[spare_GM['node_name'].isin(nodes)]
    spare_GM = spare_GM[~spare_GM['device_uuid'].isin(used_gpus_for_function)]
    if spare_GM.empty:
        print(f"No GPU available for model {model_name}")
        return None, None
        
    candidates = spare_GM.node_name.unique().tolist()

    if ('-0-' in pod['metadata']['generateName'] or '-whole' in pod['metadata']['generateName']) and 'cc225' in candidates:
        candidates.remove('cc225')
    
    if max(priority_scores.values()) <15:
        init_mappings()
    
    #  get top 4 nodes with highest priority
    selectable_nodes = list(set(candidates) & set(priority_scores.keys()))

    top_4_priority = sorted(selectable_nodes, reverse=True)[:4]
        
    selected_gpu = spare_GM.loc[spare_GM['node_name'].isin(top_4_priority)].sample(n=1).iloc[0]
    node_name = selected_gpu['node_name']
    device_uuid = selected_gpu['device_uuid']
    device_id = selected_gpu['device_id']
    if node_name:
        # Decrease priority of the selected node
        priority_scores[node_name] -= 1
        print("Selected node: %s, priority: %d" % (node_name, priority_scores[node_name]))

    gpu_metric.allocate_GM(node_name, device_uuid, gpu_require)
    print("Selected GPU: %s, memory_spare: %d, for model %s" % (device_uuid, selected_gpu['memory_spare'], model_name))
    return node_name, device_uuid


def patch_pod(pod, node_name, device_uuid):
    # Patch the pod to the selected node and device
    patch_operation = [
        # pod name
        {
            "op": "add",
            "path": "/spec/nodeSelector",
            "value": {
                "kubernetes.io/hostname": node_name
            }
        },
        {
            "op": "add",
            "path": "/spec/containers/0/resources",
            "value": {
                "limits": {
                    "nvidia.com/gpu": 4,
                    "cpu": 3
                },
                "requests": {
                    "nvidia.com/gpu": 4
                }
            }
        },
        {
            "op": "add",
            "path": "/spec/containers/0/env",
            "value": pod['spec']['containers'][0]['env'] +[
                {
                    "name": "NVIDIA_VISIBLE_DEVICES",
                    "value": device_uuid
                }
            ]
        },
        {
            # mount local disk /data/model/openfaas/ to container /data/model/openfaas/
            "op": "add",
            "path": "/spec/containers/0/volumeMounts",
            "value": [
                {
                    "name": "openfaas-model",
                    "mountPath": "/data/model/openfaas/"
                }
            ]
        },
        {
            "op": "add",
            "path": "/spec/volumes",
            "value": [
                {
                    "name": "openfaas-model",
                    "hostPath": {
                        "path": "/data/model/openfaas/"
                    }
                }
            ]
        },
        {
            "op": "add",
            "path": "/spec/containers/0/readinessProbe",
            "value": {
                "httpGet": {
                    "path": "/loaded",
                    "port": 5000
                },
                "initialDelaySeconds": 30,
                "periodSeconds": 5
            }
        }

    ]
    return base64.b64encode(json.dumps(patch_operation).encode('utf-8'))


def admission_response(uid, message,pod, node_name, device_uuid):
    if not node_name:
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": uid,
                "allowed": False,
                "status": {"message": message}
            }
        }
    # Create an admission response
    return {
       "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": True,
            "status": {"message": message},
            "patchType": "JSONPatch",
            "patch": patch_pod(pod, node_name, device_uuid).decode('utf-8')
    }
}


app = Flask(__name__)
init_mappings()

@app.route("/mutate", methods=["POST"])
def mutate():
    review = request.json
    pod = review['request']['object']
    node_name, device_uuid = schedule(pod)
    if node_name:
        addmission = admission_response(review['request']['uid'], "success", pod, node_name, device_uuid)
        return addmission
    else:
        return admission_response(review['request']['uid'], "fail", pod, node_name, device_uuid)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9008,ssl_context=context, threaded=False)
