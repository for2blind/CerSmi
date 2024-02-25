from kubernetes import client, config
import re
from perfering import GPUMetric
from flask import Flask, request
import json
import ssl, os, sys
import base64
import pandas as pd
from Kuberinit import KubernetesInstance


os.chdir(os.path.dirname(os.path.abspath(__file__)))

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain("hook/server.crt", "hook/server.key")

from prometheus_api_client import PrometheusConnect

prometheus = PrometheusConnect(url="http://172.169.8.253:30500")
cluster_config = {
    "c108": {
        "master_ip": "172.169.8.253",
        "kubeconfig_path": "config_c1",
        "prometheus_port": 31113,
        "price": 0.019,
    }
}
kinstance = KubernetesInstance(
    cluster_config["c108"]["kubeconfig_path"], cluster_config["c108"]["master_ip"]
)
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
namespace = "dasheng"
GPU_QUOTA = 1.15
# GPU_QUOTA = 1.4

memory_units = {
    "K": 1024**2,
    "M": 1024**1,
    "G": 1,
}
global model_nums


# todo:  if pod is not in model_need_to_patch, then skip it, dont patch it, to make it pending


def init_mappings():
    node_list = api.list_node().items
    server_to_pods = {}
    server_gpu_map = {}
    for node in node_list:
        node_name = node.metadata.name
        server_to_pods[node_name] = []
        gpu_allocatable = int(node.status.allocatable.get("nvidia.com/gpu", 0))
        server_gpu_map[node_name] = gpu_allocatable
        if (
            gpu_allocatable > 0 and node_name not in priority_scores
        ):  # Initialize priority score if not already present
            priority_scores[node_name] = 100  # Choose an initial value for the score
    pod_list = api.list_namespaced_pod(namespace=namespace).items
    for pod in pod_list:
        if pod.spec.scheduler_name != scheduler_name:
            continue
        node_name = pod.spec.node_name
        if node_name is None:
            continue
        server_to_pods[node_name].append(pod.metadata.name)
        gpu_requested = int(
            pod.spec.containers[0].resources.requests.get("nvidia.com/gpu", 0)
        )
        if gpu_requested > 0:
            server_gpu_map[node_name] -= gpu_requested
    return server_to_pods, server_gpu_map


# get real time memory spare
def standar_mem(memory_str):
    memory_match = re.match(r"(\d+)([KMGT]i?)?", memory_str)
    memory_value = int(memory_match.group(1))
    memory_unit = memory_match.group(2).replace("i", "")

    # Convert the memory allocation to bytes using the appropriate multiplier
    if memory_unit:
        memory_multiplier = memory_units.get(memory_unit, 1)
        memory_allocatable = memory_value / memory_multiplier
    else:
        memory_allocatable = memory_value / 1024**3
    return memory_allocatable


def get_function_gpu_require(function):
    function_model_size_df = pd.read_csv("../benchmark/function_model_size.csv")
    if "-whole" not in function:
        gpu_quota = 1.4
    else:
        gpu_quota = GPU_QUOTA
    require = (
        function_model_size_df[
            function_model_size_df["function_name_config"] == function
        ]["size"].values[0]
        * gpu_quota
    )

    return min(43, require)


def schedule(pod):
    # Get function from pod's labels
    function = pod["metadata"]["labels"].get("faas_function")
    if not function:
        return None, None
    # Get required GPU for the function
    gpu_require = get_function_gpu_require(function)
    if not gpu_require:
        return None, None
    # Get used GPUs for the function
    (
        function_to_gpu_map,
        model_to_gpu_map,
        used_gpu_list,
    ) = kinstance.get_pod_to_gpu_map()
    model_name = "-".join(function.split("-")[0:2])
    used_gpus_for_model = model_to_gpu_map.get(model_name, [])

    used_gpus_for_function = function_to_gpu_map.get(function, [])
    # Get real-time GPU metrics
    spare_GM = gpu_metric.hard_get_gpu(prometheus)
    spare_GM = spare_GM[spare_GM.memory_spare > gpu_require]
    spare_GM = spare_GM[~spare_GM["device_uuid"].isin(used_gpus_for_function)] # share model
    # spare_GM = spare_GM[~spare_GM["device_uuid"].isin(used_gpus_for_model)]# share function
    # spare_GM = spare_GM[~spare_GM["device_uuid"].isin(used_gpu_list)]  # no share
    print(used_gpu_list)
    # print(f'skip used gpu for model {model_name}', used_gpus_for_model)

    if spare_GM.empty:
        print(f"No GPU available for model {model_name}")
        return None, None

    candidates = spare_GM.node_name.unique().tolist()

    # if (
    #     "-0-" in pod["metadata"]["generateName"]
    #     or "-whole" in pod["metadata"]["generateName"]
    # ) and "cc225" in candidates:
    #     candidates.remove("cc225")

    if max(priority_scores.values()) < 15:
        init_mappings()
    node_list = {
        "3090": ["cc225"],
        "A100": ["cc234"],
        "V100": ["cc229", "cc230"],
        "A40": ["cc238", "cc239", "cc240", "cc241"],
    }
    # if "-submod-0-me-" in pod["metadata"]["generateName"]:
    #     candidates = node_list["A100"]
    candidates = node_list["A40"]
    # candidates = node_list["3090"]+node_list["V100"]
    top_5_nodes = sorted(priority_scores, key=priority_scores.get, reverse=True)[:6]
    selectable_nodes = list(set(candidates) & set(top_5_nodes))
    top_priority_gpus = spare_GM[spare_GM["node_name"].isin(selectable_nodes)]
    sorted_gpus = top_priority_gpus.sort_values(by="memory_spare", ascending=False)
    if sorted_gpus.empty:
        print(f"No GPU top available for model {model_name}")
        return None, None
    selected_gpu = sorted_gpus.iloc[0]
    node_name = selected_gpu["node_name"]
    device_uuid = selected_gpu["device_uuid"]
    # device_id = selected_gpu["device_id"]

    # only_list = [
    #     ["GPU-1f504450-e385-aa43-b3e4-1a9414bbb738", "cc238"],
    #     ["GPU-0094482e-76c9-e439-065b-949c6faa00b7", "cc239"],
    #     ["GPU-1ab2c197-176f-00d8-fad7-e7f32fd9e25e", "cc240"],
    #     ["GPU-658ac102-8823-91c4-1b62-f66587782953", "cc241"],
    # ]
    # node_name = only_list[3][1]
    # device_uuid = only_list[3][0]
    if node_name:
        # Decrease priority of the selected node
        priority_scores[node_name] -= 1
        print(
            "Selected node: %s, priority: %d" % (node_name, priority_scores[node_name])
        )

    gpu_metric.allocate_GM(node_name, device_uuid, gpu_require)

    print(
        "Selected node %s, GPU: %s, memory_spare: %d, for model %s,require %s"
        % (
            node_name,
            device_uuid,
            selected_gpu["memory_spare"],
            pod["metadata"]["labels"].get("faas_function"),
            gpu_require,
        )
    )

    record = pd.DataFrame(
        [
            [
                pd.Timestamp.now(),
                pod["metadata"]["generateName"],
                device_uuid,
                node_name,
                gpu_require,
            ]
        ],
        columns=["time", "pod_name", "GPU_uuid", "node", "gpu_require"],
    )
    record.to_csv("single_scheduler_record.csv", mode="a", header=False, index=False)

    return node_name, device_uuid


def patch_pod(pod, node_name, device_uuid):
    # Patch the pod to the selected node and device
    patch_operation = [
        # pod name
        {
            "op": "add",
            "path": "/spec/nodeSelector",
            "value": {"kubernetes.io/hostname": node_name},
        },
        {
            "op": "add",
            "path": "/spec/containers/0/resources",
            "value": {
                "limits": {"nvidia.com/gpu": 6, "cpu": 3},
                "requests": {"nvidia.com/gpu": 6},
            },
        },
        {
            "op": "add",
            "path": "/spec/containers/0/env",
            "value": pod["spec"]["containers"][0]["env"]
            + [
                {"name": "NVIDIA_VISIBLE_DEVICES", "value": device_uuid},
                {"name": "model_nums", "value": f"{model_nums}"},
            ],
        },
        {
            # mount local disk /data/model/openfaas/ to container /data/model/openfaas/
            "op": "add",
            "path": "/spec/containers/0/volumeMounts",
            "value": [{"name": "openfaas-model", "mountPath": "/data/model/openfaas/"}],
        },
        {
            "op": "add",
            "path": "/spec/volumes",
            "value": [
                {
                    "name": "openfaas-model",
                    "hostPath": {"path": "/data/model/openfaas/"},
                }
            ],
        },
        {
            "op": "add",
            "path": "/spec/containers/0/readinessProbe",
            "value": {
                "httpGet": {"path": "/loaded", "port": 5001},
                "initialDelaySeconds": 30,
                "periodSeconds": 1,
            },
        },
    ]
    return base64.b64encode(json.dumps(patch_operation).encode("utf-8"))


def admission_response(uid, message, pod, node_name, device_uuid):
    if not node_name:
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {"uid": uid, "allowed": False, "status": {"message": message}},
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
            "patch": patch_pod(pod, node_name, device_uuid).decode("utf-8"),
        },
    }


app = Flask(__name__)
init_mappings()


@app.route("/mutate", methods=["POST"])
def mutate():
    review = request.json
    pod = review["request"]["object"]
    # print(f"Receive:{pod['metadata']['labels'].get('faas_function')}")
    node_name, device_uuid = schedule(pod)
    # print(f"====Select {node_name} {device_uuid} for {pod['metadata']['labels'].get('faas_function')}")
    if node_name:
        addmission = admission_response(
            review["request"]["uid"], "success", pod, node_name, device_uuid
        )
        return addmission
    else:
        return admission_response(
            review["request"]["uid"], "fail", pod, node_name, device_uuid
        )


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        model_nums = int(sys.argv[1])
    else:
        model_nums = 1
    if 64 % model_nums != 0:
        raise ValueError("64 must be divisible by model_nums")
    os.environ["model_nums"] = str(model_nums)
    print(f"model_nums={model_nums}")
    model_need_to_patch = [x for x in range(0, 64, model_nums)]
    app.run(host="0.0.0.0", port=9008, ssl_context=context, threaded=False)
