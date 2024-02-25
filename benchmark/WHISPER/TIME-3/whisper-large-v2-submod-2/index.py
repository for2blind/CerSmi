#!/usr/bin/env python
from quart import Quart, jsonify
import os
import random
import torch
import aiohttp
import subprocess
import re
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from multiprocessing import Process

def get_cuda_device_mapping():
    try:
        # Run nvidia-smi to get the device information
        output = subprocess.check_output(["nvidia-smi", "-L"], universal_newlines=True)
    except subprocess.CalledProcessError:
        print("nvidia-smi could not be executed. Are you sure the system has an NVIDIA GPU and nvidia-smi is installed?")
        return {}
    except FileNotFoundError:
        print("nvidia-smi was not found. Are you sure it's installed?")
        return {}

    # Parse the output using regex
    devices = re.findall(r"GPU (\d+): (.* \(UUID: GPU-(.*)\))", output)

    # Build a mapping of UUID to device ID
    uuid_to_id = {uuid: int(id) for id, _, uuid in devices}
    
    return uuid_to_id

def get_device_id_by_uuid(target_uuid):
    uuid_to_id = get_cuda_device_mapping()
    
    return uuid_to_id.get(target_uuid, None)




loaded = False
infer_device = 'cpu' if not torch.cuda.is_available() else os.environ.get('infer_device', 'cpu')
if infer_device!='cpu':
    cuda_id  = get_device_id_by_uuid(os.environ.get('NVIDIA_VISIBLE_DEVICES','0').replace('GPU-',''))
    if cuda_id is None:
        print("CUDA device not found")
        exit(1)
    infer_device = (f'cuda:{cuda_id}')
function_time_out = float(os.environ.get('read_timeout', 10).replace("s",""))

with torch.inference_mode():
    model= torch.load('/data/model/openfaas/cdgp/subgraphs/whisper-large-v2/TIME_OAP/3/whisper-large-v2-submod-2.pt', map_location=infer_device)
    # warm up 
    if hasattr(model, "gm_list"):
        for gm in model.gm_list:
            if hasattr(gm, "_buffers"):
                for k, v in gm._buffers.items():
                    if isinstance(v, torch.Tensor):
                        setattr(gm, k, v.to(infer_device))
    
    x = torch.load("/data/model/openfaas/cdgp/subgraphs/whisper-large-v2/TIME_OAP/3/whisper-large-v2-inputs-2.pt", map_location=infer_device)
    output = model(*x)
            
    loaded = True


app = Quart(__name__)

async def serve_loaded_endpoint():
    app_loaded = Quart("LoadedApp")

    @app_loaded.route('/loaded', methods=['GET'])
    async def check_loaded():
        try:
            if loaded:
                return jsonify({'result': True}), 200
            else:
                return jsonify({'result': False}), 500
        except:
            return jsonify({'result': False}), 500

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    await serve(app_loaded, config)


async def call_next_function():
    timeout = aiohttp.ClientTimeout(total=function_time_out)
    uid = random.randint(0, 10e6)
    try:
        async with aiohttp.ClientSession() as session:
            return
    except:
        return "call subgraph exception", 500
    
inference_semaphore = asyncio.Semaphore(2)

async def inference():
    async with inference_semaphore:
        try:
            s = torch.cuda.Stream(infer_device)
            with torch.cuda.stream(s):
                with torch.inference_mode():
                    output = model(*x)
                    s.synchronize()
                    return output
        except:
            return "inference exception", 500


 

@app.route('/', defaults={'path': ''}, methods=['GET', 'PUT', 'POST', 'PATCH', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'PUT', 'POST', 'PATCH', 'DELETE'])
async def call_handler(path):
    try:
        await inference()
        return {"result": "last"}, 200
        response, status_code = await call_next_function()
    except:
        return "infernece except", 500
    return response, status_code



if __name__ == '__main__':
    p = Process(target=asyncio.run, args=(serve_loaded_endpoint(),))
    p.daemon = True
    p.start()
    config = Config()
    # config.worker_class = "asyncio"
    # config.workers = 2
    config.bind = ["0.0.0.0:5000"]
    asyncio.run(serve(app, config))
