
from cachetools import TTLCache,cached
import pandas as pd
from prometheus_api_client import PrometheusConnect
prometheus = PrometheusConnect(url="http://172.169.8.86:30500")


class GPUMetric:
    def __init__(self):
        self.columns = ['node_name', 'device_uuid', 'memory_spare','memory_allocating']
        self.Gspare_GPU = pd.DataFrame(columns=self.columns)


    # @cached(cache=TTLCache(maxsize=100, ttl=3))
    def get_real_time_spare_GM(self):
        query = 'DCGM_FI_DEV_FB_FREE'
        result = prometheus.custom_query(query)
        spare_GM = pd.DataFrame(columns=self.columns)
        for r in result:
            node_name = r['metric']['kubernetes_node']
            device_uuid = r['metric']['UUID']
            memory_spare = int(r['value'][1])/1024
            device_metric  = pd.DataFrame({'node_name': [node_name], 'device_uuid': [device_uuid], 'memory_spare': [memory_spare]})
            spare_GM = pd.concat([spare_GM, device_metric])
        self.Gspare_GPU = spare_GM

    def allocate_GM(self, node_name, device_uuid, memory_allocating):
        self.Gspare_GPU.loc[(self.Gspare_GPU.node_name == node_name) & (self.Gspare_GPU.device_uuid == device_uuid), 'memory_spare'] -= memory_allocating
        self.Gspare_GPU.loc[(self.Gspare_GPU.node_name == node_name) & (self.Gspare_GPU.device_uuid == device_uuid), 'memory_allocating'] += memory_allocating


    def hard_get_gpu(self):
        self.get_real_time_spare_GM()
        return self.Gspare_GPU

    def get_spare_GM(self):
        return self.Gspare_GPU        