import pandas as pd

import warnings
warnings.filterwarnings('ignore')
class GPUMetric:
    def __init__(self):
        pass

    # @cached(cache=TTLCache(maxsize=100, ttl=3))
    def get_real_time_spare_GM(self,prometheus):
        query = 'DCGM_FI_DEV_FB_FREE'
        result = prometheus.custom_query(query)
        spare_GM = pd.DataFrame()
        for r in result:
            node_name = r['metric']['kubernetes_node']
            device_uuid = r['metric']['UUID']
            device_id = r['metric']['gpu']
            memory_spare = int(r['value'][1])/1024
            metric = 'memory_spare'
            device_metric  = pd.DataFrame({'node_name': [node_name], 'device_uuid': [device_uuid], 'value': [memory_spare],'device_id':[device_id], 'metric': [metric]})
            spare_GM = pd.concat([spare_GM, device_metric])
        self.Gspare_GPU = spare_GM.reset_index(drop=True)
        return spare_GM.reset_index(drop=True)
    
    def get_GM_util(self, prometheus):
        query = 'DCGM_FI_DEV_FB_USED/(DCGM_FI_DEV_FB_USED+DCGM_FI_DEV_FB_FREE)'
        result = prometheus.custom_query(query)
        GM_util = pd.DataFrame()
        for r in result:
            node_name = r['metric']['kubernetes_node']
            device_uuid = r['metric']['UUID']
            device_id = r['metric']['gpu']
            metric = 'memory_util'
            memory_util = float(r['value'][1])*100
            device_metric  = pd.DataFrame({'node_name': [node_name], 'device_uuid': [device_uuid], 'value': [memory_util], 'metric': [metric],'device_id':[device_id]})
            GM_util = pd.concat([GM_util, device_metric])
        return GM_util.reset_index(drop=True)
    
    def get_real_time_GPU_util(self, prometheus):
        query = 'DCGM_FI_DEV_DEC_UTIL'
        result = prometheus.custom_query(query)
        GPU_util = pd.DataFrame()
        for r in result:
            node_name = r['metric']['kubernetes_node']
            device_uuid = r['metric']['UUID']
            metric = 'gpu_util'
            value = int(r['value'][1])
            device_metric  = pd.DataFrame({'node_name': [node_name], 'device_uuid': [device_uuid], 'metric': [metric], 'value': [value]})
            GPU_util = pd.concat([GPU_util, device_metric])
        return GPU_util.reset_index(drop=True)
    
    def allocate_GM(self, node_name, device_uuid, memory_allocating):
        self.Gspare_GPU.loc[(self.Gspare_GPU.node_name == node_name) & (self.Gspare_GPU.device_uuid == device_uuid), 'memory_spare'] -= memory_allocating
        self.Gspare_GPU.loc[(self.Gspare_GPU.node_name == node_name) & (self.Gspare_GPU.device_uuid == device_uuid), 'memory_allocating'] += memory_allocating


    def hard_get_gpu(self,prometheus):
        memory_spare  = self.get_real_time_spare_GM(prometheus)
        memory_util = self.get_GM_util(prometheus)
        # set columns to be node_name device_uuid device_id memory_spare memory_util
        memory_spare = memory_spare.rename(columns={'value':'memory_spare'})
        memory_util = memory_util.rename(columns={'value':'memory_util'})
        gpu = pd.merge(memory_spare, memory_util, on=['node_name', 'device_uuid','device_id'], how='left')
        gpu = gpu[['node_name', 'device_uuid', 'device_id', 'memory_spare', 'memory_util']]
        return gpu.reset_index(drop=True)
    

class CPUMetric:
    def __init__(self):
        pass
    def get_real_time_spare_CPU(self,prometheus):
        query = 'instance:node_cpu_utilisation:rate5m'
        result = prometheus.custom_query(query)
        spare_CPU = pd.DataFrame()
        for r in result:
            node_name = r['metric']['instance']
            cpu_spare = 100 - float(r['value'][1])
            node_metric  = pd.DataFrame({'node_name': [node_name], 'value': [cpu_spare], 'metric': ['cpu_spare']})
            spare_CPU = pd.concat([spare_CPU, node_metric])
        
        return spare_CPU.reset_index(drop=True)
    
    def get_cpu_util(self,prometheus):
        query = 'instance:node_cpu_utilisation:rate5m'
        result = prometheus.custom_query(query)
        CPU_Util_DF = pd.DataFrame()
        for r in result:
            node_name = r['metric']['instance']
            cpu_util = float(r['value'][1])*100
            node_metric  = pd.DataFrame({'node_name': [node_name], 'value': [cpu_util], 'metric': ['cpu_util']})
            CPU_Util_DF = pd.concat([CPU_Util_DF, node_metric])
        return CPU_Util_DF.reset_index(drop=True)

    def get_cpu_mem_ratio(self,prometheus):
        query = 'instance:node_memory_utilisation:ratio'
        result = prometheus.custom_query(query)
        cm_util = pd.DataFrame()
        for r in result:
            node_name = r['metric']['instance']
            mem_util = float(r['value'][1])*100
            node_metric  = pd.DataFrame({'node_name': [node_name], 'value': [mem_util], 'metric': ['mem_util']})
            cm_util = pd.concat([cm_util, node_metric])
        return cm_util.reset_index(drop=True)


    def get_real_time_spare_mem(self, prometheus):
        query = 'instance:node_memory_utilisation:ratio'
        result = prometheus.custom_query(query)
        spare_mem = pd.DataFrame()
        for r in result:
            node_name = r['metric']['instance']
            mem_spare = 100 - float(r['value'][1])
            node_metric  = pd.DataFrame({'node_name': [node_name], 'value': [mem_spare], 'metric': ['mem_spare']})
            spare_mem = pd.concat([spare_mem, node_metric])
        
        return spare_mem.reset_index(drop=True)
    
    def get_pod_numbers(self,prometheus):
        query = 'count(kube_pod_info)'
        result = prometheus.custom_query(query)
        # pod_numbers = pd.DataFrame()
        return int(result[0]['value'][1])
    
    def get_node_numbers(self,prometheus):
        query = 'count(kube_node_info)'
        result = prometheus.custom_query(query)
        # node_numbers = pd.DataFrame()

        return int(result[0]['value'][1])


class OpenFaasPrometheusMetrics:
    def __init__(self):
        pass

    def get_function_qps(self,prometheus,function_name):
        query = f'avg_over_time(sum(irate(gateway_function_invocation_total{{function_name="{function_name}.cdgp"}}[1m]))[8s:])'
        result = prometheus.custom_query(query)
        if result:
            return float(result[0]['value'][1])
        
        return 0
    
    def get_max_function_exec_duration(self,prometheus,function_name):
        query = f'max((rate(gateway_functions_seconds_sum{{function_name="{function_name}.cdgp"}}[60s]) / rate(gateway_functions_seconds_count{{function_name="{function_name}.cdgp"}}[60s])))'
        result = prometheus.custom_query(query)
        if result:
            return float(result[0]['value'][1])
        return 0
    
    def get_average_function_exec_duration(self,prometheus,function_name):
        query = f'avg_over_time(rate(gateway_functions_seconds_sum{{function_name="{function_name}.cdgp"}}[60s]) / rate(gateway_functions_seconds_count{{function_name="{function_name}.cdgp"}}[60s])[5m:])'
        result = prometheus.custom_query(query)
        if result:
            return float(result[0]['value'][1])
        return 0
    
    def get_function_timeout_count(self,prometheus,function_name):
        query = f'sum(sum_over_time(rate(gateway_function_invocation_started{{function_name="{function_name}.cdgp",code!="200"}}[30s])[10s:]))'
        result = prometheus.custom_query(query)
        if result:
            return float(result[0]['value'][1])
        return 0
    