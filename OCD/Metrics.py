from prometheus_api_client import PrometheusConnect
import datetime, time
import pandas as pd
from kubernetes import client, config, watch
import json
import os
import sys
config.load_kube_config()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
v1 = client.CoreV1Api()
v1_app = client.AppsV1Api()
api = client.CustomObjectsApi()
v1beta1 = client.ExtensionsV1beta1Api()

ret = v1.list_node(watch=False)
node_ip = {}
for i in ret.items:
    node_ip[i.status.addresses[0].address] = i.metadata.name

class MetricCollector:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://172.169.8.86:30500")
        self.columns = ['timestamp','value','metrics', 'instance','ip']
        self.columns_gpu = ['timestamp','value','metrics', 'instance','gpu','modelName']
        self.columns_pod = ['timestamp','value','metrics','function']
        self.mdict_node = {
            'node_cpu_util':'instance:node_cpu_utilisation:rate5m',
            'node_mem_util':'instance:node_memory_utilisation:ratio',
            'node_gpu_util':'avg(irate(DCGM_FI_DEV_GPU_UTIL{kubernetes_node!="", modelName!=""}[1m]) * 100) by (gpu, kubernetes_node, modelName)'
        }
        self.mdict_pod = {
            'function_mem_util':'sum(container_memory_rss+container_memory_cache+container_memory_usage_bytes+container_memory_working_set_bytes{namespace="openfaas-fn",pod=~"{Function}.*"})/sum(cluster:namespace:pod_memory:active:kube_pod_container_resource_limits{namespace="openfaas-fn",container=~"{Function}.*"})*100',
            'function_cpu_util':"sum(rate(container_cpu_usage_seconds_total{namespace='openfaas-fn',container=~'{Function}.*'}[30s]))/sum(kube_pod_container_resource_limits{namespace='openfaas-fn',container='{Function}',resource='cpu',service!=''})*100"
        }

        self.mdata_pod = pd.DataFrame(columns=self.columns_pod)
        self.MS_pod_resource = pd.DataFrame(columns=self.columns_pod)

        self.get_Pod_metrics()
        self.node_metrics, self.gpu_metrics = self.update_metrics(list(self.mdict_node.keys()))
        

    def get_current_Node_metrics(self):
        # if self.mdata_node is empty or lated than 5s, update the metrics
        if self.mdata_node.empty or (time.time() - self.mdata_node['timestamp'].max()) > 2:
            self.update_metrics(list(self.mdict_node.keys()))
        return self.mdata_node, self.mdata_gpu
    
    def get_Pod_metrics(self):
        # if self.mdata_node is empty or lated than 5s, update the metrics
        if self.mdata_pod.empty or (time.time() - self.mdata_pod['timestamp'].max()) > 2:
            self.fetch_pod_metrics()
        return self.mdata_pod
    
    def fetch_pod_metrics(self,range=2):
        end_time = time.time()
        start_time = end_time - range*60
        
        deployments = v1_app.list_namespaced_deployment(namespace='openfaas-fn', watch=False)
        function = [i.metadata.name for i in deployments.items]
        df_all = pd.DataFrame(columns=self.columns_pod)

        for f in function:
            df_m = pd.DataFrame(columns=self.columns_pod)
            for m in self.mdict_pod.keys():
                df_f = pd.DataFrame(columns=self.columns_pod)
                query_result = self.prom.custom_query_range(self.mdict_pod[m].replace('{Function}',f),start_time=datetime.datetime.fromtimestamp(start_time),end_time=datetime.datetime.fromtimestamp(end_time),step=10)
                for q in query_result:
                    data_t = pd.json_normalize(q)
                    for i,r in data_t.iterrows():
                        df_values = pd.DataFrame(r['values'], columns=['timestamp','value'])
                        df_values['function'] = f
                        df_values['metrics'] = m
                        df_f = pd.concat([df_f,df_values],axis=0)
                df_m = pd.concat([df_m,df_f],axis=0)
            df_all = pd.concat([df_all,df_m],axis=0)
        self.mdata_pod = df_all
        return self.mdata_pod
        


        
    def get_single_metric(self, metric_name, end_time, range=2):

        start_time = datetime.datetime.fromtimestamp(end_time - range * 60)
        end_time = datetime.datetime.fromtimestamp(end_time)

        query_result = self.prom.custom_query_range(self.mdict_node[metric_name], end_time=end_time, start_time=start_time, step=20)
        Mdata_node = pd.DataFrame(columns=self.columns)
        Mdata_GPU = pd.DataFrame(columns=self.columns)

        if metric_name == 'node_gpu_util':
            for m in query_result:
                data_t = pd.json_normalize(m)
                dk_tmp = pd.DataFrame(columns=self.columns_gpu)
                for i,r in data_t.iterrows():
                    df_values = pd.DataFrame(r['values'], columns=['timestamp','value'])
                    df_values['gpu'] = r['metric.gpu']
                    df_values['modelName'] = r['metric.modelName']
                    df_values['instance'] = r['metric.kubernetes_node']
                    dk_tmp = pd.concat([dk_tmp,df_values],axis=0)
                dk_tmp['metrics'] = metric_name
            Mdata_GPU = pd.concat([Mdata_GPU,dk_tmp],axis=0)

        # cpu based metrics
        else:
            for m in query_result:
                data_t = pd.json_normalize(m)
                dk_tmp = pd.DataFrame(columns=self.columns)
                for i,r in data_t.iterrows():
                    df_values = pd.DataFrame(r['values'], columns=['timestamp','value'])
                    dk_tmp = pd.concat([dk_tmp,df_values],axis=0)
                dk_tmp['metrics'] = metric_name
                dk_tmp['ip'] = m['metric']['instance'].split(':')[0]
                Mdata_node = pd.concat([Mdata_node,dk_tmp],axis=0)
                Mdata_node['instance'] = Mdata_node['ip'].apply(lambda x: node_ip[x])
        return Mdata_node, Mdata_GPU

    def update_metrics(self, metric_list, range=2):
        pd_node = pd.DataFrame(columns=self.columns)
        pd_gpu = pd.DataFrame(columns=self.columns_gpu)
        for m in metric_list:
            Mdata_node, Mdata_GPU = self.get_single_metric(m, time.time(), range)
            pd_node = pd.concat([pd_node,Mdata_node],axis=0)
            pd_gpu = pd.concat([pd_gpu,Mdata_GPU],axis=0)
        self.mdata_node = pd_node
        self.mdata_gpu = pd_gpu
        return self.mdata_node, self.mdata_gpu
    
    def update_MS_pod_resource(self, MS_pod_resource):
        # insert the new data and keep only data within 2 minutes
        self.MS_pod_resource = pd.concat([self.MS_pod_resource,MS_pod_resource],axis=0)
        self.MS_pod_resource = self.MS_pod_resource[self.MS_pod_resource['timestamp'] > time.time() - 120]
        return self.MS_pod_resource

    # just return average value
    def fetch_pod_metric_by_metricserver(self):
        namespace = "openfaas-fn"
        deployments = v1_app.list_namespaced_deployment(namespace='openfaas-fn')
        ms_res = pd.DataFrame()
        for deployment in deployments.items:
            selector = deployment.spec.selector.match_labels
            # change the selector to string
            selector = ",".join(["%s=%s" % (key, value) for (key, value) in selector.items()])
            # Get the list of pods for the deployment
            pod_list = v1.list_namespaced_pod(namespace, label_selector=selector)
            cpu_usage = 0
            memory_usage = 0
            for pod in pod_list.items:
                resp = ""
                try:
                    resp = api.get_namespaced_custom_object("metrics.k8s.io", "v1beta1", namespace, "pods", pod.metadata.name)
                except:
                    continue
                c = resp["containers"][1]["usage"]["cpu"]
                m = resp["containers"][1]["usage"]["memory"]
                if c.endswith("n"):
                    c = c.rstrip("n")
                    c = int(c)/10**9
                elif c.endswith("u"):
                    c = c.rstrip("u")
                    c = int(c)/10**6
                cpu_usage += c
                if m.endswith("Mi"):
                    m = m.rstrip("Mi")
                    m = int(m)
                elif m.endswith("Ki"):
                    m = m.rstrip("Ki")
                    m = int(m)/2**10
                memory_usage += m
            if resp == "":
                continue
            timestamp = datetime.datetime.strptime(resp["timestamp"],"%Y-%m-%dT%H:%M:%SZ").timestamp()
            cpu_usage = cpu_usage/len(pod_list.items) *100
            memory_usage = memory_usage/len(pod_list.items)
            # ms_res = ms_res.append({'timestamp':timestamp,'value':cpu_usage,'metrics':'pod_cpu','function':deployment.metadata.name},ignore_index=True)
            ms_res = pd.concat([ms_res,pd.DataFrame({'timestamp':[timestamp],'value':[cpu_usage],'metrics':['pod_cpu'],'function':[deployment.metadata.name]})],axis=0)
            ms_res = pd.concat([ms_res,pd.DataFrame({'timestamp':[timestamp],'value':[memory_usage],'metrics':['pod_mem'],'function':[deployment.metadata.name]})],axis=0)
        
        self.update_MS_pod_resource(ms_res)
        return self.MS_pod_resource

class OpenFaasPrometheusMetrics:
    def __init__(self):
        self.prom_openfaas = PrometheusConnect(url="http://172.169.8.86:31113")
        self.columns = ['timestamp','value','metrics','function_name']
        self.mdict_function = {
            'function_request':'irate(gateway_function_invocation_started[60s])',
            'response_time':'irate(gateway_functions_seconds_sum{code="200"}[10s])',
        }
        self.fMetric = {"updated":time.time()}
    
    def get_colunms(self):
        return self.columns
    
    def get_funtion_metrics(self):
        for m in self.mdict_function:
            self.fMetric[m] = self.get_single_metric(m, time.time())
        self.fMetric['updated'] = time.time()
        return self.fMetric
    
    def get_single_metric(self, metric_name, end_time, range=0.5):
        start_time = datetime.datetime.fromtimestamp(end_time - range * 60)
        end_time = datetime.datetime.fromtimestamp(end_time)
        query_result = self.prom_openfaas.custom_query_range(self.mdict_function[metric_name], end_time=end_time, start_time=start_time, step=5)
        Req_data_list = pd.DataFrame(columns=self.columns)
        for m in query_result:
            df_values = pd.DataFrame(pd.json_normalize(m)['values'].sum(), columns=['timestamp','value'])
            df_values['function_name'] = m['metric']['function_name']
            df_values['metrics'] = metric_name
            Req_data_list = pd.concat([Req_data_list,df_values],axis=0)
        # fill nan with 0 in value column and return
        Req_data_list.replace("NaN", 0, inplace=True)
        # value is string, convert to float
        Req_data_list['value'] = Req_data_list['value'].astype(float)
        Req_data_list['function_name'] = Req_data_list['function_name'].apply(lambda x: x.split('.')[0])
        return Req_data_list

    
    
    def get_req_queue(self):
        # if self.function_queue is None or self.function_queue out of 2s: update
        if len(self.fMetric) == 1 or self.fMetric['updated'] < time.time() - 2:
            self.get_funtion_metrics()
        return self.fMetric['function_request']
    
    def get_req_fqueue(self, function_name):
        fqueue = self.get_req_queue()
        # return 99th of last 5s
        return max(fqueue[fqueue['function_name'] == function_name]['value'])
    
    def get_queue_time(self):
        # if self.function_queue is None or self.function_queue out of 2s: update
        if len(self.fMetric) == 1 or self.fMetric['updated'] < time.time() - 2:
            self.get_funtion_metrics()
        return self.fMetric['response_time']
    
    def get_request_length(self,function_name):
        fqueue = self.get_queue_time()
        # return 99th of last 1m
        if function_name == "all":
            return pd.DataFrame(fqueue.groupby('function_name')['value'].sum()).reset_index()
        return fqueue[fqueue['function_name'] == function_name]['value'].sum()


    def get_req_fqueue_time(self, function_name):
        fqueue = self.get_queue_time()
        # return 99th of last 1m
        return fqueue[fqueue['function_name'] == function_name]['value'].avg()
    


class FunctionConfig:
    def __init__(self):
        self.flimit = {"updated":time.time()}
        self.fcapacity = {"updated":time.time()}
        with open('gss.json') as f:
            self.cjson = json.load(f)
        # init all function flimit and fcapacity
        self.get_fcapacity()

    def convert_to_bytes(self,memory_string):
        if memory_string.endswith("Mi"):
            memory_value = float(memory_string[:-2]) * 1
        elif memory_string.endswith("Gi"):
            memory_value = float(memory_string[:-2]) * 1024
        else:
            raise ValueError("Invalid memory string format")
        return memory_value
    
    # get limit cpu, memory, gpu from deployment by k8s api
    def get_function_limit(self):
        # get all deployment in namespace openfaas-fn
        ret = v1_app.list_namespaced_deployment(namespace='openfaas-fn')
        for i in ret.items:
            # get limit cpu, memory, gpu
            resource = i.spec.template.spec.containers[0].resources
            if resource is None:
                continue
            limit_cpu = resource.limits['cpu']
            limit_memory = resource.limits['memory']
            limit_gpu = 0 

            # test if i.spec.template.spec.containers[0].resources.limits have key 'nvidia.com/gpu'
            if 'nvidia.com/gpu' in resource.limits:
                limit_gpu = resource.limits['nvidia.com/gpu']
            replicas = i.spec.replicas
            function_name = i.metadata.name
            # get replicas of function
            replicas = i.status.replicas
            # get function config
            config = {'limit_cpu':float(limit_cpu.replace("m",""))/1000, 'limit_memory':self.convert_to_bytes(limit_memory), 'limit_gpu':int(limit_gpu),'parallelism':self.cjson['function']['max_inflight'],'replicas':replicas,'fork_factor':self.cjson['function']['fork_factor'],'replicas':replicas}
            self.flimit[function_name] = config
        self.flimit['updated'] = time.time()
        return self.flimit

    def get_function_replicas(self, function_name):
        # if out of 2s, update
        if self.flimit['updated'] < time.time() - 2:
            self.get_function_limit()
        # if self.functions[function_name]['replicas'] is None: only consider gpu memory. 
        return self.flimit[function_name]['replicas']

    
    def get_fcapacity(self):
        # update self.flimit
        self.flimit = self.get_function_limit()

        # if self.functions[function_name]['limit_gpu'] is None: only consider gpu memory. 
        for f in self.flimit:
                if f == 'updated':
                    continue
                self.fcapacity[f] = {"gpu_memory":self.flimit[f]['limit_gpu'] * 32 * self.flimit[f]['fork_factor'] * self.flimit[f]['replicas']*1024,
                                     "cpu":self.flimit[f]['limit_cpu'] * self.flimit[f]['parallelism'] * self.flimit[f]['fork_factor'] * self.flimit[f]['replicas'],
                                     "memory":self.flimit[f]['limit_memory'] * self.flimit[f]['fork_factor'] * self.flimit[f]['replicas']
                                     }
        self.fcapacity['updated'] = time.time()
        return self.fcapacity
    
    
    # return current capacity of function gpu_memory and cpu.
    def get_function_capacity(self, function_name):
        # if self.fcapacity is None or self.fcapacity out of 5s: update
        if len(self.fcapacity) == 0 or self.fcapacity['updated'] < time.time() - self.cjson['function']['capacity_timeout']:
                self.get_fcapacity() 
        return self.fcapacity[function_name]



if __name__ == "__main__":
    mc = MetricCollector()