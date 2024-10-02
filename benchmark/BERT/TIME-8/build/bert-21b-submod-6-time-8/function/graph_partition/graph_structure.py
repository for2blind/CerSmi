import torch
from torch import fx
from typing import Dict, Tuple, List

from graph_partition.tracer import GPTracer

class SimpleWrapperGMInput:
    def __init__(self, data: List[Tuple[torch.Tensor]], device: str="cpu") -> None:
        self.data = data
        self.dtype = None
        self.device = device
        self.shape = torch.Size([1, len(data), 1])
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if (self.data[i][j].dtype == torch.float32 or self.data[i][j].dtype == torch.bfloat16 or self.data[i][j].dtype == torch.float16):
                    self.dtype = self.data[i][j].dtype
                    break
            if self.dtype:
                break
    
    def __len__(self):
        return len(self.data)
    
    def nelement(self):
        return sum((k.nelement() for k in j) if isinstance(j, (list, tuple)) else j.nelement() for i in self.data for j in i)
    
    def element_size(self):
        return 2 if self.dtype == torch.bfloat16 or self.dtype == torch.float16 else 4
    
    def to(self, args: any):
        if args == torch.float32:
            self.dtype = torch.float32
            self.data = [tuple([i.to(args) if i.dtype == torch.bfloat16 else i for i in j]) for j in self.data]
        elif args == torch.bfloat16:
            self.dtype = torch.bfloat16
            self.data = [tuple([i.to(args) if i.dtype == torch.float32 else i for i in j]) for j in self.data]
        elif args == torch.float16:
            self.dtype = torch.float16
            self.data = [tuple([i.to(args) if i.dtype == torch.float32 else i for i in j]) for j in self.data]
        else:
            self.device = args
            self.data = [tuple([i.to(args) if hasattr(i, 'to') else i for i in j]) for j in self.data]
            
        return self

class SimpleWrapperGMList(torch.nn.Module):
    def __init__(self, gm_list: List[fx.GraphModule]) -> None:
        super().__init__()
        self.gm_list = torch.nn.ModuleList(gm_list)
    
    def __len__(self):
        return len(self.gm_list)
    
    @torch.inference_mode()
    def forward(self, inputs: SimpleWrapperGMInput) -> any:
        
        for gm, inp in zip(self.gm_list, inputs.data):
            outputs = gm(*inp)
        return outputs
    
class InputsWrapper(dict):
    def __init__(self, inputs) -> None:
        super().__init__({'x': inputs})
        self.inputs = inputs
        self.dtype = None
        self.device = None
        self.shape = torch.Size([1, 2])
        if isinstance(inputs, dict) or hasattr(inputs, 'items'):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    self.dtype = v.dtype
                    self.device = v.device
                    if v.dtype == torch.float32 or v.dtype == torch.bfloat16 or v.dtype == torch.float16:
                        break
        elif isinstance(inputs, (list, tuple)):
            for v in inputs:
                if isinstance(v, torch.Tensor):
                    self.dtype = v.dtype
                    self.device = v.device
                    if v.dtype == torch.float32 or v.dtype == torch.bfloat16 or v.dtype == torch.float16:
                        break
        elif isinstance(inputs, torch.Tensor):
            self.dtype = inputs.dtype
            self.device = inputs.device
                
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
    
    def __len__(self):
        return 2
    
    def nelement(self):
        def unfold(x):
            if isinstance(x, (list, tuple)):
                return sum((unfold(i) for i in x))
            if isinstance(x, dict) or hasattr(x, 'items'):
                return sum((unfold(i) for i in x.values()))
            if isinstance(x, torch.Tensor):
                return x.nelement()
            return 1
         
        return unfold(self.inputs)
    
    def element_size(self):
        if self.dtype == torch.int8 or self.dtype == torch.qint8 or self.dtype == torch.bool:
            return 1
        if self.dtype == torch.int16 or self.dtype == torch.bfloat16 or self.dtype == torch.float16:
            return 2
        if self.dtype == torch.int32 or self.dtype == torch.qint32 or self.dtype == torch.complex32 or self.dtype == torch.float32:
            return 4
        if self.dtype == torch.float64 or self.dtype == torch.int64 or self.dtype == torch.complex64:
            return 8
        if self.dtype == torch.complex128:
            return 16
        return 1
    
    def to_limited(self, args, limited=None):
        if isinstance(self.inputs, dict) or hasattr(self.inputs, 'items'):
            self.inputs = {k: v.to(args) if not limited or v.dtype in limited else v for k, v in self.inputs.items()}
        elif isinstance(self.inputs, (list, tuple)):
            self.inputs = tuple(i.to(args) if not limited or i.dtype in limited else i for i in self.inputs)
        elif isinstance(self.inputs, torch.Tensor):
            self.inputs = self.inputs.to(args)
    
    def to(self, args: any):
        if args == torch.float32:
            self.dtype = args
            self.to_limited(args, [torch.float16, torch.bfloat16])
        elif args == torch.bfloat16 or args == torch.float16:
            self.dtype = args
            self.to_limited(args, [torch.float32])
        else:
            self.device = args
            self.to_limited(args)
        return self
    
class GraphsWrapper(torch.nn.Module):
    def __init__(self, model_list: List[torch.nn.Module]) -> None:
        super().__init__()
        self.model_list = torch.nn.ModuleList(model_list)
        self.n = len(model_list)
        self._tracer_cls = GPTracer
        if hasattr(self.model_list[0], "device"):
            self.device = self.model_list[0].device
        if hasattr(self.model_list[0], "config"):
            self.config = self.model_list[0].config
    
    def __len__(self):
        return len(self.n)
    
    def forward(self, x: InputsWrapper) -> Dict[str, torch.Tensor]:
        extra_outputs = {}
        for i in range(self.n):
            y = self.model_list[i](**x)
            x = y['base_outputs']
            extra_outputs.update(y['extra_outputs'])
            y = None
        return {'base_outputs': x, 'extra_outputs': extra_outputs}


class ScriptGraphsWrapper(torch.nn.Module):
    __constants__ = ['model_list']
     
    def __init__(self, model_list: List[torch.nn.Module]) -> None:
        super().__init__()
        self.model_list = torch.nn.ModuleList(model_list)
        self.n = len(model_list)
        self._tracer_cls = GPTracer
        if hasattr(self.model_list[0], "device"):
            self.device = self.model_list[0].device
        if hasattr(self.model_list[0], "config"):
            self.config = self.model_list[0].config
    
    def __len__(self):
        return len(self.n)
    
    def forward(self, x) -> Tuple[torch.Tensor]:
        extra_outputs = []
        for i in range(self.n):
            x, y = self.model_list[i](*x)
            extra_outputs += list(y)
            y = None
        return tuple(extra_outputs)
    