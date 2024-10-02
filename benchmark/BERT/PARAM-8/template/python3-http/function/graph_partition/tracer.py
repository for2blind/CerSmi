import torch
import inspect
from typing import List, Optional, Tuple, Type, Dict
from transformers.utils.fx import HFTracer

class GPTracer(HFTracer):
    def __init__(self):
        super().__init__()
        
    # def trace(self, model: torch.nn.Module, *args, **kwargs) -> torch.fx.Graph:
    #     # wrap model for HFTracer
    #     if kwargs.get('device'):
    #         model.device = kwargs['device']
    #         del kwargs['device']
    #     if kwargs.get('config'):
    #         model.config = kwargs['config']
    #         del kwargs['config']
    #     return super().trace(model, *args, **kwargs)

    def to_bool(self, obj: torch.fx.Proxy) -> bool:
        return False
    
def get_concrete_args(model: torch.nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = (
            input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        )
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }


def symbolic_trace(
    model: torch.nn.Module,
    input_names: Optional[List[str]] = None,
    dummy_inputs: Optional[Dict[str, any]] = None,
    tracer_cls: Type[HFTracer] = GPTracer,
) -> torch.fx.GraphModule:
    """input_names or dummy_inputs must be provided

    Args:
        model (torch.nn.Module): model to be traced
        input_names (Optional[List[str]], optional): input names to be traced. Defaults to None.
        dummy_inputs (Optional[Dict[str, any]], optional): dummy inputs used to trace. Defaults to None.
        tracer_cls (Type[HFTracer], optional): class of tracer. Defaults to HFTracer.

    Returns:
        torch.fx.GraphModule: traced model
    """

    tracer = tracer_cls()

    if dummy_inputs is None:
        if input_names is None:
            input_names = model.dummy_inputs.keys()

        input_names = list(input_names)
        concrete_args = get_concrete_args(model, input_names)
        traced_graph = tracer.trace(model, concrete_args=concrete_args)
    else:
        traced_graph = tracer.trace(model, dummy_inputs=dummy_inputs)

    traced = torch.fx.GraphModule(model, traced_graph)

    if hasattr(model, "config"):
        traced.config = model.config
    traced.class_for_deserialization = model.__class__
    if hasattr(model, "device"):
        traced.device = model.device

    return traced