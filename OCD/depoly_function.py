import os

class FunctionManager:
    def __init__(self, root_path):
        self.root_path = root_path
        self.functions = self.find_functions()

    def find_functions(self):
        function_dict = {}
        for root, dirs, _ in os.walk(self.root_path):
            path_parts = root.split(os.sep)
            if len(path_parts) - len(self.root_path.split(os.sep)) == 3:
                if path_parts[-1] == 'template':
                    continue
                function_name = path_parts[-1] + '-' + path_parts[-2].lower()
                function_path = os.sep.join(path_parts[:-1])
                function_dict[function_name] = function_path
        return function_dict

    def deploy_function(self, function_name, gateway_address):
        function_path = self.functions.get(function_name)
        if function_path:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            os.chdir(function_path)
            os.system(f"faas-cli deploy --filter {function_name} --gateway http://{gateway_address}:31112 -f config.yml")
        else:
            print(f"Function {function_name} not found in the available functions.")

    def delete_function(self, function_name, gateway_address):
        function_path = self.functions.get(function_name)
        if function_path:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            os.chdir(function_path)
            os.system(f"faas-cli delete --filter {function_name} --gateway http://{gateway_address}:31112 -f config.yml")
        else:
            print(f"Function {function_name} not found in the available functions.")

    def display_functions(self):
        for function_name, function_path in self.functions.items():
            print(f"{function_name}: {function_path}")

# Usage example:
# root_path = '../benchmark'
# manager = FunctionManager(root_path)
# manager.deploy_function('bert-21b-submod-19-me-21', '172.169.8.15')
