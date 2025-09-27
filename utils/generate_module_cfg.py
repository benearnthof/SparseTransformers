"""
Parse a module file to extract a template config for all transformer arguments.
"""

import ast
import yaml
from pathlib import Path


def extract_cfg_keys(filepath:str):
    """
    example filepath: "/workspace/SparseTransformers/modules/llama.py"
    """
    code = Path(filepath).read_text()
    tree = ast.parse(code)

    cfg_keys = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "cfg":
                cfg_keys.add(node.attr)
            self.generic_visit(node)
    
    Visitor().visit(tree)

    return cfg_keys


def write_yaml_template(cfg_keys, output_path="/workspace/SparseTransformers/config/llama.yaml"):
    template = {key: None for key in sorted(cfg_keys)}
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False)


if __name__ == "__main__":
    cfg_keys = extract_cfg_keys("/workspace/SparseTransformers/modules/llama.py")
    write_yaml_template(cfg_keys)
    print("Extracted config keys:", cfg_keys)
