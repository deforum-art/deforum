import json
import os
from typing import List

from huggingface_hub import HfApi
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import RepoFile


def find_model_type(model_name):
    if os.path.exists(model_name):
        model_name = os.path.abspath(model_name)
        if os.path.isdir(model_name):
            model_name = os.path.join(model_name, "model_index.json")
        if not os.path.exists(model_name):
            raise ValueError(f"Model {model_name} not found")
        data = json.loads(open(model_name, "r").read())
        if "_class_name" in data:
            return data["_class_name"]
        else:
            raise ValueError(f"Model {model_name} does not have a _class_name field in model_index.json")

    api = HfApi()
    model_info: List[RepoFile] = list(api.list_files_info(model_name, ["model_index.json"]))
    if len(model_info) == 0:
        raise ValueError(f"Model {model_name} not found")
    model_info: RepoFile = model_info[0]
    file_path = hf_hub_download(model_name, model_info.rfilename)
    data = json.loads(open(file_path, "r").read())
    if "_class_name" in data:
        return data["_class_name"]
    else:
        raise ValueError(f"Model {model_name} does not have a _class_name field in model_index.json")
