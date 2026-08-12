import json
import os
import string
import random


def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def sort_key(entry):
    parts = entry["id"].rsplit("_", 1)
    index = parts[1]
    try:
        return 0, int(index)
    except ValueError:
        return 1, entry["id"]


def generate_random_string(length):
                                           
    characters = string.ascii_letters + string.digits
                                                                              
    random_string = "".join(random.choices(characters, k=length))
    return random_string


def write_list_of_dicts_to_file(filename, data, subdir=None):
    if subdir:
                                        
        os.makedirs(subdir, exist_ok=True)

                                             
        filename = os.path.join(subdir, filename)

                                                               
    with open(filename, "w") as f:
        for i, entry in enumerate(data):
                                                                                                            
            entry = make_json_serializable(entry)
            json_str = json.dumps(entry)
            f.write(json_str)
            if i < len(data) - 1:
                f.write("\n")


def make_json_serializable(value):
    if isinstance(value, dict):
                                                                                             
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
                                                                             
        return [make_json_serializable(item) for item in value]
    else:
                                                                                      
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


def write_dicts_to_file(filename, data, subdir=None):
    if subdir:
                                        
        os.makedirs(subdir, exist_ok=True)

                                             
        filename = os.path.join(subdir, filename)

                                                               
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
