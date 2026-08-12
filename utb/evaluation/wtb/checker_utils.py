import json
import re

from difflib import SequenceMatcher

try:
    import jieba
except ImportError:
    jieba = None

try:
    from rouge import Rouge
except ImportError:
    Rouge = None


def _normalize_str(s):
              
    return "".join(s.split()).lower()


def get_similarity(a, b):
                              
    return SequenceMatcher(None, a, b).ratio()


class ToolArgsChecker:
                                                             
          
                                                             
    CORRECT = "correct"

            
    ERROR_ARGS_JSON_DECODE = "error: args invalid json format"

                                 
    SCHEMA_ERROR_MISSING = "error_schema: required args missing"
    SCHEMA_ERROR_UNDEFINED = "error_schema: args not defined"
    SCHEMA_ERROR_TYPE = "error_schema: args type inconsistent"
    SCHEMA_ERROR_ENUM = "error_schema: args value not in enum"
    SCHEMA_ERROR_NESTED_MISSING = "error_schema: required nested args missing"
    SCHEMA_ERROR_NESTED_UNDEFINED = "error_schema: nested args not defined"
    SCHEMA_ERROR_NESTED_TYPE = "error_schema: nested args type inconsistent"
    SCHEMA_ERROR_NESTED_ENUM = "error_schema: nested args value not in enum"

                              
    MATCH_ERROR_KEYS_MISMATCH = "error_match: args keys mismatch"
    MATCH_ERROR_ARRAY_LENGTH = "error_match: array length mismatch"
    MATCH_ERROR_TYPE_INCONSISTENT = "error_match: value type inconsistent"
    MATCH_ERROR_VALUE_MISMATCH = "error_match: value mismatch"
    MATCH_ERROR_STR_SIMILARITY = "error_match: string similarity too low"
    MATCH_ERROR_STR_EMPTY = "error_match: string value mismatch (empty)"
    MATCH_ERROR_ROUGE_FAIL = "error_match: rouge calculation failed"
    MATCH_ERROR_EXCEPTION = "error_match: comparison exception"

                                                             
                              
                                                             
    JSON_TO_PYTHON_TYPES = {
        "string": [str],
        "integer": [int],
        "float": [float],
        "number": [int, float],
        "boolean": [bool],
        "array": [list],
        "object": [dict],
        "null": [type(None)]
    }

    def __init__(self):
        self._full_tool_schemas = {}
        self.rouge = Rouge() if Rouge is not None else None
        pass

    def _is_contains_chinese(self, text):
        """判断字符串是否包含中文字符"""
        return bool(re.search(r'[\u4e00-\u9fa5]', text))

    def _resolve_valid_types(self, prop_info):
        valid_types = []
        if "anyOf" in prop_info:
            for sub_rule in prop_info["anyOf"]:
                valid_types.extend(self._resolve_valid_types(sub_rule))
        elif "type" in prop_info:
            type_name = prop_info["type"]
            python_types = self.JSON_TO_PYTHON_TYPES.get(type_name, [])
            valid_types.extend(python_types)
        else:
            valid_types.extend(self.JSON_TO_PYTHON_TYPES["string"])
        return valid_types

    def _parse_tool_definitions(self, tools):
        tool_schemas = {}
        for tool in tools:
            try:
                func_def = tool.get("function", {})
                tool_name = func_def.get("name")
                if not tool_name:
                    continue
                parameters = func_def.get("parameters", {})
                required_params = set(parameters.get("required", []))
                param_schemas_map = {}
                properties = parameters.get("properties", {})
                for param_name, prop_info in properties.items():
                    param_schemas_map[param_name] = prop_info
                tool_schemas[tool_name] = {
                    "required": required_params,
                    "param_schemas": param_schemas_map
                }
            except Exception:
                               
                print(f"Tool definition error: {json.dumps(tool, ensure_ascii=False)}")
        self._full_tool_schemas = tool_schemas
        return tool_schemas

    def _recursive_arg_check(self, arg_value, schema, path=""):
                        
        if "anyOf" in schema:
            errors = []
            for sub_schema in schema["anyOf"]:
                                
                err = self._recursive_arg_check(arg_value, sub_schema, path)
                if err is None:
                    return None                    
                errors.append(err)

                                                 
            return errors[0]

                 
        allowed_types = self._resolve_valid_types(schema)

        if allowed_types and not isinstance(arg_value, tuple(allowed_types)):
            return self.SCHEMA_ERROR_TYPE if path == "" else self.SCHEMA_ERROR_NESTED_TYPE

                 
        if "enum" in schema and arg_value not in schema["enum"]:
            return self.SCHEMA_ERROR_ENUM if path == "" else self.SCHEMA_ERROR_NESTED_ENUM

                               
        if isinstance(arg_value, dict):
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

                   
            missing = required - set(arg_value.keys())
            if missing:
                return self.SCHEMA_ERROR_MISSING if path == "" else self.SCHEMA_ERROR_NESTED_MISSING

                        
            for k, v in arg_value.items():
                if k not in properties:
                                                                      
                    if schema.get("additionalProperties") is True:
                        continue

                             
                    return self.SCHEMA_ERROR_UNDEFINED if path == "" else self.SCHEMA_ERROR_NESTED_UNDEFINED

                new_path = f"{path}.{k}" if path else k
                err = self._recursive_arg_check(v, properties[k], new_path)
                if err:
                    return err

                     
        if isinstance(arg_value, list) and "items" in schema:
            item_schema = schema["items"]
            for i, item in enumerate(arg_value):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                err = self._recursive_arg_check(item, item_schema, new_path)
                if err:
                    return err

        return None        

    def _recursive_compare(self, predict_val, answer_val, schema_info, path=""):
        """
        递归对比两个值，并返回包含路径的错误信息。
        """
                     
        path_str = f" at '{path}'" if path else ""

        if type(predict_val) != type(answer_val):
            return f"{self.MATCH_ERROR_TYPE_INCONSISTENT}{path_str}"

                              
        if isinstance(answer_val, dict):
            predict_keys = set(predict_val.keys())
            answer_keys = set(answer_val.keys())

                              
            if predict_keys != answer_keys:
                missing_keys = answer_keys - predict_keys
                extra_keys = predict_keys - answer_keys

                detail = []
                if missing_keys:
                    detail.append(f"missing: {list(missing_keys)}")
                if extra_keys:
                    detail.append(f"extra: {list(extra_keys)}")

                return f"{self.MATCH_ERROR_KEYS_MISMATCH}{path_str} ({', '.join(detail)})"

            props = schema_info.get("properties", schema_info) if isinstance(schema_info, dict) else {}
            for key in answer_val:
                sub_schema = props.get(key, {})
                                                 
                new_path = f"{path}.{key}" if path else key
                res = self._recursive_compare(predict_val[key], answer_val[key], sub_schema, new_path)
                if res != self.CORRECT:
                    return res
            return self.CORRECT

                             
        elif isinstance(answer_val, list):
            if len(predict_val) != len(answer_val):
                return f"{self.MATCH_ERROR_ARRAY_LENGTH}{path_str}"

            item_schema = schema_info.get("items", {})
            for i, (p_item, a_item) in enumerate(zip(predict_val, answer_val)):
                new_path = f"{path}[{i}]"
                res = self._recursive_compare(p_item, a_item, item_schema, new_path)
                if res != self.CORRECT:
                    return res
            return self.CORRECT

                         
        elif isinstance(answer_val, str):
                        
            if predict_val == answer_val:
                return self.CORRECT

                                  
            if _normalize_str(predict_val) == _normalize_str(answer_val):
                return self.CORRECT

                           
            if len(answer_val) < 10:             
                similarity = get_similarity(predict_val, answer_val)
                if similarity < 0.8:
                    return f"{self.MATCH_ERROR_STR_SIMILARITY}{path_str} (edit score: {similarity:.2f})"

                                   
            if self._is_contains_chinese(answer_val) or self._is_contains_chinese(predict_val):
                if jieba is not None:
                    predict_proc = " ".join(jieba.cut(predict_val))
                    answer_proc = " ".join(jieba.cut(answer_val))
                else:
                    predict_proc = " ".join(predict_val)
                    answer_proc = " ".join(answer_val)
            else:
                predict_proc, answer_proc = predict_val, answer_val

            try:
                if self.rouge is not None:
                    scores = self.rouge.get_scores(predict_proc, answer_proc)
                    similarity = scores[0]["rouge-l"]["f"]
                else:
                    similarity = get_similarity(predict_proc, answer_proc)
                if similarity < 0.7:
                    return f"{self.MATCH_ERROR_STR_SIMILARITY}{path_str} (rouge score: {similarity:.2f})"
            except:
                return f"{self.MATCH_ERROR_ROUGE_FAIL}{path_str}"
            return self.CORRECT

                      
        else:
            if predict_val != answer_val:
                return f"{self.MATCH_ERROR_VALUE_MISMATCH}{path_str}"
            return self.CORRECT

    def tool_check(self, tools, func_name, func_args_str):
                          
        registered_tools = self._parse_tool_definitions(tools)

        tool_schema = registered_tools[func_name]
                     
        function_params_schema = {"type": "object",
                                  "required": list(tool_schema["required"]),
                                  "properties": tool_schema["param_schemas"]}

                       
        try:
            func_args = json.loads(func_args_str)
        except json.JSONDecodeError:
            return self.ERROR_ARGS_JSON_DECODE

                              
        error = self._recursive_arg_check(func_args, function_params_schema, path="")
        if error:
            return error

        return self.CORRECT

    def answer_check(self, tools, predict_func_name, predict_func_args_str, answer_func_args_str):
        """
                比较预测参数和标准答案参数。
                返回 CORRECT 或 错误描述字符串。
                """
                          
        registered_tools = self._parse_tool_definitions(tools)

        tool_schema = registered_tools[predict_func_name]
        param_schemas = tool_schema["param_schemas"]

                       
        try:
            predict_args = json.loads(predict_func_args_str)
            answer_args = json.loads(answer_func_args_str)
        except json.JSONDecodeError:
            return self.ERROR_ARGS_JSON_DECODE

                   
                                                         
        try:
            return self._recursive_compare(predict_args, answer_args, param_schemas)
        except Exception as e:
            return f"error: comparison failed - {str(e)}"

    def check(self, tools, predict_func_name, predict_func_args_str, answer_func_args_str):
                                                  
        tool_status = self.tool_check(tools, predict_func_name, predict_func_args_str)
        if tool_status != self.CORRECT:
            return tool_status

                                                 
        answer_status = self.answer_check(tools, predict_func_name, predict_func_args_str, answer_func_args_str)
        if answer_status != self.CORRECT:
            return answer_status

        return self.CORRECT
