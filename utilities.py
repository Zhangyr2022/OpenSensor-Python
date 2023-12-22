
def convert_camel_to_snake(data):
    if isinstance(data, list):
        return [convert_camel_to_snake(item) for item in data]
    elif isinstance(data, dict):
        return {
            convert_camel_to_snake(key): convert_camel_to_snake(value)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        return data
    elif isinstance(data, int):
        return data
    elif isinstance(data, float):
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
