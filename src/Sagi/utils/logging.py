import json
import logging


# format json string in log record
class JsonLogRecord(logging.LogRecord):
    @staticmethod
    def is_json_string(s: str) -> bool:
        """
        Check if the input is a valid JSON string.
        """
        if not isinstance(s, str):
            return False
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    def _recursive_parse(self, obj):
        """
        Recursively parse any nested JSON strings in the parsed object.

        - If the object is a dictionary, process each value recursively.
        - If the object is a list, process each element recursively.
        - If the object is a string and is valid JSON, parse it and then process the result recursively.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_obj[k] = self._recursive_parse(v)
            return new_obj
        elif isinstance(obj, list):
            return [self._recursive_parse(item) for item in obj]
        elif isinstance(obj, str) and JsonLogRecord.is_json_string(obj):
            try:
                parsed = json.loads(obj)
                return self._recursive_parse(parsed)
            except json.JSONDecodeError:
                return obj
        else:
            return obj

    def getMessage(self) -> str:
        """
        Override the getMessage method:

        - If the original message is a JSON string, parse it and recursively format any nested JSON strings.
        - Return the formatted JSON string with indentation.
        - If the message is not valid JSON, simply return the original message.
        """
        msg = super().getMessage()
        if JsonLogRecord.is_json_string(msg):
            try:
                parsed = json.loads(msg)
                parsed = self._recursive_parse(parsed)
                return json.dumps(parsed, indent=4, ensure_ascii=False)
            except Exception:
                # In case of any error during parsing, return the original message.
                return msg
        else:
            return msg


def format_json_string_factory(*args, **kwargs):
    return JsonLogRecord(*args, **kwargs)
