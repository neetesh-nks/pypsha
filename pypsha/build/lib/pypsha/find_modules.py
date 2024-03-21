import os
from importlib.util import find_spec
import glob

def _contains_jar(path):
    jar_file = os.path.join(path, "IM_EventSetCalc_v3_0_ASCII.jar")
    jar_file_matches = glob.glob(jar_file)
    if len(jar_file_matches) > 0:
        return jar_file_matches[0]
    else:
        return None

def _find_java_module():
    try:
        module_home = os.path.dirname(find_spec("pypsha").origin)
        jar_path = _contains_jar(os.path.join(module_home,"jars"))
        if jar_path != None:
            return jar_path
    except Exception as e:
        pass
    print("Could not find valid jar in current environment.")


if __name__ == "__main__":
  print(_find_java_module())