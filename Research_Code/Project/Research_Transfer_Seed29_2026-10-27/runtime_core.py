"""Supply a new frozen registry to an isolated instance of the unchanged native runtime."""

import importlib.util
import sys

from common import ENGINE, HERE, PRIOR, read, sha

sys.path.insert(0, str(ENGINE))
source = PRIOR / "runtime_core.py"
frozen = read(HERE / "Plan.json")
assert sha(source) == frozen["runtime_origin_sha256"]
spec = importlib.util.spec_from_file_location("seed29_immutable_native_runtime", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
# The legacy runtime resolves its registry from HERE. This is a private module
# instance in this worker process; no source file or other worker is modified.
module.HERE = HERE
Runtime = module.Runtime
