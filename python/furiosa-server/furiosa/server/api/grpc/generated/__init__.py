import sys
from pathlib import Path

# FIXME: Add PYTHONPATH for GRPC issue(https://github.com/protocolbuffers/protobuf/issues/1491)
sys.path.append(str(Path(__file__).parent))
