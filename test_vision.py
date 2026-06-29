"""
Quick test to verify the vision model loads and runs.
Usage: python test_vision.py path/to/chest_xray.jpg
"""
import sys
import json
from vision.inference import TBScreenModel

if len(sys.argv) < 2:
    print("Usage: python test_vision.py <image_path>")
    sys.exit(1)

model = TBScreenModel()
result = model.predict(sys.argv[1])
print(json.dumps(result, indent=2))
