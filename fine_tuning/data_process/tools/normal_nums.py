import os
import json
from collections import Counter

images = []
if os.path.exists("../labels/normal_category.json"):
    with open("../labels/normal_category.json", "r", encoding="utf-8") as f:
        old_data = json.load(f)

for item in old_data:
    images.append(item["images"][0])

print(len(images))
counts = Counter(images)
result = [num for num in images if counts[num] < 3]
print(result)

print(len(list(set(images))))
