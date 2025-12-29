import os
import json
import re





# print(old_data)
# print(len(old_data))


# for item in old_data:
#     message = item.get("messages")
#     for chat in message:
#         if chat["role"] == "assistant":
#             content = chat.get("content", None)
#             if "0." in content:
#                 old_data.remove(item)


# new_data = [
#     item for item in old_data
#     if not (
#         item.get("images")  # 确保 images 存在
#         and len(item["images"]) > 0  # 确保 images 非空
#         and "_铺粉_c" in item["images"][0]  # 检查第一个元素是否包含目标字符串
#     )
# ]
#
# print(len(new_data))
# with open("../labels/multi_model_label_category.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False)
file_names = [
             "../labels/corrective.json",
              "../labels/corrective_category.json",
              "../labels/multi_model_label_category.json",
              "../labels/multi_model_label.json",
              "../labels/normal_category.json",
              "../labels/normal.json",
              "../labels/withoutImage_category.json",
              "../labels/withoutImage.json"
              ]
for file in file_names:
    with open(file, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    for item in old_data:
        message = item.get("messages")
        for chat in message:
            if chat["role"] == "assistant":
                content = chat.get("content", None)
                # content = re.sub(r'（注：.*?）', '', content, flags=re.DOTALL)
                # content = re.sub(r'模板一：', '', content, flags=re.DOTALL)
                # content = re.sub(r'模板二：', '', content, flags=re.DOTALL)
                # content = re.sub(r'模板三：', '', content, flags=re.DOTALL)
                # content = re.sub(r'模板四：', '', content, flags=re.DOTALL)
                # content = re.sub(r'模板', '', content, flags=re.DOTALL)
                # content = re.sub(r'\n.*?[（(].*?技术型.*?[)）]', '', content,flags=re.DOTALL)
                # content = re.sub(r"【定位】.*?(?=【特征描述】)", "", content, flags=re.DOTALL  )
                # content = re.sub(r"【定位】.*?(?=【图片特征分析】)", "", content, flags=re.DOTALL  )
                # content = re.sub(r"【定位】.*?(?=【污染物特征描述】)", "", content, flags=re.DOTALL  )
                content = re.sub(r"\n\s*位置：.*?(?=\n)", "", content, flags=re.DOTALL  )
                content = re.sub(r"\n\s*-\s*位置：[^\n]*", "", content, flags=re.DOTALL  )
                content = re.sub(r"\n\s*-\s*分布位置：[^\n]*", "", content, flags=re.DOTALL  )
                content = re.sub(r"\n\s*-\s*位置区域：[^\n]*", "", content, flags=re.DOTALL  )
                # content = re.sub(r"和近中心带状区域", "", content, flags=re.DOTALL  )
                # content = re.sub(r"近中心带状区域和", "", content, flags=re.DOTALL  )
                # content = re.sub(r"和边缘连续区", "", content, flags=re.DOTALL  )
                # content = re.sub(r"边缘连续区和", "", content, flags=re.DOTALL  )
                # content = re.sub(r"和边缘连续区", "", content, flags=re.DOTALL  )

                chat["content"] = content
    print(len(old_data))
    file_new = "../labels/new_"+file.split("/")[-1]
    with open(file_new, "w", encoding="utf-8") as f:
        json.dump(old_data, f, ensure_ascii=False)