import os
import json
import re

def yolo_to_abs(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    # x_center_abs = x_center * img_width
    # y_center_abs = y_center * img_height
    # w_abs = w * img_width
    # h_abs = h * img_height
    # x_min = x_center_abs - w_abs / 2
    # y_min = y_center_abs - h_abs / 2
    # x_max = x_center_abs + w_abs / 2
    # y_max = y_center_abs + h_abs / 2
    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)
    return str(f"{x_min},{y_min},{x_max} ,{y_max}")

# pattern = r"[0-9].png"
if os.path.exists("../backups/multi_model_label_withDirty.json"):
    with open("../backups/multi_model_label_withDirty.json", "r", encoding="utf-8") as f:
        old_data = json.load(f)

for item in old_data:

    flag = True
    image_path = item["images"][0]
    print(image_path)
    if "橡树岭" in image_path:
        jpg_width, jpg_height = 1842,1842
    else:
        jpg_width, jpg_height = 3450,3450

    if os.path.exists("new_multi_model_label.json"):
        with open("new_multi_model_label.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
            for i in test_data:
                if image_path in i["images"]:
                    print("这个图片修改过了")
                    flag = False
                    break

    if flag:

        messages = item["messages"]
        sys_msg = messages[0]
        sys_content = sys_msg.get("content").split("\n")
        print(sys_content)
        lines = ""
        for line in sys_content[:-1]:
            if "box" in line:

                box_str = re.findall(r".*<box>(.*)</box>", line, re.DOTALL)[0]
                box = box_str.split(",")
                print(box_str)
                box = [float(item) for item in box]
                abs_box = yolo_to_abs(box, jpg_width, jpg_height)
                # print(abs_box)
                line = line.replace(box_str,abs_box)
                print(line)
            lines = lines+line+'\n'
        last = " *注意*:<box>x_min,y_min,x_max,y_max</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标"
        lines = lines+last
        item["messages"][0]["content"] = lines
        print(item)
        if os.path.exists("new_multi_model_label.json"):
            with open("new_multi_model_label.json", "r", encoding="utf-8") as f:
                old_data = json.load(f)
                old_data.append(item)  # 合并字典
        else:
            old_data = [item]

            # 写入新数据
        with open("new_multi_model_label.json", "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False, indent=4)









