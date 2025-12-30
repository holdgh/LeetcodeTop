from templates.templates import StageIdentification,AbnormalResults,SpecificAbnormalJudgment,AnomalyClassification,CauseAnalysis,Ab_StageIdentification,Ab_AbnormalResults,Ab_SpecificAbnormalJudgment,Ab_AnomalyClassification,Ab_CauseAnalysis
import argparse
from templates.templates_without_image import template_1,template_2
from templates.templates_error import ErrorDetection,ErrorSpecificAbnormalJudgment,ErrorCauseAnalysis
from CLassImageLabel import WithImageLabel
from CLassWithoutImageLabel import WithoutImageLabel
from CLassImageError import ImageErrorLabel

count = 0


Q_list = [
    ["图片属于铺粉还是打印阶段的图片？","图片中是否铺粉异常","这张照片拍摄的是铺粉环节还是打印环节？","此图是铺粉错误还是打印错误?","这是铺粉步骤的图片还是打印步骤的图片？","图片属于打印阶段异常吗","这个图片属于增材制造的哪个阶段？","此图是铺粉异常还是打印异常?","请分析该图像对应的激光熔融具体阶段","该图片来自增材制造的哪一步骤","有什么异常"],
    ["图片上可能发生了什么异常？","这张照片是否显示了某种异常","是否有什么问题发生在图片中","图片里面是否发生了异常"],# ,"图片中哪些位置可能会发生异常","框出图中可能发生问题的部分"
    ["图片中是否发生了污染物的异常？","图片中是否发生了刮刀横/竖条纹的异常？","图片中是否发生了球化的异常？","图片中是否发生了铺粉不完全的异常？","图片中是否发生了翘曲凸起的异常？"],
    ["增材制造的打印阶段可能会检测到什么异常","铺粉阶段可能会发现哪些问题"],# 通用类
    ["为什么会发生球化的异常？","哪些技术问题可能导致图中出现异常？","为什么会发生刮刀横竖条纹的异常？","为什么会发生污染物的异常？","发生图片中的异常可能原因是什么？","为什么会发生图片中反映的异常","为什么会发生铺粉不完全的异常？","为什么会发生翘曲凸起的异常？"]
]

Q_list_without_image = [
    "激光熔融增材制造过程中有哪些异常会发生",
    "铺粉和打印阶段分别会发生什么问题",
    "增材制造期间所有可能会发生的异常都有哪些",
    "增材制造的打印阶段可能会检测到什么异常",
    "铺粉阶段可能会发现哪些问题",
    "激光熔融增材制造包含几个阶段",
    "temp1",
    "异常发生的原因",
    "哪些操作会导致异常",
    "为什么会发生刮刀横竖条纹的异常？",
    "为什么会发生污染物的异常？",
    "为什么会发生铺粉不完全的异常？",
    "为什么会发生球化的异常？",
    "为什么会发生翘曲凸起的异常？"
]

not_normal_path = "labels/multi_model_label.json"  # 缺陷样本标注文件
normal_path = "labels/normal.json"  # 正常样本标注文件
without_image_path = "labels/withoutImage.json"  # 通用问答对
without_image_CategoryPath = "labels/withoutImage_category.json"
not_normal_CategoryPath = "labels/multi_model_label_category.json" # 缺陷样本分类文件目录
normal_CategoryPath = "labels/normal_category.json" # 正常样本分类文件目录

detect_wrong_path = "labels/corrective.json"
detect_wrong_CategoryPath = "labels/corrective_category.json"


# not_normal_path = "labels/test1.json"  # 缺陷样本标注文件
# normal_path = "labels/test2.json"  # 正常样本标注文件
# without_image_path = "labels/test3.json"  # 通用问答对
# not_normal_CategoryPath = "labels/test4.json" # 缺陷样本分类文件目录
# normal_CategoryPath = "labels/test5.json" # 正常样本分类文件目录
#
# detect_wrong_path = "labels/test6.json"
# detect_wrong_CategoryPath = "labels/test7.json"



Abnormal_list_template = [StageIdentification,AbnormalResults,SpecificAbnormalJudgment,AnomalyClassification,CauseAnalysis]
Normal_list_template = [Ab_StageIdentification,Ab_AbnormalResults,Ab_SpecificAbnormalJudgment,Ab_AnomalyClassification,Ab_CauseAnalysis]
without_image_template = [template_1,template_2]
# Error_template = [ErrorDetection,ErrorSpecificAbnormalJudgment,ErrorCauseAnalysis]


# model = "gpt-4o"
# base_url="https://api.fe8.cn/v1"
# api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz"
base_url="http://192.168.0.194:6662/v1"
api_key="test"
model = "Qwen3-32B"




def main():


    parser = argparse.ArgumentParser(description="示例脚本")
    # parser.add_argument("--with_image", action="store_true", help="是否增加图片")
    parser.add_argument(
        "--mode",  # 参数名（推荐使用 `--mode` 形式，表示可选参数）
        type=str,  # 参数类型（str 是默认值，可省略）
        choices=["normal", "detect_error","detect_right"],  # 可选值限制
        default="detect_right",  # 默认值（如果不提供 --mode，则自动使用 "normal"）
    )
    args = parser.parse_args()
    # 包含有缺陷和没有缺陷
    if args.mode == "detect_right":
        ImageLabel = WithImageLabel(Q_list,Abnormal_list_template,Normal_list_template,not_normal_path,normal_path,not_normal_CategoryPath,normal_CategoryPath,base_url, api_key,model)
        ImageLabel.start()
    # 噪声样本
    if args.mode == "detect_error":
        row_indices = [1,2,3,4]
        Q =  [Q_list[i] for i in row_indices]
        Error_template = Abnormal_list_template[1:]
        detect_error = ImageErrorLabel(Q,Error_template,detect_wrong_path,detect_wrong_CategoryPath, base_url, api_key,model)
        detect_error.start(1000)
    # 通用知识样本
    if args.mode == "normal": 
        TextLabel = WithoutImageLabel(Q_list_without_image, without_image_template, without_image_path,without_image_CategoryPath,
                                    base_url, api_key,model)
        TextLabel.start(2000)


if __name__ ==main():
    main()