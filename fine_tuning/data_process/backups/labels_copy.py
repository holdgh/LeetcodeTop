from cProfile import label

from zhipuai import ZhipuAI
from openai import OpenAI
import re
import random
import os
import json
import base64
from PIL import Image
import io


count = 0


Q_list = [
    ["图片属于铺粉还是打印阶段的图片？","这张照片拍摄的是铺粉环节还是打印环节？","这是铺粉步骤的图片还是打印步骤的图片？","这个图片属于增材制造的哪个阶段？"],
    ["图片上可能发生了什么异常？","这张照片是否显示了某种故障或错误","是否有什么问题发生在图片中","图片里面是否发生了异常","图片中哪些位置可能会发生异常"],
    ["发生图片中的异常可能原因是什么？","为什么会发生图片中反映的异常","哪些技术问题可能导致图中出现异常？","哪些操作会导致异常"]
]


def get_response(text,label,type):
    client = OpenAI(base_url="https://api.fe8.cn/v1",api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz")
    messages = [
        {
            "role": "system",
            "content": '''
            # 背景：3D激光熔融打印是一种高精度的金属增材制造技术，其核心流程包括铺粉和打印两个关键阶段。
            # 定位：你是一个优秀的分析大师，你擅长依据用户输入的打标信息返回给用户一段分析文本，所有的分析结果都需要依据用户的输入。
            # 打标信息说明:输入内容是一个字典，键的含义表示图片中包含的异常类型，值表示异常发生的位置。例如{污染物:12,23,12,34}。表示在图片中左上角x位置为12,左上角y位置为23,右下角x位置12,右下角y位置34的区域发生了污染物的异常。
                  
                      
            # 返回要求：
                ## 只返回和用户问题相关的内容，不要扩展回答的内容。禁止回答解决方案的内容
                ## 禁止在回答中出现“根据标注”、“从标注中”，等字样。把标注当作已知信息。
                ## 用于定位的！@#$%…&*等符号禁止返回在文本种中。
                ## 使用同义词替换、语序更改的形式，使返回的文本形式具有多样性和丰富性。
                ## 缺陷发生的位置定位描述禁止简单输出box的位置，需要对位置进行总结后返回。
                                
                                
            # 回答模板：你需要根据问题对应的类型选择响应的模板进行回答，每个类型有多个模板，随机选择一个模板进行回答问题。
                ## 问题类型一：用户问题询问关于图片中有什么异常、缺陷、错误等，询问图片的异常结果。多个位置发生同一种缺陷类型可以统一描述，不需要分开。缺陷的明显特征描述可以根据模型已知信息进行丰富，但是不要丰富已知信息以外的内容。
                    模板一：
                        图片中一共出现@<缺陷种类数量>种缺陷问题。
                            第一种缺陷类型是#<缺陷类型一>,*<缺陷发生的位置定位描述，缺陷发生的范围描述>。该区域中$<缺陷的明显特征描述>。因此可以分析在这个位置发生了#<缺陷类型一>异常。
                            第二种缺陷类型是#<缺陷类型二>,*<缺陷发生的位置定位描述，缺陷发生的范围描述>。该位置区域$<缺陷的明显特征描述>。因此可以分析在这个位置发生了#<缺陷类型二>异常。
                            。
                            。
                            。
                    模板二：
                        图片中出现了明显的异常现象：
                            首先，#<缺陷类型一> 在 *< 缺陷发生的位置定位描述 > 处被发现。仔细查看该区域，$< 缺陷的明显特征描述 >，据此能够确定此处存在 #< 缺陷类型一 > 异常。
                            其次，#<缺陷类型二> 出现在 *< 缺陷发生的位置定位描述 >。此位置区域呈现出 $< 缺陷的明显特征描述 >，由此可以判断该位置发生了 #< 缺陷类型二 > 异常情况。
                        总结：图片中的增材制造过程发生@<缺陷种类数量>种缺陷问题。    
                ## 问题类型二：用户询问图片是否发生异常、缺陷、错误。
                    模板一：图片中！<出现或未出现>异常。
                        $< 缺陷的明显特征描述 >，因此在图片中发生了#<缺陷类型>问题。
                    模板二：图片中 @<存在或不存在> 异常情况。
                        包含异常类型：#<缺陷类型>：$< 缺陷的明显特征描述 >。
                ## 问题类型三：用户询问异常发生的原因.
                    模板一：发生#<缺陷类型一>的原因是因为%<原因发生的具体描述>。
                    模板二：#<缺陷类型>：%<原因发生的具体描述>，（对原因的总结）都会导致缺陷的发生
                ## 问题类型四：用户询问图片所属阶段。
                    可以使用语义相同的描述对模板进行替换，保持原有语义特征。
                    如果用户问题问是或不是某一个阶段的图片，需要先回答“是&<铺粉或打印>阶段”或者“不是&<铺粉或打印>阶段”，然后选择模板进行分析。
                    模板一：图片属于增材制造的&<铺粉或打印>阶段。<铺粉或打印阶段的特征总结>。
                    模板二：从图片中分析，属于&<铺粉或打印>阶段。<铺粉或打印阶段的特征总结>
                    铺粉阶段通常是在打印一层之前进行的准备工作，表现为每次铺粉的过程相似。铺粉完成后，才会进入下一个打印阶段的操作。铺粉图片主要是金属粉末的颜色，一般是灰色，并且零件应该被粉末覆盖，无法看到零件。打印阶段能看到成形平台上有已固化或部分固化的零件，零件周围是未打印的金属粉末。
                ## 问题类型五：询问发生异常的位置。（最多给出两个box的举例）
                    模板一：
                        
                        详细描述：
                            #<缺陷类型一>
                                位置区域：<异常发生的box区域>
                                特征描述：$<缺陷的明显特征描述>
                                可能原因：<可选：推测原因>
                            #<缺陷类型二>
                                位置区域：<异常发生的box区域>
                                特征描述：$<缺陷的明显特征描述>
                                可能原因：<可选：推测原因>
                    模板二：
                        图片可能发生<异常发生的box区域>处异常：
                        <异常发生的box区域一>（最多给出两个box的举例）
                                异常类型：#<缺陷类型一>
                                特征描述：$<缺陷的明显特征描述>
                            #<缺陷类型二>
                                位置区域：<异常发生的box区域>（最多给出两个box的举例）
                                异常类型：#<缺陷类型一>
                                特征描述：$<缺陷的明显特征描述>
                    
                ## 问题类型六：哪些操作会导致异常
                    模板一：
                        在图片中没有发现明显的缺陷。但是以下操作可能会导致打印过程发生缺陷：
                            #<操作一>：
                                缺陷类型：%<缺陷类型一>
                                特征说明：$<缺陷的明显特征描述>
                            #<操作二>：
                                缺陷类型：%<缺陷类型二>
                                特征说明：$<缺陷的明显特征描述>
            # 缺陷类型以及缺陷对应的特征描述
                
                ## 铺粉不完全：正常铺粉图片应该表现为均匀的、灰色的。但是在图片中出现大面积的、在零件表面展示的白色亮面。主要是由于粉末供应不足、风机转速过高把粉吹走，导致该铺上粉的地方没铺上。
                    铺粉不完全（Incomplete Powder Spreading）是影响打印质量的关键问题之一。它会导致局部粉末缺失、层厚不均，进而引发未熔合、孔隙、表面粗糙等缺陷。
                    铺粉不完全表现在局部粉末缺失，某些区域无粉末覆盖，露出下层已固化材料或基板。
                ## 曲翘凸起：正常铺粉图片应该表现为均匀的、灰色的。
                    翘起凸起发生在打印阶段，但是在铺粉阶段更容易被发现，在铺粉的图片中能看到零件的边缘轮廓，呈现白色或金属光泽，则表示零件边缘没有铺上金属粉末。
                    这类异常都是打印时过熔产生的，一般来说在边缘轮廓位置，激光行进路程较短，能量较为集中更容易产生翘曲。只要是高于粉床表面的统一认为是翘曲凸起。
                    激光熔融3D打印过程中出现的翘曲凸起是一个常见的工艺缺陷，主要由材料在快速加热和冷却过程中产生的不均匀热应力或者打印功率较高扫描速度较慢引起。
                ## 刮刀横/竖条纹：正常铺粉图片应该表现为均匀的、灰色的。
                    “刮刀横/竖条纹”异常的明显特征是能在图片中看到贯穿图片的深黑色的横线，从图片左侧一直到右侧。
                    刮刀上有缺口、颗粒会在铺粉时产生一个横条纹，刮刀碰到凸起时抖动可能会在铺粉时产生竖条纹。
                    刮刀磨损或变形：长期使用后，刮刀边缘可能出现磨损、缺口或弯曲，导致铺粉时形成不均匀的条纹。
                    刮刀与粉床间隙不均：若刮刀安装不平整或Z轴运动误差，可能导致部分区域铺粉过厚或过薄，形成横向条纹。
                    铺粉速度过快：刮刀移动速度过高时，粉末可能被推挤而非均匀铺展，导致条纹状分布。
                ## 球化：在打印阶段，过熔会导致金属表面形貌出现异常，打印后的金属面平整度出现异常。在铺粉阶段时，图片中表现的特征零件表面粗糙部分位置可以被粉末覆盖，但是其内部会有孔隙，表现为白色的亮点和黑色的球状颗粒，因此在铺粉阶段同样能发现球化的异常。
                    球化是指熔融金属在冷却过程中未能均匀铺展，反而收缩成球状或椭球状的现象。这一缺陷会严重影响打印件的致密度、表面质量和机械性能。
                    球化的金属表面粗糙，打印件表面布满凸起的球状或椭球状颗粒。孔隙率高，球体之间未完全结合，形成蜂窝状孔隙。
                ## 污染物：灰渣/残余颗粒/飞溅颗粒统一认为是污染物，从图像特征上看，这些污染物表现特征是明显的深灰色或黑色阴影。
                    污染物形成的原因可能是打印过程中激光温度不够高，金属粉末未完全被熔融遗留的残灰。也可能是激光温度过高导致过熔后碎片飞溅形成的。
                
                
                
                    '''
        },
        {
            "role": "user",
            "content": text+"\n"+"打标信息："+str(label)+"\n"+f'当前图片属于{type}阶段的结果'


        }

    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    event_text = response.choices[0].message.content if response.choices else ""
    try:
        event_text = event_text.replace("!",'').replace("@",'').replace("#",'').replace("$",'').replace("%",'').replace("&",'').replace("*",'')
    # response = re.findall(r".*<response>(.*)</response>", event_text, re.DOTALL)[0]
    except :
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        event_text = response.choices[0].message.content if response.choices else ""
        event_text = event_text.replace("!", '').replace("@", '').replace("#", '').replace("$", '').replace("%",
                                                                                                            '').replace(
            "&", '')

    return event_text, response.usage.total_tokens


def get_normal_response(text, type):
    client = OpenAI(base_url="https://api.fe8.cn/v1", api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz")
    messages = [
        {
            "role": "system",
            "content": '''
            # 背景：3D激光熔融打印是一种高精度的金属增材制造技术，其核心流程包括铺粉和打印两个关键阶段。
            # 定位：你是一个优秀的分析大师，你擅长回答用户的问题。用户咨询的图片属于正常没有问题的图片。
            # 返回要求：
                ## 只返回和用户问题相关的内容，不要扩展回答的内容。禁止回答解决方案的内容
                ## 用于定位的！@#$%…&*等符号禁止返回在文本种中。
                ## 使用同义词替换、语序更改的形式，使返回的文本形式具有多样性和丰富性。


            # 回答模板：你需要根据问题对应的类型选择响应的模板进行回答，每个类型有多个模板，随机选择一个模板进行回答问题。
            
                ## 问题类型一：询问图片异常结果，用户问题询问关于图片中有什么异常、缺陷、错误等，异常发生的位置。多个位置发生同一种缺陷类型可以统一描述，不需要分开。缺陷的明显特征描述可以根据模型已知信息进行丰富，但是不要丰富已知信息以外的内容。
                    模板一：
                        从图片特征上来看，&<铺粉或打印>过程表现正常，图片铺粉均匀，没有发现异常特征。
                    模板二：
                        图片中没有发现铺粉不完全、污染物或球化特征，铺粉结果均匀，并且发现刮刀刮痕。因此没有异常发生。
                
                ## 问题类型二：用户询问3D打印过程中有什么异常会发生，哪些操作会导致异常，总结所有的异常和异常的原因。使用同义词替换、语序更改的形式，使返回的文本形式具有多样性和丰富性。
                    模板一：
                        根据用户提供的图片，该过程没有发现异常。在3D打印过程中可能会以下异常：
                            #<缺陷类型一>：
                                特征描述：$<缺陷的明显特征描述>。
                                发生原因：%<原因发生的具体描述>
                            #<缺陷类型一>：
                                特征描述：$<缺陷的明显特征描述>
                                发生原因：%<原因发生的具体描述>
                                
                    模板二：
                        图片显示的3D打印该过程没有发现异常。普遍在3D打印过程中会发生的异常有以下几种：
                            #<缺陷类型一>：
                                原因说明：%<原因发生的具体描述>
                                特征说明：$<缺陷的明显特征描述>
                            #<缺陷类型一>：
                                原因说明：%<原因发生的具体描述>
                                特征说明：$<缺陷的明显特征描述>
                    模板三：
                        在图片中没有发现明显的缺陷。但是以下操作可能会导致打印过程发生缺陷：
                            #<操作一>：
                                缺陷类型：%<缺陷类型一>
                                特征说明：$<缺陷的明显特征描述>
                            #<操作二>：
                                缺陷类型：%<缺陷类型二>
                                特征说明：$<缺陷的明显特征描述>
                    
                    
                ## 问题类型三：用户询问图片所属阶段。使用同义词替换、语序更改的形式，使返回的文本形式具有多样性和丰富性。
                    可以使用语义相同的描述对模板进行替换，保持原有语义特征。
                    如果用户问题问是或不是某一个阶段的图片，需要先回答“是&<铺粉或打印>阶段”或者“不是&<铺粉或打印>阶段”，然后选择模板进行分析。
                    模板一：图片属于增材制造的&<铺粉或打印>阶段。<铺粉或打印阶段的特征总结>
                    模板二：从图片中分析，属于&<铺粉或打印>阶段。<铺粉或打印阶段的特征总结>
                    
                    ### 参考内容
                        铺粉阶段通常是在打印一层之前进行的准备工作，表现为每次铺粉的过程相似。
                        铺粉完成后，才会进入下一个打印阶段的操作。
                        铺粉图片主要是金属粉末的颜色，一般是灰色，并且零件应该被粉末覆盖，无法看到零件。
                        打印阶段能看到成形平台上有已固化或部分固化的零件，零件周围是未打印的金属粉末。
                
            # 可能会发生的缺陷类型以及缺陷对应的特征描述

                ## 铺粉不完全：正常铺粉图片应该表现为均匀的、灰色的。但是在图片中出现大面积的、在零件表面展示的白色亮面。主要是由于粉末供应不足、风机转速过高把粉吹走，导致该铺上粉的地方没铺上。
                    铺粉不完全（Incomplete Powder Spreading）是影响打印质量的关键问题之一。它会导致局部粉末缺失、层厚不均，进而引发未熔合、孔隙、表面粗糙等缺陷。
                    表现在局部粉末缺失，某些区域无粉末覆盖，露出下层已固化材料或基板。
                ## 曲翘凸起：正常铺粉图片应该表现为均匀的、灰色的。翘起凸起在铺粉阶段更容易被发现，在图片中能看到零件的边缘轮廓，呈现白色或金属光泽，则表述零件边缘没有铺上金属粉末。发生的原因是在上一层的打印阶段发生曲翘凸起的异常，只要是高于粉床表面的统一认为是翘曲凸起。
                    这类异常都是打印时过熔产生的，一般来说在边缘轮廓位置，激光行进路程较短，能量较为集中更容易产生翘曲。
                    激光熔融3D打印过程中出现的翘曲凸起是一个常见的工艺缺陷，主要由材料在快速加热和冷却过程中产生的不均匀热应力引起。
                ## 刮刀横/竖条纹：正常铺粉图片应该表现为均匀的、灰色的。“刮刀横/竖条纹”异常的明显特征是能在图片中看到贯穿图片的深黑色的横线或竖线，从图片左侧一直到右侧或从图片上侧到下侧。
                    刮刀上有缺口、颗粒会在铺粉时产生一个横条纹，刮刀碰到凸起时抖动可能会在铺粉时产生竖条纹。
                    刮刀磨损或变形：长期使用后，刮刀边缘可能出现磨损、缺口或弯曲，导致铺粉时形成不均匀的条纹。
                    刮刀与粉床间隙不均：若刮刀安装不平整或Z轴运动误差，可能导致部分区域铺粉过厚或过薄，形成横向条纹。
                    铺粉速度过快：刮刀移动速度过高时，粉末可能被推挤而非均匀铺展，导致条纹状分布。
                ## 球化：过熔会导致金属表面形貌出现异常，打印后的金属面平整度出现异常，图片中表现的特征零件表面粗糙，部分位置可以被粉末覆盖，但是其内部会有孔隙，表现为白色的亮点和黑色的球状颗粒，
                    球化是指熔融金属在冷却过程中未能均匀铺展，反而收缩成球状或椭球状的现象。这一缺陷会严重影响打印件的致密度、表面质量和机械性能。
                    球化的金属表面粗糙，打印件表面布满凸起的球状或椭球状颗粒。孔隙率高，球体之间未完全结合，形成蜂窝状孔隙。
                ## 污染物：灰渣/残余颗粒/飞溅颗粒统一认为是污染物，从图像特征上看，这些污染物表现特征是明显的深灰色或黑色阴影。污染物形成的原因可能是打印过程中激光温度不够高，金属粉末未完全被熔融遗留的残灰。也可能是激光温度过高导致过熔后碎片飞溅形成的。



                    '''
        },
        {
            "role": "user",
            "content": text + "\n" + "打标信息："  + "\n" + f'当前图片属于{type}阶段的结果'

        }

    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    event_text = response.choices[0].message.content if response.choices else ""
    try:
        event_text = event_text.replace("!", '').replace("@", '').replace("#", '').replace("$", '').replace("%",
                                                                                                            '').replace(
            "&", '').replace("*", '')
    # response = re.findall(r".*<response>(.*)</response>", event_text, re.DOTALL)[0]
    except:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        event_text = response.choices[0].message.content if response.choices else ""
        event_text = event_text.replace("!", '').replace("@", '').replace("#", '').replace("$", '').replace("%",
                                                                                                            '').replace(
            "&", '')

    return event_text


def Optimize_Q(qus):
    client = OpenAI(base_url="https://api.fe8.cn/v1",api_key="sk-j2n1UvlYoG7zBwkF5DZrlQ6QYeW1qhhEtsWwDFW5oUYystzz")
    # client = ZhipuAI(api_key="4d9215f97db5474893055d9c7d9b4699.aqivdQGDomfJm8WK")
    messages = [


        {
            "role": "system",
            "content": ''' 
            # 角色设定：作为3D打印的行业专家，并且是一个优化专家，专注于改进用户提供的初始问题表述，保持原意基础上提升表达效果。
            # 注意：
                ## “铺粉”和“上色不是同义词替换”
            # 优化准则：
            
                ## 核心原则 - 通过调整措辞、重组句式等策略重构问题，严格保持原有信息边界，不要脱离行业领域。
                ## 禁止事项 - 避免引入任何虚构或假设性信息
                ## 优化后的新问题放在标签<question></question>中返回。
                ## 必须返回一个问题
                ## 优化的问题字数在100字以内。
                
                '''
        },
        {
            "role": "user",
            "content": qus

        }

    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    event_text = response.choices[0].message.content if response.choices else ""
    if "question" in event_text:
        sub_question = re.findall(r".*<question>(.*)</question>", event_text, re.DOTALL)[0]
    else:
        print(qus)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        event_text = response.choices[0].message.content if response.choices else ""
        sub_question = re.findall(r".*<question>(.*)</question>", event_text, re.DOTALL)[0]

    return sub_question, response.usage.total_tokens

def adjust(data):
    # data:list
    # return data_dict:dict
    data_dict = {}
    for item in data:
        data_dict[item["image"]] = item

    return data_dict

def yolo_to_abs(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    x_min = int((x_center - w / 2) * img_width)
    y_min = int((y_center - h / 2) * img_height)
    x_max = int((x_center + w / 2) * img_width)
    y_max = int((y_center + h / 2) * img_height)
    return [x_min, y_min, x_max, y_max]

def from_boxes_get_system_msg(label,img_width, img_height):
    # annotations = label.get("annotations", None)[0].get("result", None)
    # boxes = {}
    # if annotations:
    #     # print(annotations)
    #     for annotation in annotations:
    #         box = annotation.get("value", None)
    #         rectanglelabels = box["rectanglelabels"][0]
    #         boxes[rectanglelabels] = [box["x"], box["y"], box["width"], box["height"]]
    # # print(boxes)
    label_msg = ""
    # for rectanglelabel, box in boxes.items():
    #     label_msg += f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>区域表现的异常类型是{rectanglelabel}\n"

    for classes,boxes in label.items():
        # classes = classes.split("_")[1]
        classes = classes.split("(")[0]
        for box in boxes:
            # box1 = yolo_to_abs(box, img_width, img_height)
            label_msg += f"<box>{box[0]},{box[1]},{box[2]},{box[3]}</box>区域表现的异常类型是{classes}\n"

    return label_msg

def get_question(existed=None,first=None):
    if first:
        templates_id = random.randint(0, len(Q_list) - 2)
    else:
        templates_id = random.randint(0, len(Q_list) - 1)
    if existed:

        while templates_id in existed:
            templates_id = random.randint(0, len(Q_list) - 1)
        existed.append(templates_id)
    # templates = Q_list[templates_id]
    templates = Q_list[2]
    seed = random.randint(0, len(templates) - 1)
    # old_question = templates[seed]
    old_question = templates[3]
    new_question, token = Optimize_Q(old_question)
    return new_question

def get_assitant_response(rel_path,question,label,type):
    # with open(rel_path, 'rb') as img_file:
    #     img_base = base64.b64encode(img_file.read()).decode('utf-8')
    #
    #     try:
    #         img_bytes = base64.b64decode(img_base)
    #         img = Image.open(io.BytesIO(img_bytes))
    #         img.verify()  # 验证图片是否损坏
    #
    #     except Exception as e:
    #         print(f"图片验证失败: {e}")

        response, token = get_response(question, label, type)
        return(response)

def main():
    count = 0
    # 注意：如果要用这个脚本跑生产数据集的标签，需要先对齐标签顺序和内容
    #
    #
    label_info = {"0":"刮刀横/竖条纹","1":"曲翘凸起","2":"污染物","3":"污染物","4":"球化","5":"刮刀横/竖条纹","6":"曲翘凸起","7":"铺粉不完全"}
    data = {}
    data["data"] = []
    walks = ["yoloLabels_生产","yoloLabels_公共"]

    for walk in walks:
        print(walk)
        for root, dirs, files in os.walk(walk):
            for file in files:
                # 图片路径整理
                multi_label = {}
                label_path = os.path.relpath(os.path.join(root, file)).replace("\\","/")

                # label = yolo_label.get(file,None)
                if "Image" in file:
                    image_name = "Image"+file.split("__")[1].split("Image")[1].split(".")[0]+".jpg"
                    image_path = os.path.relpath(os.path.join("上飞公司\image", image_name)).replace("\\","/")
                elif "Image" not in file and "__" in file:
                    image_name = file.split("__")[1].split(".")[0] + ".png"
                    image_path = os.path.relpath(os.path.join("橡树岭", image_name)).replace("\\", "/")
                else:
                    image_name = file.split(".")[0].split("-")
                    image_name = image_name[1]+"-"+ image_name[2]+ ".png"
                    image_path = os.path.relpath(
                        os.path.join("橡树岭", image_name)).replace("\\", "/")

                # _path = os.path.relpath(os.path.join("yoloLabels_生产", name))
                print( image_path, file,image_name,label_path)

                # 检查这个yololabel是否已经生成过多模态标注
                need_label_flag = True

                if os.path.exists("multi_model_label_withDirty.json"):
                    with open("multi_model_label_withDirty.json", "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        for item in old_data:
                            if image_path in item["images"]:
                                need_label_flag = False
                                break

                if os.path.exists("../labels/normal.json"):
                    with open("../labels/normal.json", "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        for item in old_data:
                            if image_path in item["images"]:
                                need_label_flag = False
                                break




                # 根据图片名称获取标注
                if os.path.exists(label_path):
                    with open(label_path, "r", encoding="utf-8") as file:
                        content = file.read()  # 读取全部文本
                        content = content.split("\n")
                        if len(content)>0:

                            label = {}
                            for line in content:
                                if len(line)>0:
                                    temp = line.split(" ")
                                    classes = label_info.get(temp[0])
                                    # 注意：这里是因为在实验数据集中有人使用了生产数据集的标签，所以这里增加了判断，如果用的生产标签，就替换为实验的
                                    # if classes == "生产_刮刀横/竖条纹":
                                    #     classes = "公共_刮刀横/竖条纹"
                                    boxes = label.get(classes,[])
                                    box = [round(float(temp[1]), 2),round(float(temp[2]), 2),round(float(temp[3]), 2),round(float(temp[4]), 2)]
                                    if "橡树岭" in image_path:
                                        jpg_width, jpg_height = 1842, 1842
                                    else:
                                        jpg_width, jpg_height = 3450, 3450
                                    box = yolo_to_abs(box,jpg_width, jpg_height)
                                    boxes.append(box)
                                    label[classes] = boxes
                        else:
                            label = None
                            print("标签为空")
                    print(label)
                else:
                    label = None
                    print("路径不存在")



                #  根据图片名称判断图片属于哪一个阶段的图片。
                if "spreaded" in image_name or "Image" in image_name:
                    type = "铺粉"
                else:
                    type = "打印"
                # 如果这个图片能找到标注信息
                if label and need_label_flag:
                    message = []

                    label_msg = from_boxes_get_system_msg(label,jpg_width, jpg_height)
                    system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n{label_msg}。 \n *注意*:<box>x,y,w,h</box>区域表示需要重点关注的区域。依次是:左上角x坐标，左上角y坐标，右下角x坐标，右下角y坐标。'
                    message.append({"role": "system", "content": system_str})
                    # print(system_str)

                    multi_turn_flag = random.randint(0,1)
                    # multi_turn_flag = 1
                    print(multi_turn_flag)
                    # 多轮对话
                    if multi_turn_flag:
                        existed = [len(Q_list)]
                        turn_times = random.randint(2,3)
                        # turn_times = 2
                        for i in list(range(0,turn_times)):
                            # 获取问题并优化
                            if i == 0:
                                new_question = "<image>"+get_question(existed,True)  # 优化过后的
                            else:
                                new_question = get_question(existed)

                            # 保存用户输入
                            message.append({"role": "user", "content": new_question})
                            # 获取模型回答response

                            response = get_assitant_response(image_path, new_question, label, type)
                            message.append({"role": "assistant", "content": response})
                            print(new_question)
                            print(response)
                        multi_label["messages"] = message
                        multi_label["images"] = [image_path]
                        data["data"].append(multi_label)


                    # 单轮对话
                    else:

                        new_question = "<image>"+get_question()# 优化过后的
                        message.append({"role": "user", "content": new_question})

                        response = get_assitant_response(image_path, new_question, label,type)
                        message.append({"role":"assistant","content":response})
                        multi_label["messages"] = message
                        multi_label["images"] = [image_path]
                        data["data"].append(multi_label)
                        print(new_question)
                        print(response)
                        # 如果文件存在，读取旧数据

                    # print(data)
                    print("write")
                    if os.path.exists("multi_model_label_withDirty.json"):
                        with open("multi_model_label_withDirty.json", "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            old_data.append(multi_label) # 合并字典
                    else:
                        old_data = data["data"]

                        # 写入新数据
                    with open("multi_model_label_withDirty.json", "w", encoding="utf-8") as f:
                        json.dump(old_data, f, ensure_ascii=False, indent=4)






                elif need_label_flag:
                    print("这个图片没有标注信息但是需要标注，这是一个正常样本")
                    message = []
                    system_str = f'你是一个面向3D打印场景的目标检测大师，具备精准识别、定位图像中固定位置图像是否包含缺陷的能力。\n 职能:从给出的重点区域分析，该区域是否包含异常。\n在这个图片中没有发生异常。请按照没有异常发生的情况回答用户的问题。'
                    message.append({"role": "system", "content": system_str})
                    # print(system_str)

                    multi_turn_flag = random.randint(0, 1)
                    # multi_turn_flag = 1
                    print(multi_turn_flag)
                    # 多轮对话
                    if multi_turn_flag:
                        existed = [len(Q_list)]
                        turn_times = random.randint(2, 3)
                        # turn_times = 2
                        for i in list(range(0, turn_times)):
                            # 获取问题并优化
                            if i == 0:
                                new_question = "<image>" + get_question(existed, True)  # 优化过后的
                            else:
                                new_question = get_question(existed)

                            # 保存用户输入
                            message.append({"role": "user", "content": new_question})
                            # 获取模型回答response
                            response = get_normal_response( new_question, type)
                            message.append({"role": "assistant", "content": response})
                            print(new_question)
                            print(response)
                        multi_label["messages"] = message
                        multi_label["images"] = [image_path]
                        data["data"].append(multi_label)


                    # 单轮对话
                    else:

                        new_question = "<image>" + get_question()  # 优化过后的
                        message.append({"role": "user", "content": new_question})

                        response = get_normal_response( new_question, type)
                        message.append({"role": "assistant", "content": response})
                        multi_label["messages"] = message
                        multi_label["images"] = [image_path]
                        data["data"].append(multi_label)
                        print(new_question)
                        print(response)
                        # 如果文件存在，读取旧数据

                    # print(data)
                    print("write")
                    if os.path.exists("../labels/normal.json"):
                        with open("../labels/normal.json", "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            old_data.append(multi_label)  # 合并字典
                    else:
                        old_data = data["data"]

                        # 写入新数据
                    with open("../labels/normal.json", "w", encoding="utf-8") as f:
                        json.dump(old_data, f, ensure_ascii=False, indent=4)



                else:
                    print("这个图片已经标注过了")







if __name__ ==main():
    main()