# Requires transformers>=4.51.0
import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                     query=query, doc=doc)
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def get_chunk_list_from_json(embedding_json: list):
    document_texts = []
    for item in embedding_json:
        document_texts.append(item['chunk'])
    return document_texts


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(r"D:\aimodel\Qwen3-Reranker-0.6B", padding_side='left')

    model = AutoModelForCausalLM.from_pretrained(r"D:\aimodel\Qwen3-Reranker-0.6B").eval()
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-8B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = ["测试叶片什么情况下允许修复？",
               ]

    embedding_json = [{"score": 0.7309813396604868,
                       "chunk": "\n## 11 试验结果评估|11.4 表观损伤\n\n在试验过程中，允许对该型叶片的运行和维护手册中规定的表观损伤进行维修。例如层内或粘接面上的微小裂纹、胶衣裂纹或涂层剥落。如果对叶片进行了维修，应根据第6章的要求进行记录。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md", "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.721906206664636,
                       "chunk": "\n## 9 试验加载与试验载荷评估|9.2 载荷引入的影响\n\n当试验载荷以集中载荷的形式施加到有限的几个位置时（如加载器位置），加载截面会受到影响，并且可能导致该截面的某一区域因加载装置的影响而被加强，进而导致叶片的这些区域可能无法被准确的测试，也不能在分析和评估中考虑。受影响区域的范围（叶片长度方向）可以通过计算或测量进行评估。\n如果没有进一步分析，可假设受影响区域为加载装置两侧各延伸  $3 / 4$  当地弦长的长度范围。在夹具设计中，应特别注意易屈曲区域（如后缘受压区域）。\n如果基于试验的目的对试验叶片做了特殊修改(见第6章),那么也需要按照上述要求进行评估。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md",
                       "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6968288267893834,
                       "chunk": "\n## 9 试验加载与试验载荷评估|9.4 疲劳试验\n\n尤其在变幅加载中，这些局限性会在相对低的载荷放大系数下出现。在这种情况下，可以通过增加中间水平的载荷循环次数，使试验趋向于恒幅加载。\n由于均值载荷对疲劳强度影响较大，因此在疲劳试验中施加的均值载荷应尽可能与风力发电机组运行的均值载荷相近。\n如果基于试验载荷的损伤（例如Miner累积损伤值）等于或者高于基于目标载荷的理论损伤，则认为该区域已经经过充分的试验验证。\n理论试验损伤可以通过不同时段的试验损伤累积得到。\n如果叶片的某一区域在试验过程中发生失效，而该区域的损伤不低于基于目标载荷的损伤，则认为该区域已经通过试验。理论上讲，如果该失效并未导致应力的重新分布，则可以继续试验，直到他区域达到目标载荷的损伤。\n如果叶片某区域因为试验载荷高于目标载荷而发生失效，则允许对该区域进行维修，但所有维修都应进行评估。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md", "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6769340240041696,
                       "chunk": "\n# 6 试验叶片的文档和生产过程文件|bullet_list_open\n- 试验叶片具有代表性的制造缺陷和在役损伤的修复；\n- 因试验载荷高于目标载荷所造成的损伤的修复（见9.3和9.4）。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md",
                       "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6656251547858396,
                       "chunk": "\n## 9 试验加载与试验载荷评估|9.3 静力试验|Part 4\n如果因为试验载荷大于目标载荷而使叶片发生失效，那么在疲劳试验开始前允许对叶片进行修复。如对叶片采用复合载荷进行试验，则无需将一个方向的最大载荷与另一个方向最大载荷进行组合，而应当使用一个方向的最大载荷与另一方向的适当载荷。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md", "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6550946636469905,
                       "chunk": "\n# 6 试验叶片的文档和生产过程文件\n\n修复也应被记录入文件，记录文件应包括上述所列。可能的修复有：\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md",
                       "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6510953709156843,
                       "chunk": "\n# 重新实施静力和疲劳试验必要性的指导方针|html_block\n由于生产中的调整、设计中的改进和整体上的优化，叶片的生产通常会偏离用于全尺寸试验的叶片。\n由于每进行一次调整或改进就重新做全尺寸试验是不现实的，因此有必要区分哪些变化需要重新开展全尺寸试验，但这已经超出本标准的范围，应由制造商和（或）认证机构进行评定。本附录仅提供一定的参考。\n应考虑对之前全尺寸试验的评估结果，因为这可能证明设计假定的正确性，同时对于评估重做试验必要性是非常有意义的。通过给定改变的程度，有必要重做的全尺寸试验可能仅包括限定的试验，比如仅静力试验、仅疲劳试验或仅就一个方向进行试验等等。\n一般而言，对叶片进行加强的调整或改进倾向于减少重做全尺寸试验的必要性。此外，对于有较大安全余量区域的改变，应倾向于不重做全尺寸试验。然而，影响风电机组载荷，进而影响叶片设计假定的变化，应考虑重做全尺寸试验。\n在生产和设计中一些典型的调整和改进，要求或不要求重做全尺寸试验的案例见表A.1中。\n表 A. 1 是否重做试验典型情况的案例\n<table><tr><td>要求重做全尺寸试验的典型调整和改进</td><td>不要求重做全尺寸试验的典型调整和改进</td></tr><tr><td>重要测试区域附近对轮廓进行的修改(如最大弦长位置)</td><td>叶尖形状的修改</td></tr><tr><td>某些纤维层的缩短</td><td>某些纤维层的延长</td></tr><tr><td>树脂或纤维类型的转换(如从聚酯树脂转换为环氧树脂,或从玻璃纤维转换为碳纤维)</td><td>材料供应商把原材料的微小调整作为连续开发的一部分,或是同一种材料换为新的供应商。后者可能需要试样水平的试验</td></tr><tr><td>在夹心结构中换为新类型的芯材(弹性模量或剪切模量不同),通常还伴随着芯材厚度的变化</td><td>夹心结构中某些芯材倒角的修改</td></tr><tr><td>在夹心结构中铺层顺序有较大变化</td><td>在较厚的叠层中铺层顺序的微小变化</td></tr><tr><td>换为新的生产方法(如从手糊成型变为灌注成型)</td><td>生产过程的微小变化(如固化过程的调整)</td></tr></table>\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md", "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6356980366346892,
                       "chunk": "\n## 11 试验结果评估|11.1 总则\n\n在每次试验开始前、试验完成后以及疲劳试验过程中定期对叶片的内腔、外表面进行外观检查。\n外观检查也可采用红外成像、超声波或声发射技术进行。\n所有的检查结果均应形成记录，并在相应的文件中予以体现。\n在试验过程中，应对安装或预埋的关键电气系统，如防雷引下线、与控制相关的传感器，进行定期的功能检查和确认。\n叶片损伤是指对叶片造成的不可逆的特性改变，叶片损伤可以分为如下类型：\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md",
                       "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6322087614722958,
                       "chunk": "\n# 6 试验叶片的文档和生产过程文件\n\n为了达到试验目的，可以对叶片进行特殊的修改。在疲劳试验过程中，为在一个可接受的时间内完成试验，可能需要放大载荷。在某些情况下，疲劳载荷的放大可能导致非测试区域失效。在这些情况下，可对叶片进行特殊的修改。修改也可能是加载点位置处的加强。应记录所有特殊的叶片修改。\n",
                       "file_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验_splitted.md",
                       "file_link": "cef331b411754d6ea628f1ba63f368d21765334767413.md", "title": "----------",
                       "document_id": "73f916ba-d572-11f0-8a58-7ac0afde6275",
                       "dataset_name": "GB∕T 25384-2018 风力发电机组 风轮叶片全尺寸结构试验"},
                      {"score": 0.6286803228676803,
                       "chunk": "\n# D.2.1.6 Test procedure|Part 3\nf) Inspect the test specimen and document the results.g) If puncture has occurred, perform an assessment to determine if the test specimen has failed the test.If it is deemed to have failed, then the test sequence may need to be terminated, or repairs of test damage or modifications to the blade lightning protection system made before continuing with the tests.\n",
                       "file_name": "IEC 61400-24-2010 风力机 第24部分：雷电防护_splitted.md",
                       "file_link": "a03d3bd633e34bc298f99c5714ec89c71765521519389.md",
                       "title": "----------",
                       "document_id": "4a806ebe-d725-11f0-b1f5-7ac0afde6275",
                       "dataset_name": "IEC 61400-24-2010 风力机 第24部分：雷电防护"}]

    # documents = [
    #     "The capital of China is Beijing.",
    #     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    # ]
    documents = get_chunk_list_from_json

    pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

    # Tokenize the input texts
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)

    print("scores: ", scores)
