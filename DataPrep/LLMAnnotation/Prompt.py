'''
README:

这个class是模仿VLM-AD这个论文的free/structured prompt设计的，主要是为了让LLM更好地理解输入的内容和任务要求。
这个prompt包含了system message和user message两部分

'''
'''
Prompt Engineering in a nutshell:

System message是给LLM的角色设定和背景信息，告诉它应该以什么样的身份来回答问题。
比如在这个例子中，
User message是给LLM的具体任务描述和输入内容，告诉它需要完成什么样的任务。
比如在这个例子中，

你可以把System message看成是LLM的全局“角色设定”，告诉它应该以什么样的身份来回答问题。
你可以把User message看成是LLM的本地“任务描述”，告诉它这次推理需要完成什么样的任务。

优先级自然是System message > User message，因为System message是全局的角色设定，会影响LLM的整体行为和回答风格，而User message只是针对当前任务的具体描述。

一些常见的trick包括在System message中加入关于输出格式的要求，比如要求LLM输出JSON格式，或者要求LLM输出特定的字段，这样可以让LLM更好地理解和遵循输出要求。
同时也方便做后续下游任务的处理，比如解析LLM的输出，或者把LLM的输出作为其他模型的输入。
'''

FREEDOM_SYS = (
    "You are an expert in traffic safety and autonomous driving. "
    "You will be given a single front-view image from the ego vehicle's dashboard camera. "
    "The image may include a gaze overlay showing where the driver is looking. "
    "Analyse the scene carefully, focusing on work zone elements, surrounding traffic, and driver attention."
)

FREEDOM_USER = (
    "1. Identify whether a work zone is present and describe key elements (workers, cones, lane closures, signs).\n"
    "2. Describe surrounding traffic conditions (vehicles, traffic lights, congestion).\n"
    "3. Describe where the driver is looking based on the gaze overlay and whether it aligns with important elements.\n"
    "4. Assess potential risks considering both the work zone and traffic.\n"
    "5. Explain whether the driver's attention is appropriate for safe driving."
)


STRUCTURED_SYS = (
    "You are an expert in traffic safety and autonomous driving. "
    "You will be given a single front-view image from the ego vehicle's dashboard camera. "
    "The image may include a gaze overlay showing where the driver is looking. "
    "Use only visible evidence from the current frame. "
    "Do not infer future events or hidden dynamics unless clearly supported by the image. "
    "Respond with one valid JSON object only and no extra explanation."
)

STRUCTURED_USER = (
    "Classify the current frame using exactly these fields:\n"
    "{\n"
    "  \"workzone_present\": <yes | no>,\n"
    "  \"workzone_type\": <none | lane closure | worker activity | merging zone | mixed | unclear>,\n"
    "  \"traffic_condition\": <free flow | following vehicle | dense traffic | intersection | traffic light | unclear>,\n"
    "  \"primary_hazard\": <none | worker | cone | vehicle | traffic light | mixed | unclear>,\n"
    "  \"gaze_target\": <road center | worker | cone | vehicle | traffic light | workzone area | uncertain | irrelevant>,\n"
    "  \"attention_alignment\": <good | partial | poor>,\n"
    "  \"risk_level\": <low | medium | high>,\n"
    "  \"recommended_action\": <continue | slow down | prepare to stop | stop | prepare lane change>,\n"
    "  \"reasoning\": \"<one short sentence, max 20 words, based only on visible scene + gaze + traffic>\"\n"
    "}\n\n"
    "Rules:\n"
    "- Use only visible evidence from the frame.\n"
    "- Use unclear or uncertain when confidence is low.\n"
    "- attention_alignment = good if gaze matches the most safety-relevant area.\n"
    "- attention_alignment = partial if gaze is relevant but not on the main hazard.\n"
    "- attention_alignment = poor if gaze misses both the path and the main hazard.\n"
    "- Keep reasoning short and factual.\n"
    "- Do not include extra keys."
)

class Prompt:
    def __init__(self, system_message:str = None, user_message:str = None, seed:str = ""):
        seed_upper = seed.upper() if seed else ""
        if seed_upper == "FREEDOM":
            system_message = FREEDOM_SYS
            user_message = FREEDOM_USER
        elif seed_upper == "STRUCTURED":
            system_message = STRUCTURED_SYS
            user_message = STRUCTURED_USER

        assert system_message is not None and user_message is not None, (
            "Either seed must be provided or both system_message and user_message must be provided."
        )
        self.system_message = system_message
        self.user_message = user_message

    def __str__(self):
        return f"System Message: {self.system_message}\nUser Message: {self.user_message}"
