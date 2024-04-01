TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="./dynabench/models/videoqa_models_library/Video_Chat/video_chat2/configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)