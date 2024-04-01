import torch
from .utils.config import Config
from .utils.easydict import EasyDict
from .models.videochat2_it import VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType
from .conversation import Chat



class ChatBot:
    def __init__(self, llama_model_path, vit_blip_path, videochat2_model_stage2_path, videochat2_model_stage3_path):
        self.chat = self._init_model(llama_model_path, vit_blip_path, videochat2_model_stage2_path, videochat2_model_stage3_path)
        self.chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        self.img_list = []

    def _init_model(self, llama_model_path, vit_blip_path, videochat2_model_stage2_path, videochat2_model_stage3_path):
        print('Initializing VideoChat')

        config_dict = {
            "model": {
                "model_cls": "VideoChat2_it",
                "vit_blip_model_path": "/mnt/sdb1/videoqa_model/VideoChat2/umt_l16_qformer.pth",
                "llama_model_path": "/mnt/sdb1/videoqa_model/vicuna-7b-v0",
                "videochat2_model_path": "/mnt/sdb1/videoqa_model/videochat2_7b_stage2.pth",
                "freeze_vit": False,
                "freeze_qformer": False,
                "max_txt_len": 512,
                "low_resource": False,
                "vision_encoder": {
                    "name": "vit_l14",
                    "img_size": 224,
                    "patch_size": 16,
                    "d_model": 1024,
                    "encoder_embed_dim": 1024,
                    "encoder_depth": 24,
                    "encoder_num_heads": 16,
                    "drop_path_rate": 0.0,
                    "num_frames": 16,
                    "tubelet_size": 1,
                    "use_checkpoint": False,
                    "checkpoint_num": 0,
                    "pretrained": "",
                    "return_index": -2,
                    "vit_add_ln": True,
                    "ckpt_num_frame": 4
                },
                "num_query_token": 32,
                "qformer_hidden_dropout_prob": 0.1,
                "qformer_attention_probs_dropout_prob": 0.1,
                "qformer_drop_path_rate": 0.2,
                "extra_num_query_token": 64,
                "qformer_text_input": True,
                "system": "",
                "start_token": "<Video>",
                "end_token": "</Video>",
                "img_start_token": "<Image>",
                "img_end_token": "</Image>",
                "random_shuffle": True,
                "use_lora": False,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            },
            "device": "cuda"
        }

        cfg = EasyDict(config_dict)  
        # mo
        cfg.model.llama_model_path = llama_model_path
        cfg.model.vit_blip_path = vit_blip_path
        cfg.model.videochat2_model_path = videochat2_model_stage2_path

        
        cfg.model.vision_encoder.num_frames = 16
        self.num_frames = cfg.model.vision_encoder.num_frames
        model = VideoChat2_it(config=cfg.model)
        model = model.to(torch.device(cfg.device))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        state_dict = torch.load(videochat2_model_stage3_path, "cpu")
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
            
        print(msg)
        model = model.eval()

        chat = Chat(model)
        print('Initialization Finished')
        return chat
    
    def ask_answer(self, user_message):
        self.chat.ask(user_message, self.chat_state)
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       max_new_tokens=1000)

        return llm_message[0]


    def upload(self, video_path, num_frames):
        llm_message, self.img_list, self.chat_state = self.chat.upload_video(video_path, self.chat_state, self.img_list, num_frames)
        


    def reset(self):
        if self.chat_state is not None:
            self.chat_state.messages = []
        self.img_list = []
        