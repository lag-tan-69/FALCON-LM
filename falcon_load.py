from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
import torch 


device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is :{device}")
model_name = "tiiuae/falcon-7b"

custom_config = AutoConfig.from_pretrained(model_name)
# print(custom_config)

custom_config.num_hidden_layers = 12  
custom_config.hidden_size = 768*2   
custom_config.intermediate_size = 3072*2
custom_config.num_attention_heads = 24

tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModelForCausalLM.from_config(custom_config)
model=model.to(device=device)
print(model)
num_params = sum(p.numel() for p in model.parameters())
model_size_gb_fp32 = (num_params * 4) / (1024 ** 3)
print(f"size of the model :{model_size_gb_fp32:.2f}")

input_text = "once upon a time"
inputs = tokenizer(input_text,return_tensors='pt').to(device=device)


# with torch.no_grad():
#     outputs = model.generate(**inputs,max_length=50)
# generated_text = tokenizer.decode(outputs[0],skip_special_tokens=True)
# print(generated_text)