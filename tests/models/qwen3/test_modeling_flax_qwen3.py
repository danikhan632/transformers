from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
import jax.numpy as jnp

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the Flax model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxAutoModelForCausalLM.from_pretrained(model_name, dtype=jnp.float32)

# prepare the prompt in Qwen’s chat format
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

# tokenize into JAX arrays
inputs = tokenizer([text], return_tensors="jax", padding=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# generate with the built-in JAX-backed generate()
# you can also pass temperature, top_k, etc.
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
)

# outputs.sequences is a JAX array of shape (batch, seq_len)
generated = outputs.sequences[0]
# strip off the prompt tokens
generated_ids = generated[len(input_ids[0]):]

# find Qwen’s </think> special-token ID
think_id = 151668
ids_list = list(generated_ids)
try:
    idx = ids_list.index(think_id) + 1
except ValueError:
    idx = 0

# decode
thinking = tokenizer.decode(ids_list[:idx], skip_special_tokens=True).strip()
content  = tokenizer.decode(ids_list[idx:], skip_special_tokens=True).strip()

print("thinking content:", thinking)
print("content:", content)
