from pydantic import BaseModel


transfer_passenger_count = 30

condition = [
    {"charge_name": "transfer passenger charge", "transfer_passenger_count": transfer_passenger_count, "period": "summer airline scheduling season", "rate": "?"},
    {"charge_name": "transfer passenger charge", "transfer_passenger_count": transfer_passenger_count, "period": "winter airline scheduling season", "rate": "?"},
    {"charge_name": "runway landing and takeoff charge", "period": "summer airline scheduling season", "atm": "landing", "per tonne MTOW": 1, "rate": "?"},
    {"charge_name": "runway landing and takeoff charge", "period": "summer airline scheduling season", "atm": "takeoff", "per tonne MTOW": 1, "rate": "?"},
    {"charge_name": "runway landing and takeoff charge", "period": "winter airline scheduling season", "atm": "landing", "per tonne MTOW": 1, "rate": "?"},
    {"charge_name": "runway landing and takeoff charge", "period": "winter airline scheduling season", "atm": "takeoff", "per tonne MTOW": 1, "rate": "?"}
]

MODEL_ID = "Qwen/Qwen3-30B-A3B"
import torch, outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1️⃣  HF loads & shards the model – ONE LINE does the heavy lifting.
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",                # shards across all visible GPUs
    low_cpu_mem_usage=True,           # avoids a large RAM peak
)

# 2️⃣  Wrap in Outlines.
tok        = AutoTokenizer.from_pretrained(MODEL_ID)
model      = outlines.from_transformers(hf_model, tok)          # note new API

# 3️⃣ Define the output structure
class Rate(BaseModel):
    rate: float

# 4️⃣ Read the markdown file
with open("output/2025-airport-charges-terms-and-conditions/tinychargesmarkdown.md", "r") as f:
    markdown_content = f.read()

# 5️⃣ Create the generator
generator = outlines.generate.json(model, Rate)

# 6️⃣ Loop over conditions and update the rate
for cond in condition:
    prompt = f"""You are a world-class algorithm for extracting information from text. 

    Here is a document:
    ---
    {markdown_content}
    ---

    Given the following condition:
    {cond}

    Extract the rate based on the document and condition.
    """
    rate_object = generator(prompt)
    cond["rate"] = rate_object.rate

# 7️⃣ Print the updated conditions
print(condition)
