# <img src='https://camo.githubusercontent.com/57d5fd32c5b51e73fce9077a45f155db3edecd5dfe31d272d73569cb23ef779c/68747470733a2f2f692e696d6775722e636f6d2f6c41645a6a376d2e6a706567' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/>  LLM.cpp Persona
 
Give OpenVoiceOS some sass with [LlamaCPP](https://github.com/ggerganov/llama.cpp),  [AlpacaCPP](https://github.com/antimatter15/alpaca.cpp) or [GPT4All](https://github.com/nomic-ai/gpt4all)

This plugin requires providing the path to the executable and model, it uses subprocess which allows it to work with these programs without requiring python bindings

Dedicated plugins may exist for each LLM


## Examples 

* "What is best in life?"
* "Do you like dogs"
* "Does God exist?"


## Usage

Spoken answers api

```python
from ovos_solver_llmcpp import LLMcppSolver

ALPACA_MODEL_FILE = "./models/ggml-alpaca-7b-q4.bin"
GPT4ALL_MODEL_FILE = "./models/gpt4all-lora-quantized.bin"

# binpath = "~/alpaca.cpp/chat"
binpath = "~/gpt4all.cpp/chat"

bot = LLMcppSolver({"model": GPT4ALL_MODEL_FILE,
                    "executable_path": binpath})

sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
# Meus animais favoritos são cães, gatos e tartarugas!

for q in ["Does god exist?",
          "what is the speed of light?",
          "what is the meaning of life?",
          "What is your favorite color?",
          "What is best in life?"]:
    a = bot.get_spoken_answer(q)
    print(q, a)

```

