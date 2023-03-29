from neon_solvers import AbstractSolver

from ovos_solver_llmcpp.engine import LLMcppInterface


class LLMcppSolver(AbstractSolver):
    def __init__(self, config=None):
        super().__init__(name="LLMcpp", priority=94, config=config,
                         enable_cache=False, enable_tx=True)
        checkpoint = self.config.get("model")
        executable_path = self.config.get("executable_path")
        self.model = LLMcppInterface(executable_path, checkpoint)

    # officially exported Solver methods
    def get_spoken_answer(self, query, context=None):
        return self.model.ask(query)


if __name__ == "__main__":
    ALPACA_MODEL_FILE = "/home/user/PycharmProjects/ovos-solver-plugin-llamacpp/models/ggml-alpaca-7b-q4.bin"
    ALPACA_MODEL_FILE = "/home/user/PycharmProjects/gpt4all.cpp/gpt4all-lora-quantized.bin"

    binpath = "/home/user/PycharmProjects/alpaca.cpp/chat"
    binpath = "/home/user/PycharmProjects/gpt4all.cpp/chat"

    bot = LLMcppSolver({"model": ALPACA_MODEL_FILE,
                        "executable_path": binpath})

    sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
    print(sentence)

    for q in ["Does god exist?",
              "what is the speed of light?",
              "what is the meaning of life?",
              "What is your favorite color?",
              "What is best in life?"]:
        a = bot.get_spoken_answer(q)
        print(q)
        print("    ", a)
