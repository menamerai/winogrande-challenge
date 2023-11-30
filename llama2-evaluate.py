import threading
import re
import pandas as pd
from typing import Dict
from datasets import load_dataset
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from tqdm import tqdm

dataset = load_dataset("winogrande", "winogrande_debiased", split="train")
llm = Ollama(model="llama2-uncensored")


def infer(prompt: str, option1: str, option2: str, correct: str) -> Dict:
    winogrande_prompt = PromptTemplate.from_template(
        """
        Fill in the [BLANK] using the option that make the most logical meaning: "{prompt}"
        Option 1: {option1}
        Option 2: {option2}
        Answer only Option 1 or Option 2:
        """
    )
    prompt_filled = winogrande_prompt.format(prompt=prompt, option1=option1, option2=option2)
    model_output = llm(prompt_filled)
    option1_regex = "{}".format(option1)
    option2_regex = "{}".format(option2)
    if re.match(r"Option 1", model_output, re.IGNORECASE) or re.match(option1_regex, model_output, re.IGNORECASE):
        answer = "1"
    elif re.match(r"Option 2", model_output, re.IGNORECASE) or re.match(option2_regex, model_output, re.IGNORECASE):
        answer = "2"
    else:
        answer = "0"
    correct = "1" if correct == "1" else "2"

    output_dict = {
        "prompt": prompt,
        "option1": option1,
        "option2": option2,
        "model_output": model_output,
        "answer": answer,
        "correct": correct,
        "correctness": answer == correct,
    }
    return output_dict

if __name__ == "__main__":
    # use threading to speed up inference
    answers = []
    threads = []
    for idx in tqdm(range(100)):
        # replace all "_" with "[BLANK]"
        prompt = dataset["sentence"][idx].replace("_", "[BLANK]")
        option1 = dataset["option1"][idx]
        option2 = dataset["option2"][idx]
        correct_ans = dataset["answer"][idx]
        # print(f"[MAIN]: create and start thread {idx}")
        t = threading.Thread(target=lambda: answers.append(infer(prompt, option1, option2, correct_ans)))
        threads.append(t)
        t.start()
        
    for idx, t in enumerate(tqdm(threads)):
        # print(f"[MAIN]: join thread {idx}")
        t.join()
        # print(f"[MAIN]: thread {idx} joined")


    # get average correctness
    correctness = [answer["correctness"] for answer in answers]
    print(f"Average correctness: {sum(correctness) / len(correctness)}")

    # save answers to csv with pandas
    output_df = pd.DataFrame.from_records(answers)
    output_df.to_csv("winogrande-evaluate-100.csv", index=False)
        