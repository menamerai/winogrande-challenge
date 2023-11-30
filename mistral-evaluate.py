import re
import pandas as pd
from typing import Dict
from datasets import load_dataset
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from tqdm import tqdm

dataset = load_dataset("winogrande", "winogrande_debiased")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]
llm = Ollama(model="mistral", temperature=0.8, top_p=0.9, top_k=0)


def infer(prompt: str, option1: str, option2: str, correct: str) -> Dict:
    # chain-of-thought prompt
    winogrande_prompt = PromptTemplate.from_template(
        """
        John moved the couch from the garage to the backyard to create space. The _ is small. "garage" or "backyard"?
        A: The question is asking about whether the garage or the backyard is small. The couch was moved from the garage to make space, so the garage is small. The answer is "garage".
        The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently. "Justin" or "Robert"?
        A: The question is asking about who has terrible nerves. Robert was diagnosed with anxiety, which has symptoms of nervousness. The answer is "Robert".
        Dennis drew up a business proposal to present to Logan because _ wants his investment. "Dennis" or "Logan"?
        A: The question is asking about who wants the investment. Dennis drew up the business proposal, so he wants the investment. The answer is "Dennis".
        Felicia unexpectedly made fried eggs for breakfast in the morning for Katrina and now _ owes a favor. "Felicia" or "Katrina"?
        A: The question is asking about who owes a favor. Katrina received Felicia's breakfast, but hasn't done anything in return, so she owes a favor. The answer is "Katrina".
        My shampoo did not lather easily on my Afro hair because the _ is too dirty. "shampoo" or "hair"?
        A: The question is asking about what is dirty. The shampoo did not lather easily because dirty hair is hard to apply shampoo on. The answer is "hair".
        {prompt} "{option1}" or "{option2}"?
        A: 
        """
    )
    prompt_filled = winogrande_prompt.format(prompt=prompt, option1=option1, option2=option2)
    model_output = llm(prompt_filled)
    stripped = str(model_output).strip()
    option1_regex = f"The answer is \"{option1}\""
    option2_regex = f"The answer is \"{option2}\""
    if re.search(option1_regex, stripped, re.IGNORECASE):
        answer = "1"
    elif re.search(option2_regex, stripped, re.IGNORECASE):
        answer = "2"
    else:
        print(f"[ERROR]: {stripped}")
        answer = "0"

    output_dict = {
        "prompt": prompt,
        "option1": option1,
        "option2": option2,
        "model_output": model_output,
        "stripped": stripped,
        "answer": answer,
        "correct": correct,
        "correctness": answer == correct,
    }
    return output_dict

if __name__ == "__main__":
    # use threading to speed up inference
    answers = []
    for idx in tqdm(range(10)):
        # replace all "_" with "[BLANK]"
        prompt = val_dataset["sentence"][idx]
        option1 = val_dataset["option1"][idx]
        option2 = val_dataset["option2"][idx]
        correct_ans = val_dataset["answer"][idx]
        # print(f"[MAIN]: create and start thread {idx}")
        answers.append(infer(prompt, option1, option2, correct_ans))


    # get average correctness
    correctness = [answer["correctness"] for answer in answers]
    print(f"Average correctness: {sum(correctness) / len(correctness)}")

    # save answers to csv with pandas
    output_df = pd.DataFrame.from_records(answers)
    output_df.to_csv("./output/mistral-evaluate-{}.csv".format(len(answers)), index=False)
        