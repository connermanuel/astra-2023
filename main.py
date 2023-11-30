import json
import os
import time
from copy import deepcopy

from openai import OpenAI, RateLimitError

INSTRUCTIONS = (
    "You are given a list of examples of input-label pairs. "
    "Your task is to infer how the labels were assigned to the inputs, "
    "then to assign the corresponding labels to the provided set of test inputs."
)

def evaluate_articulation(client: OpenAI, result: dict):
    messages = create_messages(result["task"])
    messages.append(
        {
            "role": "assistant",
            "content": result["return"],
        },
    )
    messages.append(
        {"role": "user", "content": "How did you make those classifications?"}
    )

    completion = get_completion(client, messages)
    return_message = completion.choices[0].message.content

    return return_message


def evaluate_articulation_with_choices(client: OpenAI, result: dict):
    messages = create_messages(result["task"])
    messages.append(
        {
            "role": "assistant",
            "content": result["return"],
        },
    )

    user_string = (
        "How did you make those classifications? Select one from the following choices.\n\n"
        f"(a) {result["choices"][0]}\n"
        f"(b) {result["choices"][1]}\n"
        f"(c) {result["choices"][2]}\n"
    )
    messages.append(
        {"role": "user", "content": user_string}
    )

    completion = get_completion(client, messages)
    return_message = completion.choices[0].message.content

    return return_message


def evaluate_classification(client: OpenAI, task: dict):
    messages = create_messages(task)
    completion = get_completion(client, messages)

    return_message = completion.choices[0].message.content

    predicted_labels = []
    for line in return_message.split("\n"):
        if "Label:" in line:
            if "True" in line:
                predicted_labels.append(True)
            elif "False" in line:
                predicted_labels.append(False)

    test_labels = [label for _, label in task["tests"]]
    accuracy = get_accuracy_from_labels(predicted_labels, test_labels)

    return {"task": task, "return": return_message, "accuracy": accuracy}


def get_completion(client: OpenAI, messages: list[str]):
    done = False
    while not done:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, temperature=0
            )
            done = True
        except RateLimitError:
            time.sleep(20)

    return completion


def create_messages(task: dict):
    system_string = (
        INSTRUCTIONS
        + "\n\nExamples:\n"
        + "".join(
            [f"Input: {input}\nLabel: {label}\n" for input, label in task["examples"]]
        )
    )

    user_string = "\n".join(f"Input: {input}" for input, _ in task["tests"])

    messages = [
        {
            "role": "system",
            "content": system_string,
        },
        {
            "role": "user",
            "content": user_string,
        },
    ]

    return messages


def get_accuracy_from_labels(predicted_labels, test_labels):
    def compare_lists(list_1, list_2):
        if len(list_1) != len(list_2):
            print("Note: two provided lists don't have the same length.")
        total = min(len(list_1), len(list_2))

        ctr = 0
        for i in range(total):
            if list_1[i] == list_2[i]:
                ctr += 1

        return ctr / total

    if isinstance(test_labels[0], list):
        return [
            compare_lists(predicted_labels, [label[i] for label in test_labels])
            for i in range(len(test_labels[0]))
        ]
    return compare_lists(predicted_labels, test_labels)

def parse_results(results: dict):    
    predicted_labels = []
    for line in results["return"].split("\n"):
        if "Label:" in line:
            if "True" in line:
                predicted_labels.append(True)
            elif "False" in line:
                predicted_labels.append(False)

    for input, result in zip(results["task"]["tests"], predicted_labels):
        print(f"Input: {input}")
        print(f"Result: {result}")


if __name__ == "__main__":
    client = OpenAI()
    with open("tasks.json", "r") as f:
        tasks = json.load(f)

    if not os.path.exists("results_cls.json"):

        results_cls = {
            task: evaluate_classification(client, tasks[task]) for task in tasks.keys()
        }

        with open("results_cls.json", "w") as f:
            json.dump(results_cls, f)
    
    else:
        with open("results_cls.json", "r") as f:
            results_cls = json.load(f)
    
    if not os.path.exists("results_art.json"):
        results_art = deepcopy(results_cls)
        for task in results_cls.keys():
            results_art[task]["articulation"] = evaluate_articulation(client, results_art[task])
        
        with open("results_art.json", "w") as f:
            json.dump(results_art, f)

    else:
        with open("results_art.json", "r") as f:
            results_art = json.load(f)

    
    if not os.path.exists("results_art_choices.json"):
        results_art_choices = deepcopy(results_art)
        for task in results_art_choices.keys():
            if "choices" in results_art_choices[task].keys():
                results_art_choices[task]["selected_choice"] = evaluate_articulation_with_choices(client, results_art[task])
        
        with open("results_art_choices.json", "w") as f:
            json.dump(results_art_choices, f)

    else:
        with open("results_art_choices.json", "r") as f:
            results_art = json.load(f)
