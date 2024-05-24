from transformers import AutoTokenizer, BertForQuestionAnswering, TrainingArguments, Trainer
import torch
import pandas as pd
import json
from datasets import Dataset

"""
}
"key": 0,
"transcript": "Turret, prepare to deploy electromagnetic pulse.
                Heading zero six five, target is grey and white fighter jet.
                Engage when ready.",
"tool": "electromagnetic pulse",
"heading": "065",
"target": "grey and white fighter jet"
}
"""

# Define model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Function to combine all json examples to pd dataframe
def raw_data_to_df(instances):
    # questions: [target, heading, tool]
    # obtain ALL context, questions and answers
    question_keys = ["What is the target to neutralize?",
                    "What is the heading in numbers?",
                    "What is the tool to neutralize the target?"]
    answer_keys = ["target", "heading", "tool"]
    context = []
    questions = []
    answers = []
    answer_start = []
    num_to_word = {"0":"zero", "1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six",
                    "7":"seven", "8":"eight", "9":"nine"}
    for _instance in instances: # each _instance has format above
        for i in range(len(question_keys)):
            _context = _instance["transcript"]
            context.append(_context)
            _question = question_keys[i]
            questions.append(_question)
            _answer = _instance[answer_keys[i]]
            if answer_keys[i]=="heading":
                _answer = " ".join([num_to_word[num] for num in [*_answer]])
            answers.append(_answer)
            answer_start.append(_context.find(_answer))
    return pd.DataFrame({'context':context,
                        'questions':questions,
                        'answers':answers,
                        'answer_start':answer_start})

# Function to preprocess data for model input
def preprocess_df(examples):
    
    questions = [q.strip() for q in examples['questions']]
    context = examples['context']
    inputs = tokenizer(
        questions, context,
        max_length=384, truncation='only_second',
        return_offsets_mapping=True, padding='max_length'
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    answer_start = examples["answer_start"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        ##################################################
        answer = answers[i]
        start_char = answer_start[i]
        end_char = start_char + len(answer)
        sequence_ids = inputs.sequence_ids(i)
        ##################################################
        index = 0
        while sequence_ids[index] != 1:
            index += 1
        context_start = index
        while sequence_ids[index] == 1:
            index += 1
        context_end = index-1
        ##################################################
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            index = context_start
            while index <= context_end and offset[index][0] <= start_char:
                index += 1
            start_positions.append(index-1)
            index = context_end
            while index >= context_start and offset[index][1] >= end_char:
                index -= 1
            end_positions.append(index+1)
        ##################################################
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Convert pd df to Dataset object
with open("../../../advanced/nlp.jsonl", "r") as f:
    instances = [json.loads(line.strip()) for line in f if line.strip() != ""]
df = raw_data_to_df(instances)
#print(df['context'].head())
dataset = Dataset.from_pandas(df)

# Split and tokenize the entire datasets
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
train_tokenized_dataset = train_dataset.map(preprocess_df, batched=True, remove_columns=dataset.column_names)
test_tokenized_dataset = test_dataset.map(preprocess_df, batched=True, remove_columns=dataset.column_names)
#print(f"Train: {train_tokenized_dataset}")
#print(f"Test: {test_tokenized_dataset}")

##################################################

training_args = TrainingArguments(
     output_dir='./results',
     evaluation_strategy='epoch',
     learning_rate= 2e-5,
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     num_train_epochs=3,
     weight_decay=0.01,
     logging_dir='./logs',
     logging_steps=10
)

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=train_tokenized_dataset,
     eval_dataset=test_tokenized_dataset,
     tokenizer=tokenizer
)

trainer.train()
model.save_pretrained('./fine-tuned-bert-qa')
tokenizer.save_pretrained('./fine-tuned-bert-qa')