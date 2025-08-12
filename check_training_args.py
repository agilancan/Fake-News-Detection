from transformers import TrainingArguments
import inspect

print("Transformers version:", __import__("transformers").__version__)
print("TrainingArguments file:", TrainingArguments.__module__)
print(inspect.getsourcefile(TrainingArguments))
print("Constructor signature:", inspect.signature(TrainingArguments.__init__))
