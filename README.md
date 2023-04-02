## LLaMA on Mac M1

#### before

env:

```shell
# recommend use python 3.9, cause sentencepiece does not yet support 3.11
python3 -V
Python 3.9.6
pip3 -V
pip 23.0.1 from /Users/david/Library/Python/3.9/lib/python/site-packages/pip (python 3.9)
```

#### step1 download LLaMA models:

total space: 264G

```shell
mkdir llama
cd llama
wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model
wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk

# 7B models
mkdir 7B 
wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./7B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./7B/params.json
wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./7B/checklist.chk
# 13B models
mkdir 13B
wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.00.pth -O ./13B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.01.pth -O ./13B/consolidated.01.pth
wget https://agi.gpt4.org/llama/LLaMA/13B/params.json -O ./13B/params.json
wget https://agi.gpt4.org/llama/LLaMA/13B/checklist.chk -O ./13B/checklist.chk
#30B models
wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.00.pth -O ./30B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.01.pth -O ./30B/consolidated.01.pth
wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.02.pth -O ./30B/consolidated.02.pth
wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.03.pth -O ./30B/consolidated.03.pth
wget https://agi.gpt4.org/llama/LLaMA/30B/params.json -O ./30B/params.json
wget https://agi.gpt4.org/llama/LLaMA/30B/checklist.chk -O ./30B/checklist.chk
#65B models
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.00.pth -O ./65B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.01.pth -O ./65B/consolidated.01.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.02.pth -O ./65B/consolidated.02.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.03.pth -O ./65B/consolidated.03.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.04.pth -O ./65B/consolidated.04.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.05.pth -O ./65B/consolidated.05.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.06.pth -O ./65B/consolidated.06.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.07.pth -O ./65B/consolidated.07.pth
wget https://agi.gpt4.org/llama/LLaMA/65B/params.json -O ./65B/params.json
wget https://agi.gpt4.org/llama/LLaMA/65B/checklist.chk -O ./65B/checklist.chk
```

#### step2 get llama.cpp

```shell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# 1. if yuu already installed this libs, ignore this
pip3 install torch numpy sentencepiece
# 2. convert the 7B model to ggml FP16 format
# after this, output {llama_home}/13B/ggml-model-f16.bin
python3 convert-pth-to-ggml.py {llama_home}/13B/ 1
# 3. quantize the model to 4-bits (using method 2 = q4_0)
# after this, output {llama_home}/13B/ggml-model-q4_0.bin
./quantize {llama_home}/13B/ggml-model-f16.bin {llama_home}/13B/ggml-model-q4_0.bin 2
# 4. run main to enter the chat interactive command line interface
./main -m /Volumes/Agilest/AI/models/LLaMA/13B/ggml-model-q4_0.bin  --color -f ./prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
```

#### Chinese support

1. #### source install hugging face transforms

```
# if you already installed transformers, uninstall first
#
git clone https://github.com/huggingface/transformers.git
cd tramsformers
python setup.py install
```

2. see [GitHub - ymcui/Chinese-LLaMA-Alpaca: 中文LLaMA&amp;Alpaca大语言模型+本地部署 (Chinese LLaMA &amp; Alpaca LLMs)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)



### Study:

The following is from: https://github.com/nomic-ai/gpt4all

you can clone and run it om Mac M1

```shell
# download the gpt4all models
https://huggingface.co/Sosaka/GPT4All-7B-4bit-ggml
# clone gpt4all
git clone https://github.com/nomic-ai/gpt4all.git
cd gpt4all/chat
mv ../../GPT4All-7B-4bit-ggml/gpt4all-lora-quantized.bin .
./gpt4all-lora-quantized-OSX-m1
# enjoy it
```



What's the ggml?

> The Genetic Global Modeling Language (GGML) is a language that can be used to describe and share models, algorithms, experiments or any other computational artifact. It was developed by Google Brain team as an alternative for sharing their research results with others in order to make it easier to reproduce the work done on deep learning projects. GGML allows users to define different components of a model like layers, activation functions, optimizers and hyperparameters that can be shared across multiple models or even other languages such as Python

What's the GGJT?

> The Genetic Algorithm for Joint Training (or simply, Generic Global Optimization) is a methodology that can be used in conjunction with machine learning to improve performance on tasks where there are no labeled data available. It works by optimizing hyperparameters of the model and then using those optimized values as initialization parameters when training from scratch or fine-tuning an already existing network architecture, which allows for better generalization capabilities than starting directly at random weights (which is what happens in most cases)

Why do I need to convert LLaMA model files convert to GGJT-compatible files ?

> The reason for converting LLAM models into GGJT compatible format is because it allows them to be used in a wider range of applications. The GGJT format can handle more complex and larger datasets than the LLaMA model, which makes it an ideal choice when working with large-scale data sets that require greater computational power or memory resources for processing


