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
python3 -m pip install torch numpy sentencepiece
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

see [GitHub - ymcui/Chinese-LLaMA-Alpaca: 中文LLaMA&amp;Alpaca大语言模型+本地部署 (Chinese LLaMA &amp; Alpaca LLMs)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)






