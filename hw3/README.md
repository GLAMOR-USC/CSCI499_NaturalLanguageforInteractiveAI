
# Coding Assignment #3

In this assignment, you will implement an Encoder-Decoder model that takes in [ALFRED](https://askforalfred.com/) instructions for an entire episode and predicts the sequence of corresponding, high-level actions and target objects. The problem is the same as in Homework #1 except you will model this as a sequence prediction problem instead of multiclass classification. You will implement a sequence-to-sequence (seq2seq) model. 

ALFRED instructions were written by Mechanical Turk workers who aligned them to a virtual agent's behavior in a recorded simulation-based video of the agent doing a task in a room.

For example, the instructions for [an example task](https://askforalfred.com/?vid=8781) and their associated high-level action and object targets are:

| Instruction                                                                                                                          | Action       | Target     |
| :----------------------------------------------------------------------------------------------------------------------------------- | ------------:| ----------:|
| Go straight and to the left to the kitchen island.                                                                                   | GotoLocation | countertop |
| Take the mug from the kitchen island.                                                                                                | PickupObject | mug        |
| Turn right, go forward a little, turn right to face the fridge.                                                                      | GotoLocation | fridge     |
| Put the mug on the lowest level of the top compartment of the fridge. Close then open the fridge door. Take the mug from the fridge. | CoolObject   | mug        |
| Turn around, go straight all the way to the counter to the left of the sink, turn right to face the sink.                            | GotoLocation | sinkbasin  |
| Put the mug in the sink.                                                                                                             | PutObject    | sinkbasin  |

Initially, you should implement a encoder-decoder seq2seq model that encodes the low-level instructions into a context vector which is decoded autoregressively into the high-level instruction. Then you will implement an attention mechanism that allows the decoder to attend to each hidden state of the encoder model when making predictions. Finally, you will compare this against a Transformer-based model. You may use any functionality in the HuggingFace library for these implementations. (That is, we do not expect you to implement RNNs, LSTMs, Attention layers, Tranformers, or any other architectural component from scratch.)

We provide starter code that tokenizes the instructions and provides dictionaries mapping instruction tokens to their numerical indexes. It's up to you to write methods that convert the inputs and outputs to tensors, an encoder-decoder attention model that processes input tensors to produce predictions, and the training loop to adjust the parameters of the model based on its predictions versus the ground truth, target outputs. Note, you will need to implement some function for decoding the target text given the context vector. For this decoding to work, you will need to append special tokens to the input to mark the beginning of sentence (\<BOS\>) and end of sentence (\<EOS\>). This is a standard practice for decoder models. 

You will evaluate your model as it trains against both the training data that it is seeing and validation data that is "held out" of the training loop. 

## Clone this repo
```
git clone https://github.com/GLAMOR-USC/CSCI499_NaturalLanguageforInteractiveAI.git

cd CSCI499_NaturalLanguageforInteractiveAI/hw3

export PYTHONPATH=$PWD:$PYTHONPATH
```

## Install some packages

```
# first create a virtualenv 
virtualenv -p $(which python3) ./hw3

# activate virtualenv
source ./hw1/bin/activate

# install packages
pip3 install -r requirements.txt
```

## Train model

The training file will throw some errors out of the box. You will need to fill in the TODOs before anything starts to train.
While debugging, consider taking a small subset of the data and inserting break statements in the code and print the values of your variables.

```
Train:
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu 

Evaluation:
python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/s2s \
    --batch_size=1000 \
    --num_epochs=100 \
    --val_every=5 \
    --force_cpu \
    --eval


# add any additional argments you may need
# remove force_cpu if you want to run on gpu
```


## Grading
This assignment will be scored out of 40 points on the *correctness* and *documentation detail and accuracy* of the following aspects of your code:

- [ ] (5pt) Encoding the input and output data into tensors. Adding special tokens to denote the start and end of decoding.
- [ ] (10pt) Implementation of the base encoder-decoder model without attention (you may use RNN, LSTM, or any other recurrent neural net cell for this step) that takes in a sequence of language instructions for an episode and predicts a sequence of (action, target) high-level goal outputs.
- [ ] (5pt) Implementation and of an attention model over the input hidden states of the previous model; note that this attention could be over word-level states if your encoder treats all instructions as a single sequence, or over the instruction-level states if your encoder is hierarchical and consumes instructions first, then sequences of instruction hidden states second.
- [ ] (5pt) Implement a Transformer-based encoder-decoder and compare performance against encoder-decoder model with attention.
- [ ] (5pt) Implementation of the training and evaluation loops
- [ ] (10pt) *Report* your results through an .md file in your submission; discuss your implementation choices and document the performance of your models (both training and validation performance) under the conditions you settled on. Compare the base encoder-decoder, encoder-decoder with attention, and transformer-based model performance. Discuss your encoder-decoder attention choices (e.g., flat or hierarchical, recurrent cell used, etc.). Discuss the attention mechanism you implemented for the encoder-decoder model using the taxonomy we discussed in class. 

Remember that coding assignments 1 and 2 were worth 30 points, bringing the total to 100 points after adding these 40, which all told account for 25% of your course grade.

## Available Bonus Points

You may earn up to 10pt of *bonus points* by implementing the following bells and whistles that explore further directions. For these, you will need to compare the performance of the base model against whatever addition you try. Add those details to your report. If you implement bonus items, your base code implementing the main assignment must remain intact and be runnable still.

- [ ] (*10pt*) In addition to language instructions, the ALFRED dataset also provides visual features corresponding to the agent's egocentric observations encoded using a pretrained ResNet model. Download the ResNet features and modify your model (either encoder-decoder or Transformer) to use the ResNet features (e.g., you could create a single visual observation alongside the lang one by argmaxing the ResNet features thru time and concatenating them to the final encoder hidden state) and test whether this improves the prediction performance. Discuss your implementation, the performance differences, and whether this information is "fair".
- [ ] (*10pt*) We discussed global vs local and soft vs. hard attention in class. Implement the other of one of these and compare to your base implementation. Up to 5pt for comparing global versus local attention. Up to 5pt for comparing soft versus hard attention. For the latter, note that some tricks are needed to preserve gradients (see slides for paper link).
- [ ] (*5pt*) The most straightforward encoder-decoder implementation will treat words in the instructions as one long sequence. Instead, try encoding each instruction with one RNN, then encoding each of the final latent states of that "low-level" RNN using a "high-level" RNN whose inputs are those final hidden states. How does the performance compare?
- [ ] (*5pt*) The low-level versus high-level encoding can also be tried using the Transformer-based model, where the final representation for each low-level instruction is the CLS token representation output of the low-level Transformer. The high-level Transformer will consume an output CLS representation from each instruction to produce the final input to the decoder.
- [ ] (*5pt*) Instead of learning word embeddings from scratch, initialize your embeddings with pretrained vectors like Word2Vec, GLoVe, or Fasttext.
- [ ] (*10pt*) Using your code from HW2, try learning initial word embeddings via the language modeling objectives of your own SkipGram / CBOW model on the ALFRED instruction data as a pretraining step. How does pretraining on in-domain ALFRED instruction data affect downstream performance?