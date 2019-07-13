#!/usr/bin/env python
# coding: utf-8

# # 生成电视剧剧本
# 
# 在这个项目中，你将使用 RNN 创作你自己的[《辛普森一家》](https://zh.wikipedia.org/wiki/%E8%BE%9B%E6%99%AE%E6%A3%AE%E4%B8%80%E5%AE%B6)电视剧剧本。你将会用到《辛普森一家》第 27 季中部分剧本的[数据集](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data)。你创建的神经网络将为一个在 [Moe 酒馆](https://simpsonswiki.com/wiki/Moe's_Tavern)中的场景生成一集新的剧本。
# 
# ## 获取数据
# 我们早已为你提供了数据`./data/Seinfeld_Scripts.txt`。我们建议你打开文档来看看这个文档内容。
# 
# >* 第一步，我们来读入文档，并看几段例子。
# * 然后，你需要定义并训练一个 RNN 网络来生成新的剧本！

# In[1]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)


# ## 探索数据
# 使用 `view_line_range` 来查阅数据的不同部分，这个部分会让你对整体数据有个基础的了解。你会发现，文档中全是小写字母，并且所有的对话都是使用 `\n` 来分割的。

# In[2]:


view_line_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


# ---
# ## 实现预处理函数
# 对数据集进行的第一个操作是预处理。请实现下面两个预处理函数：
# 
# - 查询表
# - 标记符号
# 
# ### 查询表
# 要创建词嵌入，你首先要将词语转换为 id。请在这个函数中创建两个字典：
# 
# - 将词语转换为 id 的字典，我们称它为 `vocab_to_int`
# - 将 id 转换为词语的字典，我们称它为 `int_to_vocab`
# 
# 请在下面的元组中返回这些字典
#  `(vocab_to_int, int_to_vocab)`

# In[ ]:


import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    
    # return tuple
    return (None, None)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)


# ### 标记符号的字符串
# 我们会使用空格当作分隔符，来将剧本分割为词语数组。然而，句号和感叹号等符号使得神经网络难以分辨“再见”和“再见！”之间的区别。
# 
# 实现函数 `token_lookup` 来返回一个字典，这个字典用于将 “!” 等符号标记为 “||Exclamation_Mark||” 形式。为下列符号创建一个字典，其中符号为标志，值为标记。
# 
# - period ( . )
# - comma ( , )
# - quotation mark ( " )
# - semicolon ( ; )
# - exclamation mark ( ! )
# - question mark ( ? )
# - left parenthesis ( ( )
# - right parenthesis ( ) )
# - dash ( -- )
# - return ( \n )
# 
# 这个字典将用于标记符号并在其周围添加分隔符（空格）。这能将符号视作单独词汇分割开来，并使神经网络更轻松地预测下一个词汇。请确保你并没有使用容易与词汇混淆的标记。与其使用 “dash” 这样的标记，试试使用“||dash||”。

# In[ ]:


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
        
    return None

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)


# ## 预处理并保存所有数据
# 运行以下代码将预处理所有数据，并将它们保存至文件。建议你查看`helpers.py` 文件中的 `preprocess_and_save_data` 代码来看这一步在做什么，但是你不需要修改`helpers.py`中的函数。

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# # 检查点
# 这是你遇到的第一个检点。如果你想要回到这个 notebook，或需要重新打开 notebook，你都可以从这里开始。预处理的数据都已经保存完毕。

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## 创建神经网络
# 在本节中，你会构建 RNN 中的必要 Module，以及 前向、后向函数。
# 
# ### 检查 GPU 访问权限

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


# ## 输入
# 让我们开始预处理输入数据。我们会使用 [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) 来为数据库提供一个数据格式；以及一个 [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), 该对象会实现 batching，shuffling 以及其他数据迭代功能。
# 
# 你可以通过传入 特征 和目标 tensors 来创建 TensorDataset，随后创建一个 DataLoader 。
# ```
# data = TensorDataset(feature_tensors, target_tensors)
# data_loader = torch.utils.data.DataLoader(data, 
#                                           batch_size=batch_size)
# ```
# 
# ### Batching
#  通过 `TensorDataset` 和 `DataLoader` 类来实现  `batch_data` 函数来将 `words` 数据分成 `batch_size` 批次。
# 
# >你可以使用 DataLoader 来分批 单词, 但是你可以自由设置 `feature_tensors` 和 `target_tensors` 的大小以及 `sequence_length`。
# 
# 比如，我们有如下输入:
# ```
# words = [1, 2, 3, 4, 5, 6, 7]
# sequence_length = 4
# ```
# 
# 你的第一个 `feature_tensor` 会包含:
# ```
# [1, 2, 3, 4]
# ```
# 随后的 `target_tensor` 会是接下去的一个字符值:
# ```
# 5
# ```
# 那么，第二组的`feature_tensor`, `target_tensor` 则如下所示:
# ```
# [2, 3, 4, 5]  # features
# 6             # target
# ```

# In[ ]:


from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    
    # return a dataloader
    return None

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own


# ### 测试你的 dataloader 
# 
# 你需要改写下述代码来测试 batching 函数，改写后的代码会现在的比较类似。
# 
# 下面，我们生成了一些测试文本数据，并使用了一个你上面写 dataloader 。然后，我们会得到一些使用`sample_x`输入以及`sample_y`目标生成的文本。
# 
# 你的代码会返回如下结果(通常是不同的顺序，如果你 shuffle 了你的数据):
# 
# ```
# torch.Size([10, 5])
# tensor([[ 28,  29,  30,  31,  32],
#         [ 21,  22,  23,  24,  25],
#         [ 17,  18,  19,  20,  21],
#         [ 34,  35,  36,  37,  38],
#         [ 11,  12,  13,  14,  15],
#         [ 23,  24,  25,  26,  27],
#         [  6,   7,   8,   9,  10],
#         [ 38,  39,  40,  41,  42],
#         [ 25,  26,  27,  28,  29],
#         [  7,   8,   9,  10,  11]])
# 
# torch.Size([10])
# tensor([ 33,  26,  22,  39,  16,  28,  11,  43,  30,  12])
# ```
# 
# ### 大小
# 你的 sample_x 应该是 `(batch_size, sequence_length)`的 大小 或者是(10, 5)， sample_y 应该是 一维的: batch_size (10)。
# 
# ### 值
# 
# 你应该也会发现 sample_y, 是 test_text 数据中的*下一个*值。因此，对于一个输入的序列 `[ 28,  29,  30,  31,  32]` ，它的结尾是 `32`, 那么其相应的输出应该是 `33`。

# In[ ]:


# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)


# ---
# ## 构建神经网络
# 使用 PyTorch [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module) 来实现一个 循环神经网络 RNN。你需要选择一个 GRU 或者 一个 LSTM。为了完成循环神经网络。为了实现 RNN，你需要实现以下类:
#  - `__init__` - 初始化函数
#  - `init_hidden` - LSTM/GRU 隐藏组昂泰的初始化函数
#  - `forward` - 前向传播函数
#  
# 初始化函数需要创建神经网络的层数，并保存到类。前向传播函数会使用这些网络来进行前向传播，并生成输出和隐藏状态。
# 
# 在该流程完成后，**该模型的输出是 *最后的* 文字分数结果** 对于每段输入的文字序列，我们只需要输出一个单词，也就是，下一个单词。 
# 
# ### 提示
# 
# 1. 确保 lstm 的输出会链接一个 全链接层，你可以参考如下代码 `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
# 2. 你可以通过 reshape 模型最后输出的全链接层，来得到最终的文字分数:
# 
# ```
# # reshape into (batch_size, seq_length, output_size)
# output = output.view(batch_size, -1, self.output_size)
# # get last batch
# out = output[:, -1]
# ```

# In[ ]:


import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        
        # define model layers
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function   

        # return one batch of output word scores and the hidden state
        return None, None
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        return None

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_rnn(RNN, train_on_gpu)


# ### 定义前向及后向传播
# 
# 通过你实现的 RNN 类来进行前向及后项传播。你可以在训练循环中，不断地调用如下代码来实现：
# ```
# loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
# ```
# 
# 函数中需要返回一个批次以及其隐藏状态的loss均值，你可以调用一个函数`RNN(inp, hidden)`来实现。记得，你可以通过调用`loss.item()` 来计算得到该loss。
# 
# **如果使用 GPU，你需要将你的数据存到 GPU 的设备上。**

# In[ ]:


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    
    # move data to GPU, if available
    
    # perform backpropagation and optimization

    # return the loss over a batch and the hidden state produced by our model
    return None, None

# Note that these tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)


# ## 神经网络训练
# 
# 神经网络结构完成以及数据准备完后，我们可以开始训练网络了。
# 
# ### 训练循环
# 
# 训练循环是通过 `train_decoder` 函数实现的。该函数将进行 epochs 次数的训练。模型的训练成果会在一定批次的训练后，被打印出来。这个“一定批次”可以通过`show_every_n_batches` 来设置。你会在下一节设置这个参数。

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


# ### 超参数
# 
# 设置并训练以下超参数:
# -  `sequence_length`，序列长度 
# -  `batch_size`，分批大小
# -  `num_epochs`，循环次数
# -  `learning_rate`，Adam优化器的学习率
# -  `vocab_size`，唯一标示词汇的数量
# -  `output_size`，模型输出的大小 
# -  `embedding_dim`，词嵌入的维度，小于 vocab_size
# -  `hidden_dim`， 隐藏层维度
# -  `n_layers`， RNN的层数
# -  `show_every_n_batches`，打印结果的频次
# 
# 如果模型没有获得你预期的结果，调整 `RNN`类中的上述参数。

# In[ ]:


# Data params
# Sequence Length
sequence_length =   # of words in a sequence
# Batch Size
batch_size = 

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)


# In[ ]:


# Training parameters
# Number of Epochs
num_epochs = 
# Learning Rate
learning_rate = 

# Model parameters
# Vocab size
vocab_size = 
# Output size
output_size = 
# Embedding Dimension
embedding_dim = 
# Hidden Dimension
hidden_dim = 
# Number of RNN Layers
n_layers = 

# Show stats for every n number of batches
show_every_n_batches = 500


# ### 训练
# 下一节，通过预处理数据来训练神经网络。如果你的loss结果不好，可以通过调整超参数来修正。通常情况下，大的隐藏层及层数会带来比较好的效果，但同时也会消耗较长的时间来训练。
# > **你应该努力得到一个低于3.5的loss** 
# 
# 你也可以试试不同的序列长度，该参数表明模型学习的范围大小。

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')


# ### 问题: 你如何决定你的模型超参数？
# 比如，你是否试过不同的 different sequence_lengths 并发现哪个使得模型的收敛速度变化？那你的隐藏层数和层数呢？你是如何决定使用这个网络参数的？

# **答案:** (在这里写下)

# ---
# # 检查点
# 
# 通过运行上面的训练单元，你的模型已经以`trained_rnn`名字存储，如果你存储了你的notebook， **你可以在之后的任何时间来访问你的代码和结果**. 下述代码可以帮助你重载你的结果!

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')


# ## 生成电视剧剧本
# 你现在可以生成你的“假”电视剧剧本啦！
# 
# ### 生成文字
# 你的神经网络会不断重复生成一个单词，直到生成满足你要求长度的剧本。使用 `generate` 函数来完成上述操作。首先，使用 `prime_id` 来生成word id，之后确定生成文本长度 `predict_len`。同时， topk 采样来引入文字选择的随机性!

# In[ ]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences


# ### 生成一个新剧本
# 是时候生成一个剧本啦。设置`gen_length` 剧本长度，设置 `prime_word`为以下任意词来开始生成吧:
# - "jerry"
# - "elaine"
# - "george"
# - "kramer"
# 
# 你可以把prime word 设置成 _任意 _ 单词, 但是使用名字开始会比较好(任何其他名字也是可以哒!)

# In[ ]:


# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'jerry' # name for starting the script

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)


# #### 存下你最爱的片段
# 
# 一旦你发现一段有趣或者好玩的片段，就把它存下啦！

# In[ ]:


# save script to a text file
f =  open("generated_script_1.txt","w")
f.write(generated_script)
f.close()


# # 这个电视剧剧本是无意义的
# 如果你的电视剧剧本不是很有逻辑也是ok的。下面是一个例子。
# 
# ### 生成剧本案例
# 
# >jerry: what about me?
# >
# >jerry: i don't have to wait.
# >
# >kramer:(to the sales table)
# >
# >elaine:(to jerry) hey, look at this, i'm a good doctor.
# >
# >newman:(to elaine) you think i have no idea of this...
# >
# >elaine: oh, you better take the phone, and he was a little nervous.
# >
# >kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.
# >
# >jerry: oh, yeah. i don't even know, i know.
# >
# >jerry:(to the phone) oh, i know.
# >
# >kramer:(laughing) you know...(to jerry) you don't know.
# 
# 
# 如果这个电视剧剧本毫无意义，那也没有关系。我们的训练文本不到一兆字节。为了获得更好的结果，你需要使用更小的词汇范围或是更多数据。幸运的是，我们的确拥有更多数据！在本项目开始之初我们也曾提过，这是[另一个数据集](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data)的子集。我们并没有让你基于所有数据进行训练，因为这将耗费大量时间。然而，你可以随意使用这些数据训练你的神经网络。当然，是在完成本项目之后。
# # 提交项目
# 在提交项目时，请确保你在保存 notebook 前运行了所有的单元格代码。请将 notebook 文件保存为 "dlnd_tv_script_generation.ipynb"，并将它作为 HTML 文件保存在 "File" -> "Download as" 中。请将 "helper.py" 和 "problem_unittests.py" 文件一并打包成 zip 文件提交。

# In[ ]:




