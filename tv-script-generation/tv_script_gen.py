import helper
import numpy as np
import tensorflow as tf
import problem_unittests as tests

# data_dir = './data/simpsons/moes_tavern_lines.txt'
# text = helper.load_data(data_dir)
# # Ignore notice, since we don't use it for analysing the data
# text = text[81:]

# def create_lookup_tables(text):
#     """
#     Create lookup tables for vocabulary
#     :param text: The text of tv scripts split into words
#     :return: A tuple of dicts (vocab_to_int, int_to_vocab)
#     """
#     vocab = set(text)
#     vocab_to_int = {c: i for i, c in enumerate(vocab)}
#     int_to_vocab = dict(enumerate(vocab))
#     return (vocab_to_int, int_to_vocab)

# def token_lookup():
#     """
#     Generate a dict to turn punctuation into a token.
#     :return: Tokenize dictionary where the key is the punctuation and the value is the token
#     """
#     # TODO: Implement Function
#     return {
#         '.': '||Period||',
#         ',': '||Comma||', 
#         '"': '||Quotation_Mark||',
#         ';': '||Semicolon||',
#         '!': '||Exclamation_Mark||',
#         '?': '||Question_Mark||',
#         '(': '||Left_Parentheses||',
#         ')': '||Right_Parentheses||',
#         '--': '||Dash||',
#         '\n': '||Return||'
#     }

# tests.test_create_lookup_tables(create_lookup_tables)
# tests.test_tokenize(token_lookup)

# Preprocess Training, Validation, and Testing Data
# helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inp = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return inp, targets, learning_rate

# tests.test_get_inputs(get_inputs)

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(256)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm] * rnn_size)
    initial_state = rnn_cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name="initial_state")
    return rnn_cell, initial_state

# tests.test_get_init_cell(get_init_cell)

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed_tensor = tf.nn.embedding_lookup(embedding, input_data)
    return embed_tensor

# tests.test_get_embed(get_embed)

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, "final_state")
    return outputs, final_state

# tests.test_build_rnn(build_rnn)

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    inputs = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, inputs)
    logits = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=vocab_size, activation_fn=None)

    return logits, final_state

# tests.test_build_nn(build_nn)

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    words_per_batch = batch_size * seq_length
    n_batches = len(int_text) // words_per_batch
    n_words = n_batches * words_per_batch
    batch_offset = n_words // batch_size
    input_arr = int_text[:n_words]
    target_arr = int_text[1:n_batches * words_per_batch] + int_text[:1]

    inputs = [ [ [input_arr[nb*seq_length + bs*batch_offset + s] for s in range(0,seq_length) ] for bs in range(0,batch_size)] for nb in range(0,n_batches)]
    targets = [ [ [target_arr[nb*seq_length + bs*batch_offset + s] for s in range(0,seq_length) ] for bs in range(0,batch_size)] for nb in range(0,n_batches)]

    batches = np.array([ [inputs[i], targets[i]] for i in range(0,n_batches)])

    print(batches)
    
    return batches

get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)    

tests.test_get_batches(get_batches)

# int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
