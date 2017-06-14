<title> Word2Vec with Projector Splitting </title>
<b>Usefulness</b>
Word2vec has multiple applications, in standard machine learning context it is very tough to retain information from a single word unless more information is given. For example from a machine’s perspective when it hears the word “king” it doesn’t know anything about it, but when we hear that word we automatically associate it with royalty and male. Word2vec essentially gives the word more information and gives the machine a sort of “intuition” into what a word means. By using word2vec the information is turned from a sparse vector into a dense one, and this allows the machine to do things with it.
 
Word2vec is essentially a type of unsupervised learning, but it is trained using supervised learning. It starts with the basic idea that words that appear in similar context will have similar meanings. With this in mind we try to to maximize the probability of the context word from the actual word. This model with fake model has two layers a embedding layer and then a output layer with softmax regression. This embedding layer will be size [vocab_size,embedding_size] since the input to this layer will be a one hot encoding, the product will be just a single row of that matrix. So every single row of the matrix becomes a vector, and then u pass the vector through an output layer. Every neuron is dot producted by the embedding of the current context, so this works out. This layer outputs a vector that is size [voca_size] this vector goes through a softmax layer so that each of the outputs become probabilities of a certain word being in the context.
 
 Tricks of the trade
 
This basic model works well but there are several downfalls to the basic mode, it is very slow and it takes a lot of time to train. It requires a lot training data and is all-around very hard to do. They do three things that make it slightly better: they treat a couple of words as phrases, they subsample words so that the frequent words may be removed more frequently to keep the data relatively. This does two things, most frequent words such as the are filler words to make the sentence grammatically correct.  It uses this formula. Negative Sampling is used to generate a random sample which you label the weight zero and then it works better and creates more random data. You find a bunch of words that aren’t your current word and say that you don’t want those to show up at all this means that you don’t have to update all the weights the entire time. Instead of updating everything at one time by only putting some of the things as negative it allows the model to update the output layer much faster, while the embedding layer is unchanged from the previous version. The reason you want to pick the words that come more frequently is logic similar to the subsampling idea, the more often a word appears, the more likely it is to be a filler word and have no correlation to the current word.
 
SGNS
Previously we explained how we use the context embeddings and the word embedding to create a probability that the context is related to the word. This approach may be kind of limited in a sort of way because you are directly changing the embedding instead of making the objective better. By directly taking the gradient with respect to X instead of taking it with respect to C and W we can try to keep the rank of x low and the rank of c and w relatively low as well. By keeping the rank low it allows the model to peform better.
 
With this in mind we design the loss function as followed
()
This loss function only depends on the product of the two embeddings
 
Explanation of SVD
Use of SVD to derive the current  W and current C
 
A good test for the closeness of two vectors is the cosine similarity test,
 
 
The standard grandient descent is shown as followed, but the rank still comes out too high when you just use this method
 
Use a retrator operator to bring the rank to where we want it to be so that the next step is ready for gradient acsent too
