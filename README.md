<h2>Usefulness and Applications</h2>
Word2vec has multiple applications, in standard machine learning context it is very tough to retain information from a single word unless more information is given. In most forms are data, for a standard sized image there could be thousands of bytes of information, but for words this is not the case. In the case of words it is very simple to imagine each word as one particular number, making the total information that the sentence contains minisculey small compared to the image that we may process to have the same amount of information. For example from a machine’s perspective when it hears the word “king” it doesn’t know anything about it, but when we hear that word we automatically associate it with royalty and male. Word2vec essentially gives the word more information and gives the machine a sort of “intuition” into what a word means.
<br/><br/>
For and example of how this kind of works with our mind let's say that we said "a cat riding a toy horse" an image like this might pop up in your mind <br/>
 
 
![alt text](https://github.com/supersteph/ro_sgns/blob/master/images/Lr7J8ab.jpg "cat on a horse")
 
 
However, the data obtained by strictly the words in the sentence is 22 bytes, while the image itself is 18 kb and almost 1000 times difference between the information obtained from text and information obtained by image.<br/>
 
By using word2vec the information is turned from a sparse vector into a dense one, and this allows the machine to do things with it.<br/>
 
 <h2>Basic Concept of Skip-Gram Model</h2>
 
Word2vec is essentially a type of unsupervised learning, but it is trained using supervised learning. It starts with the basic idea that words that appear in similar context will have similar meanings. With this in mind we try to to maximize the probability of the context word from the actual word. This model with fake model has two layers a embedding layer and then a output layer with the contexts that implement softmax regression. This embedding layer will be size [vocab_size,embedding_size] since the input to this layer will be a one hot encoding of size [vocab_size], the product will be just a single row of that matrix. So every single row of the matrix becomes a vector, and then u pass the vector through an output layer. 
<br/>
 
![alt text](https://github.com/supersteph/ro_sgns/blob/master/images/word2vec_weight_matrix_lookup_table.png "Layers")
 
 
Every neuron is dot producted by the embedding of the current context, so this works out.
 
This layer outputs a vector that is size [vocab_size] this vector goes through a softmax layer so that each of the outputs become probabilities of a certain word being in the context.<br/>
 
<h2>Clever Tricks</h2>
 
This basic model works well but there are several downfalls to the basic mode, it is very slow and it takes a lot of time to train. It requires a lot training data and is all-around very hard to do. They do three things that make it slightly better: they treat a couple of words as phrases, they subsample words so that the frequent words may be removed more frequently to keep the data relatively. 
#Phrasing
The main idea behind phrasing is that certain sequence of words should be treated as one word, and words of certain caliber could be treated not as phrases but as single words. A prime example of this disparity between the addition of words is the word “Boston” and the word “Globe” these words combined is a newspaper and should be treated like a different object that a globe that comes from boston. The phrasing algorithim is not going to be gone over in this blog but if you want to read on it the link is over [here](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/)
<br/><br/>
The adding of phrases turns the already big vocabulary humongous but ultimately allows for the model to have a better understanding of words. 
#SubSampling
The main purpose of SubSampling is to remove words from the context that do not provide any meaning to the current word, or appear too frequently. As it turns out, words that don’t provide much meaning and appear too frequently are often the same words. For example, the word “the” the only new information captured by the word “the” is that perhaps the word following is a noun. However “the” appears too many times then is necessary to learn a good vector for the word. So we decide to remove some of these words from the context, which provides us with enough information to embed these words. The probability of removing a particular word from the context is a function with the frequency as a variable. This function removes words that appear more frequently with higher probability.
 
#Negative Sampling
Negative Sampling is used to generate a random sample which you label the weight zero and then it works better and creates more random data. Negative Sampling usually takes 5 words and apply negative weights to these five words. The point of this is to minimize the weights that you update, if you put negative weights on everything other than your current word two things happen, you are updating the entire model every single training iteration which leads to training being extremely slow. You are also doing some work in the opposite way because you might be labelling things negative that may appear in other contexts. For example, just because the word “person” appears in the context of the word “fat” it does not mean that “person” can’t also appear in the context of the word “tall”. On the other hand if you do not provide any other negative weights your training ends up very slow once again, and the model might be able to train things you know aren’t supposed to appear in that context.You find a bunch of words that aren’t your current word and say that you don’t want those to show up at all this means that you don’t have to update all the weights the entire time. Instead of updating everything at one time by only putting some of the things as negative it allows the model to update the output layer much faster, while the embedding layer is unchanged from the previous version. The reason you want to pick the words that come more frequently is logic similar to the subsampling idea, the more often a word appears, the more likely it is to be a filler word and have no correlation to the current word.<br/>
 
<h1>SGNS</h1>
Previously we explained how we use the context embeddings and the word embedding to create a probability that the context is related to the word. This approach may be kind of limited in a sort of way because you are directly changing the embedding instead of making the objective better. By directly taking the gradient with respect to X instead of taking it with respect to C and W we can try to keep the rank of x low and the rank of c and w relatively low as well. By keeping the rank low it allows the model to peform better.
 
With this in mind we design the loss function as followed
 
LOSS = log σ(hw, ci)+#(w)#(c)
+k/|D|*log σ(−hw, ci)*#w*#c
 
The important thing to note about this loss function is that this function is designed as the scalar product of w and c instead of relying on w and c individually.
 
This loss function leads us to a gradient like this, note how it is taken with respect to X instead of taken with respect to w and c. It is an important thing to note how to get from X to the W and C embeddings since we are no longer taking the gradient with respect to those two anymore. The way that we do this is by using Singular Value Decomposition on matrix X.
#SVD
The basic idea behind Singular Value Decomposition is to take an m*n matrix M and decompose it into three separate matrices, a matrix U, a matrix Σ and a matrix V. The product of U* Σ *V would be equal to M. Each of these Matrices have special properties, if U is multiplied by its inverse it would result in a Identity matrix, Σ is a diagonal matrix of size m*n, and then V is a n*n matrix that is also a unitary matrix. After this the only thing we need to do is to find W and C from the U  Σ V. We define W as U * Σ and C as equal to  Σ*V. The reason why we divvy it up like this is because it is proven to work the best for this case, but varying it could lead to different outputs. But it is shown over here that this way works well.
<br/>
 
<br/>
 
#Projector Splitting
The general idea behind gradient descent is that xi+1 = xi + ∇F(X), while this may work for conventional gradient descent it does not work for our case. By adding the gradient, it brings the rank super high, so to bring down the rank we use a retractor with the sole purpose of brining down a rank of a matrix.
Projector Splitting is based on this idea it uses a combination of SVD and QR decompositions to find xi+1 the benefits of using projector splitting is that you do not need an 
