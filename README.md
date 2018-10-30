# DL-finance-project
The repository for the code of our work on the project so far


So far we have read and understood the selected paper. We chose to focus our attention on the second application of the paper: the order book depth of futures on a given underlying. The paper tries to train a Neural Network to predict the variations of the mid price of futures on a stock each nanosecond depending on the structure of the order book. The main objective being to be able to detect selling agressors i.e. a market crossing limit order. In order to do so, the paper uses an undersampling and oversampling technique in order to label the mid price movment in a pertinent manner. 
We have, so far, reproduced the structure of the neural network used in the paper and the sampling procedure. 
We also extended their work by coding an alternative structure of neural network: the LSTM. Such a choice is motivated by the success LSTM networks have in the litterature when used in Spatio-Temporal prediction problems. LSTM networks have the capacity to retain certain informations through time and get rid of another part. In this sense, it successfully retains long term structure  as the periodicity of a phenomen (the aggressor sell apparition )and short term variations: the mid price variation functions of the actual state of the order book.
