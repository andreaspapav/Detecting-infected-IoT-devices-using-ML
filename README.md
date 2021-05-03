# Detecting infected Internet of Things (IoT) devices using Machine Learning
My dissertation project as a Master's student at UCL. 

The number of attacks on Internet of Things (IoT) devices is escalating over the last 
few years, currently consisting of a third of all attacks. In this work, a suggested 
solution to this increasing problem is put to the test. The suggestion is using Machine 
Learning to filter the network traffic produced by IoT devices and find infected devices. 
More specifically by deploying the classifier on the network router there is direct 
access to the whole network traffic. Different algorithms are tested to find the best 
performing one. The performance of the algorithms is further tested by keeping 
specific families of attacks out of the training set and predicting on them later. In this 
way the performance of the classifier against newly created attacks is tested. Finally, 
the classifier is deployed and used to predict on a Raspberry Pi to prove that a low 
resource device can handle such a task. The results of the study show that this can be 
a suitable solution as the accuracy of the classifiers are extremely high on the test data. 
Artificial Neural Network and Support Vector Machines perform exceptionally well on 
the new attacks testing as well. Artificial Neural Network is proven to be the best 
overall as its benchmarks on the Raspberry Pi are way much better than any other 
algorithm.
