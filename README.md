* Dataset

1. Yahoo Music Dataset: Please visit the following link to request and download the Yahoo music dataset (R2 - Yahoo! Music User Ratings of Songs with Artist, Album, and Genre Meta Information). 
https://webscope.sandbox.yahoo.com/catalog.php?datatype=r
The training Yahoo dataset used in our experiments, even in compressed format, is more than 1GB which violates the Github upload limit restrictions. Please contact the author (firstname AT cs.ucla.edu) for one-time private Dropbox link.

2. Amazon Reviews Dataset: This dataset is managed by Julian McAuley (julian.mcauley@gmail.com). Please visit this link to request for the data. http://jmcauley.ucsd.edu/data/amazon/
For reproducability purposes, our processed training, validation and test datasets used in our experiments have been uploaded under the folder 'data'.



* Code

1. Spark ALS code is part of the open source Spark MLlib package. Please visit https://github.com/apache/spark/tree/master/mllib/src/main/scala/org/apache/spark/mllib/recommendation to check the code, documentation and revision history.

2. Petuum (Strads) setup completed following the documentation at http://pmls.readthedocs.io/en/latest/installation.html

3. Instructions on how to run matrix factorization (MF) using Strads are given at http://pmls.readthedocs.io/en/latest/matrix-fact.html

4. Spark experimentation script, rmse evaluation script, baseline predictor computation script are provided under the directory 'scripts'.
