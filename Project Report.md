# Abstract

The problem that I am tackling is identifying whales and dolphins in the ocean using a deep
neural network. The kaggle competition gave thousands of images of fins of different whales and
dolphins and a key for the type of species it is. Reading through manually would take a while so
I usedhttps://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5to get the data I
would use to generate the model. I then used this data to make an elegy model and ran the data
through the model with 100 epochs each with 200 steps. By doing this I found that the accuracy
of the model increased and the ability to predict the species of marine life increased.

# Introduction

The problem I am working on is identifying different species of whales and dolphins based off
their backfin. This problem is important because it allows researchers to properly estimate where
these species are and how many of them are in different areas. This helps excerpts in these fields
properly see if the species are doing well, and if they are becoming endangered, help stop it from
getting worse. This can also be used to tell where to release certain species that were being
nursed back to health, or where held captive before. This problem has typically been solved by
using the human eye. The problem with this technique is that it takes a long time and is subject to
human error. Also since this technique uses a lot of people, they might have different opinions on
the species it is and different perceptions. Through my method, I was able to predict with around
95% for training and validation accuracy. This might be less accurate than predicting by eye, but
it is more consistent and takes much less time and manpower.


# Related Work

The related work I used was the backfin notebook listed in the introduction and looking at the
solutions that the top results for the competition had. I also looked at examples of using elegy to
make deep neural networks in order to more effectively utilize it in my project. The backfin
notebook was able to more effectively process the data and make it more manageable. It utilizes
another tool called Yolov5 to detect the objects and properly identify the fins faster. The
difference between my approach and the approach of the top results on the happy whale
competition is that my model utilizes less data, is less accurate, and overall much simpler than
those used by the people who did well. This was due to the nature of my assignment and the fact
that I am much less experienced than those competing, and have virtually no experience with
machine learning or jax.

# Data

The data I am using in this project is from the kaggle competition itself. The data provided are
images of certain fins of different whales and dolphins. With this data is a csv file that contains
the information on the species of these particular species. There is also a folder of images to test
the model generated on. These images themselves were difficult for me to work with as there


were so many, around 50,000, so I used a public notebook called backfin detection with yolov
to get data. This notebook uses a bounding box to isolate the fin of the whale or dolphin and
make the image smaller. This notebook also only looks at a particular whale, whale flute, to
make it much easier and faster to train the model with about 1,2000 samples instead of 50,000.
Examples of the bounding box are shown below.

This method was also used to label all the data using the csv instead of having to constantly
check the csv. This made using the data a lot easier, and more similar to the cats and dogs
example we did in the homework.

# Methods

My approach for solving the process was a combination of previously established methods and
using elegy to utilize jax. I initially tried to just unzip the train file and use all the files in there,
but since the file was very large, about 50 GB, it required too much RAM. Because of this
I was having trouble coming up with a solution, but eventually found that some other solutions
utilized the backfin notebook. This notebook helped me identify the data mre properly, as it
assigned weights to each image and also identified them based on weights and heights. The
graphs of this are shown below.



These graphs portray the x_center vs. y_center, width vs. height, and area respectively. This data
helped classify the data and group it. My approach was similar to the cats vs. dogs approach, but
with more data classified in a different way. I used elegy to utilize jax to train the data using the
epoch method. By doing this I was able to effectively train the model.
**Experiment**
I had a lot of problems with the data as the original backfin notebook took me 9 hours to run, so I
don’t know how accurate my data was. I was able to obtain a graph that displayed that the
accuracy of the model increased over time to north of 90%, shown below.

The problem is that this model doesn’t always work as the amount of time the notebook takes is
long and is prone to crashing. I didn’t know how to implement a more complex and effective
way to test the data and make the model, due to my inexperience and lack of examples with jax.
The example notebook helped show graphs that seemed to show more accurate and meaningful
results shown below.


These help model the confidence and precision of the model. Overall my model seems worse
than the previously published work, but it was the best I could do.

# Conclusion

My key results were that this model was much more complex than our previous example ones
and took much more time to process and various other tools to more accurately utilize the data in
a more timely manner than just processing the whole data set. I also found it annoying that the
images had seemingly random names as compared to the previous data we saw, which labeled it
in numerical order. I also had to label the data utilizing the guiding csv because of this. Future
ideas I would have is to familiarize myself with jax more beforehand, because I didn’t know how
to properly utilize non-jax code in my project. I would also spend more time on this in the future,


as my results were rushed due to the notebooks used taking a considerable amount of time to run,
almost a full day of time each time ran.



