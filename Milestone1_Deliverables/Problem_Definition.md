## Problem Definition

Jeffrey and I are going to undertake entering a kaggle machine-learning competition for our course project. The competition is the MABe challenge -Social Action Recognition in Mice. An overview of the competition can be found [here](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview).   
   
**Context and Overview:**   
    The problem is important because manually reviewing footage to describe animal behaviours is an extremely time intensive, and boring task. It requires researchers to watch footage for hours and painstakingly annotate every action and behaviour they observe and then take careful, precise, notes detailing the video, the animal subjects, the action or behaviour, etc. It would be very beneficial if this task could be automated using machine learning to train a model to accurately describe these actions and behaviours, and even predict new ones. A well-trained model could observe and predict patterns, behaviours, and interactions that a human observer may miss. It is a very interesting problem because animal behaviour has often been somewhat foreign to humans, always striving to understand their languages, their behaviours, their interactions. But it has always eluded us why animals behave how they do in most common-place situations. Machine learning could fundamentally change our understanding of the animal kingdom.     
           
**Specific Objective:**        
    We will be trying to classify animal actions. The competition provides a data set of coordinates of mouse body-parts during pre-annotated interactions and actions, from which the model (ideally) can distinguish patterns of movement to ascribe to the different actions. This wil be supervised learning as the data set already has labelled actions with data describing the mouse movements during those actions and trying to classify un-labelled movements in the test data.   
                   
**Scope and Feasability:**     
    The data set we will use has been provided by the kaggle competition. It is suitable to use because it is suitably large without being unnessecarilly so for this task. It contains information that should be suitable to use to train a model to classify interactions between mice based on the labelled sets of training data provided in the kaggle data set.        
            
**Expected Outcome:**
    We expect the ML solution will provide insight into new animal behaviours and interactions that human observers may have missed. It should also allow for some degree of automation when labelling and annotating future interactions and hopefully provide a foundation for expanding into other members of the animals kingdom to make meaningful observations about the interactions of other species as well.         
