## Dataset Source

The dataset we will use is freely available on Kaggle through the competition. It can be found through [this link](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data).     
It can be accessed directly using a Kaggle notebook and importing the data into the notebook. It can also be downloaded locally for uses such as doing EDA on your local machine.     
The data set is essentially 2 directories containing parquet files; one directory's parquet files contains the "features" (tracking) and one directory's parquet files contain the labels (annotations). Reading all the parquet files into dataframes and loading them all into memory uses about 14GB of memory from Kaggle's cloud computing, using a Kaggle notebook.      
The annotation parquet files all have many many rows but the same number of columns. The columns are agent_id (mouse performing an action), target_id (mouse they perform it on/with), action (the action performed), start_frame (when the action was started/initiated), and end_frame (when the action ended).     
The tracking parquet file columns are video_frame (when the data was recorded), mouse_id (the mouse the data is associated with), bodypart (the bodypart beinhg described by the x and y coordinates), x, and y (the x and y coordinates of the body part being described).       
There are also 2 CSV files with different metadata about the mice, labs, and videos, etc. One for the training sets, and an example one for how the test sets will be structured.         
The key features of the data are the video, the labelled actions, the mice IDs, and the mouse body parts with x and y coordinates per frame (can be restructured/visualized to show a projection in 2D of the mouse movements.)
