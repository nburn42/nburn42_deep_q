This is a Deep Q Network running tensorflow.   
It is aimed at running open ai gym.

Network Description:   
2 relu hidden layers with 100 neurons each (overkill)   
Adam optimizer with .01 learning rate   
.95 Discount rate for Q Learning   
a random chance of taking an exploratory action   
chance decays over time   

Note:   
I have not yet optimized those parameters   

Code availible here:   
https://github.com/nburn42/nburn42_deep_q   

reproducing should be as easy as running:
python deep_q.py
(assuming gym, box2d, tensorflow, and numpy are installed and you are using python2)

Nathan Burnham 
nburn42@gmail.com   
