## Cats and Dogs

In this experiment we'll train a machine to distinguish between cats and dogs. 
We'll be using Support vector machines for the same.
A support vector machine find the best possible seperating hyperplane between two seperable classes.
In this case our two classes are dogs and cats. There are subclasses too, but lets just look at the global picture

We start the experiment with a simple test. We'll take about only 2000 animals out of all for all our training purposes and about 500 for testing. As solving the SVM optimization problem is a bit computationally expensive we are bringing down the number of data points. Of course this will affect the results but we hope that the overall trends remain the same.

Also we won't be using libSVM to do our bidding, instead we'll use sklearn. It does contain all the general purpose kernels except the intersection kernel which isn't hard to implement on our own. 
But there is a catch here. The purpose of using intersection kernel is to speed up the process of computation by using a trick, this is not possible in my implementation :(. Please refer to these slides by Prof Jitendra Malik for more information how using intersection kernel speeds up computation.

The first thing we do is to normalize the feature vectors by their 2nd norm. Essentially project all the vectors onto a unit sphere. 

We then use an Intersection kernel with the same. 
Accuracy: 65.2%         Precision: 0.652    Recall: 0.691       F-Score: 0.693
Lets not hastily make a comment here.

The next thing we did was to try normalizing with sum
Accuracy: 56.3103%      Precision: 0.563    Recall: 1.0         F-Score: 0.720

Interestingly the results are bit lower than the 2nd norm normalization. 

Turns out 2nd norm is good for weighted and correlated measurements as is in this case. sum norm can be used in cases where we need to clean out gross errors in the data. (May Be!)

####
SVM solution also involves tuning a regularization parameter (C) to allow for a bit of flexibility in the sepearting hyper plane.

Before tuning C we've set the kernel to be an Intersection kernel.

Now lets vary C

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">C</th>
    <th class="tg-yw4l">Precision</th>
    <th class="tg-yw4l">Recall</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">0.1</td>
    <td class="tg-yw4l">0.659</td>
    <td class="tg-yw4l">0.689</td>
  </tr>
  <tr>
    <td class="tg-yw4l">10</td>
    <td class="tg-yw4l">0.654</td>
    <td class="tg-yw4l">0.707</td>
  </tr>
  <tr>
    <td class="tg-yw4l">20</td>
    <td class="tg-yw4l">0.658</td>
    <td class="tg-yw4l">0.708</td>
  </tr>
  <tr>
    <td class="tg-yw4l">50</td>
    <td class="tg-yw4l">0.654</td>
    <td class="tg-yw4l">0.71</td>
  </tr>
  <tr>
    <td class="tg-yw4l">100</td>
    <td class="tg-yw4l">0.654</td>
    <td class="tg-yw4l">0.709</td>
  </tr>
</table>

There doesn't seem to be any scope of improvement! Why would this be so?

This suggests us that while using the intersection kernel there aren't many points that need relaxing on the other side of the hyperplane. In a way there are few points that need to allowed to be on the other side of the classifying barrier.


Lets proceed further with our experiments.

While using a laplacian kernel we can vary the gamma of the laplacian! It decides how much smoothing it will cause over the data. 

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">G</th>
    <th class="tg-yw4l">Precision</th>
    <th class="tg-yw4l">Recall</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">0.0001</td>
    <td class="tg-yw4l">0.712</td>
    <td class="tg-yw4l">0.804</td>
  </tr>
  <tr>
    <td class="tg-yw4l">0.002</td>
    <td class="tg-yw4l">0.541</td>
    <td class="tg-yw4l">0.99</td>
  </tr>
  <tr>
    <td class="tg-yw4l">0.1</td>
    <td class="tg-yw4l">0.541</td>
    <td class="tg-yw4l">0.98</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1</td>
    <td class="tg-yw4l">0.541</td>
    <td class="tg-yw4l">0.99</td>
  </tr>
</table>

Just look at the recall rates. Almost everything is recalled. It seems that lowering gamma increases the accuracy. More experimentation was done by fine tuning gamma near 0.0001 but the data is currently lost due to a mistake.


Lets also look into other methods.
Random Forests:
Using 100 estimators.
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Class</th>
    <th class="tg-yw4l">precision</th>
    <th class="tg-yw4l">recall</th>
    <th class="tg-yw4l">f1-score</th>
    <th class="tg-yw4l">support</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">0</td>
    <td class="tg-yw4l">0.58</td>
    <td class="tg-yw4l">0.33</td>
    <td class="tg-yw4l">0.43</td>
    <td class="tg-yw4l">212</td>
  </tr>
  <tr>
    <td class="tg-yw4l">1</td>
    <td class="tg-yw4l">0.63</td>
    <td class="tg-yw4l">0.82</td>
    <td class="tg-yw4l">0.71</td>
    <td class="tg-yw4l">288</td>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">avg</td>
    <td class="tg-yw4l">0.61</td>
    <td class="tg-yw4l">0.62</td>
    <td class="tg-yw4l">0.59</td>
    <td class="tg-yw4l">500</td>
  </tr>
</table>



Random forest in general seem faster and better than using a SVM for this particular problem.
The below graph suggests that even Random Forests saturate at around 60% accuracy.



Now lets look into finer splitting of data... 



This is the classification report for leopard vs tiger.
Using a RandomForestClassifier

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Class</th>
    <th class="tg-yw4l">precision</th>
    <th class="tg-yw4l">recall</th>
    <th class="tg-yw4l">f1-score</th>
    <th class="tg-yw4l">support</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">leopard</td>
    <td class="tg-yw4l">0.77</td>
    <td class="tg-yw4l">0.74</td>
    <td class="tg-yw4l">0.75</td>
    <td class="tg-yw4l">156</td>
  </tr>
  <tr>
    <td class="tg-yw4l">tiger</td>
    <td class="tg-yw4l">0.71</td>
    <td class="tg-yw4l">0.74</td>
    <td class="tg-yw4l">0.72</td>
    <td class="tg-yw4l">133</td>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">avg</td>
    <td class="tg-yw4l">0.74</td>
    <td class="tg-yw4l">0.74</td>
    <td class="tg-yw4l">0.74</td>
    <td class="tg-yw4l">289</td>
  </tr>
</table>


This is the classification report for leopard vs tiger.
Using a LinearSVC

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Class</th>
    <th class="tg-yw4l">precision</th>
    <th class="tg-yw4l">recall</th>
    <th class="tg-yw4l">f1-score</th>
    <th class="tg-yw4l">support</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">leopard</td>
    <td class="tg-yw4l">0.76</td>
    <td class="tg-yw4l">0.81</td>
    <td class="tg-yw4l">0.78</td>
    <td class="tg-yw4l">141</td>
  </tr>
  <tr>
    <td class="tg-yw4l">tiger</td>
    <td class="tg-yw4l">0.81</td>
    <td class="tg-yw4l">0.76</td>
    <td class="tg-yw4l">0.78</td>
    <td class="tg-yw4l">148</td>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">avg</td>
    <td class="tg-yw4l">0.78</td>
    <td class="tg-yw4l">0.78</td>
    <td class="tg-yw4l">0.78</td>
    <td class="tg-yw4l">289</td>
  </tr>
</table>


So why not do a multiclass classification directly...

Here is the classification report for all classes using RandomForests using 500 estimators.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Class</th>
    <th class="tg-yw4l">precision</th>
    <th class="tg-yw4l">recall</th>
    <th class="tg-yw4l">f1-score</th>
    <th class="tg-yw4l">support</th>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">bobcat</td>
    <td class="tg-yw4l">0.18</td>
    <td class="tg-yw4l">0.06</td>
    <td class="tg-yw4l">0.09</td>
    <td class="tg-yw4l">158</td>
  </tr>
  <tr>
    <td class="tg-yw4l">chihuahua</td>
    <td class="tg-yw4l">0.27</td>
    <td class="tg-yw4l">0.06</td>
    <td class="tg-yw4l">0.1</td>
    <td class="tg-yw4l">169</td>
  </tr>
  <tr>
    <td class="tg-yw4l">collie</td>
    <td class="tg-yw4l">0.2</td>
    <td class="tg-yw4l">0.5</td>
    <td class="tg-yw4l">0.28</td>
    <td class="tg-yw4l">286</td>
  </tr>
  <tr>
    <td class="tg-yw4l">dalmatian</td>
    <td class="tg-yw4l">0.44</td>
    <td class="tg-yw4l">0.12</td>
    <td class="tg-yw4l">0.19</td>
    <td class="tg-yw4l">146</td>
  </tr>
  <tr>
    <td class="tg-yw4l">fox</td>
    <td class="tg-yw4l">0.17</td>
    <td class="tg-yw4l">0.02</td>
    <td class="tg-yw4l">0.03</td>
    <td class="tg-yw4l">109</td>
  </tr>
  <tr>
    <td class="tg-yw4l">germanshepherd</td>
    <td class="tg-yw4l">0.22</td>
    <td class="tg-yw4l">0.4</td>
    <td class="tg-yw4l">0.28</td>
    <td class="tg-yw4l">280</td>
  </tr>
  <tr>
    <td class="tg-yw4l">leopard</td>
    <td class="tg-yw4l">0.28</td>
    <td class="tg-yw4l">0.47</td>
    <td class="tg-yw4l">0.35</td>
    <td class="tg-yw4l">143</td>
  </tr>
  <tr>
    <td class="tg-yw4l">lion</td>
    <td class="tg-yw4l">0.57</td>
    <td class="tg-yw4l">0.03</td>
    <td class="tg-yw4l">0.06</td>
    <td class="tg-yw4l">138</td>
  </tr>
  <tr>
    <td class="tg-yw4l">persiancat</td>
    <td class="tg-yw4l">0.3</td>
    <td class="tg-yw4l">0.51</td>
    <td class="tg-yw4l">0.38</td>
    <td class="tg-yw4l">173</td>
  </tr>
  <tr>
    <td class="tg-yw4l">siamesecat</td>
    <td class="tg-yw4l">0.28</td>
    <td class="tg-yw4l">0.08</td>
    <td class="tg-yw4l">0.13</td>
    <td class="tg-yw4l">112</td>
  </tr>
  <tr>
    <td class="tg-yw4l">tiger</td>
    <td class="tg-yw4l">0.83</td>
    <td class="tg-yw4l">0.13</td>
    <td class="tg-yw4l">0.23</td>
    <td class="tg-yw4l">152</td>
  </tr>
  <tr>
    <td class="tg-yw4l">wolf</td>
    <td class="tg-yw4l">0.25</td>
    <td class="tg-yw4l">0.01</td>
    <td class="tg-yw4l">0.02</td>
    <td class="tg-yw4l">124</td>
  </tr>
  <tr>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
    <td class="tg-yw4l"></td>
  </tr>
  <tr>
    <td class="tg-yw4l">avg</td>
    <td class="tg-yw4l">0.32</td>
    <td class="tg-yw4l">0.24</td>
    <td class="tg-yw4l">0.2</td>
    <td class="tg-yw4l">1990</td>
  </tr>
</table>

Seems pretty rugged, may be increasing the number of trees will boost up the precision, but is really heavy on the machine.