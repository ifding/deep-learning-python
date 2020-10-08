
## Components of Autonomous Driving System


![Alt](images/overview.png "Standard components in a modern autonomous driving systems pipeline.")

The Autonomous Driving survey paper (https://arxiv.org/pdf/2002.00444.pdf) demonstrates the above pipeline from sensor stream to control actuation.

The **sensor architecture** includes multiple sets of cameras, radars and LIDARs as well as a GPS-GNSS system for absolute localization and Inertial Measurement Units (IMUs) that provide 3D pose of the vehicle in space.

The goal of the **perception module** is the creation of an intermediate level representation of the environment state that is be later utilized by a decision making system that produces the driving policy.

This state would include lane position, drivable zone, location of agents such as cars, pedestrians, state of traffic lights and others.

Several perception tasks like _semantic segmentation_, _motion estimation_, _depth estimation_, _soiling detection_, etc which can be unified into a multi-task model.


## Courses
* [[Coursera] Machine Learning](https://www.coursera.org/learn/machine-learning) - presented by [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng), as of 2020 Jan 28 it has 125,344 ratings and 30,705 reviews.
* [[Coursera+DeepLearning.ai]Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - presented by [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng), 5 Courses, teaches foundations of deep learning, programming language: python
* [[Udacity] Self-Driving Car Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) - teaches the skills and techniques used by self-driving car teams. Program syllabus can be found [here](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08#.bfgw9uxd9).
* [[University of Toronto] CSC2541
Visual Perception for Autonomous Driving](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/CSC2541_Winter16.html) - A graduate course in visual perception for autonomous driving. The class briefly covers topics in localization, ego-motion estimaton, free-space estimation, visual recognition (classification, detection, segmentation).
* [[INRIA] Mobile Robots and Autonomous Vehicles](https://www.fun-mooc.fr/courses/inria/41005S02/session02/about?utm_source=mooc-list) - Introduces the key concepts required to program mobile robots and autonomous vehicles. The course presents both formal and algorithmic tools, and for its last week's topics (behavior modeling and learning), it will also provide realistic examples and programming exercises in Python.
* [[Universty of Glasgow] ENG5017 Autonomous Vehicle Guidance Systems](http://www.gla.ac.uk/coursecatalogue/course/?code=ENG5017) - Introduces the concepts behind autonomous vehicle guidance and coordination and enables students to design and implement guidance strategies for vehicles incorporating planning, optimising and reacting elements.
* [[David Silver - Udacity] How to Land An Autonomous Vehicle Job: Coursework](https://medium.com/self-driving-cars/how-to-land-an-autonomous-vehicle-job-coursework-e7acc2bfe740#.j5b2kwbso) David Silver, from Udacity, reviews his coursework for landing a job in self-driving cars coming from a Software Engineering background.
* [[Stanford] - CS221 Artificial Intelligence: Principles and Techniques](http://stanford.edu/~cpiech/cs221/index.html) - Contains a simple self-driving project and simulator.
* [[MIT] 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/) - *"This class is an introduction to the practice of deep learning through the applied theme of building a self-driving car. It is open to beginners and is designed for those who are new to machine learning, but it can also benefit advanced researchers in the field looking for a practical overview of deep learning methods and their application. (...)"* 
* [[MIT] Deep Learning](https://deeplearning.mit.edu/) - *"This page is a collection of MIT courses and lectures on deep learning, deep reinforcement learning, autonomous vehicles, and artificial intelligence organized by Lex Fridman."* 
* [[MIT] Human-Centered Artificial Intelligence](https://hcai.mit.edu/) - *"Human-Centered AI at MIT is a collection of research and courses focused on the design, development, and deployment of artificial intelligence systems that learn from and collaborate with humans in a deep, meaningful way."*
* [[UCSD] - MAE/ECE148 Introduction to Autonomous Vehicles](https://guitar.ucsd.edu/maeece148/index.php/Introduction_to_Autonomous_Vehicles) - A hands-on, project-based course using DonkeyCar with lane-tracking functionality and various advanced topics such as object detection, navigation, etc.
* [[MIT] 2.166 Duckietown](http://duckietown.mit.edu/index.html) - Class about the science of autonomy at the graduate level. This is a hands-on, project-focused course focusing on self-driving vehicles and high-level autonomy. The problem: **Design the Autonomous Robo-Taxis System for the City of Duckietown.**
* [[Coursera] Self-Driving Cars](https://www.coursera.org/specializations/self-driving-cars#about) - A 4 course specialization about Self-Driving Cars by the University of Toronto. Covering all the way from the Introduction, State Estimation & Localization, Visual Perception, Motion Planning.