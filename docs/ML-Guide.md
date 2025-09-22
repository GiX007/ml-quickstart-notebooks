# Framework for Approaching Machine Learning Projects

Machine learning is a big field. In the news, it often sounds like magic.  
This guide gives a clear picture of two things:  
1. The main kinds of problems machine learning can be used for.  
2. A simple way to start small machine learning projects.  

Many people also wonder about the difference between machine learning, artificial intelligence (AI), and data science.  
The truth is, there are no exact answers. Even people who work in the field for years often find it hard to explain. So if someone says they have the perfect definition, it’s good to be careful.  

For this guide, machine learning is kept simple:  

> *It means using data to find patterns, understand things better, or make guesses about the future.*  

The steps that follow are practical. They focus on building something, testing it, and learning from the results. The best way to understand machine learning is by **doing it**.

## 6 Steps for Your Next Machine Learning Project

A machine learning workflow can be thought of in three big parts: collecting data, building models, and deploying them. Each part affects the others.  

Often, a project starts with gathering data. Then the data is used to train a model. Sometimes the data isn’t good enough, so it’s back to collecting more or better data. The model gets built again, tested, and maybe deployed. If it doesn’t work well, the cycle repeats.  

How data is collected depends on the problem. For example, it could be customer purchases in a spreadsheet, sensor readings, or images.  

Modeling means using a machine learning method to find patterns and insights in that data.  

So, what’s the difference between a normal algorithm and a machine learning one?  
Think of a cooking recipe: a regular algorithm is like step-by-step instructions for turning raw ingredients into a finished dish. In machine learning, you start with the ingredients (the data) and the final dish (the answer you want). The algorithm figures out the steps needed to get from one to the other.  

There are many types of machine learning algorithms. Some perform better on certain problems than others, but the goal is always the same: to find patterns or rules hidden in data.  

Deployment means taking the trained model and using it in practice. This could mean an app that suggests products, or a system in a hospital that helps doctors predict disease.  

The details of each step change depending on the project, but the main principles stay the same.  

This guide focuses on the **modeling** part. It assumes that data has already been collected, and looks at how to build a small proof-of-concept model. The process can be broken into six steps:

!

1. **Problem definition** — What issue are we trying to solve? How can it be phrased as a machine learning task?  
2. **Data** — What data do we have? Is it structured (tables) or unstructured (text, images)? Static or streaming? How does it match the problem we want to solve?  
3. **Evaluation** — How do we measure success? For example, is a model that’s 95% accurate “good enough”?  
4. **Features** — Which parts of the data will we use? What can we add to help the model learn better?  
5. **Modeling** — Which model should be used? How can it be trained, tested, and compared to others?  
6. **Experimentation** — What else can we try? Does the model behave as expected in practice? How do the results change if we tweak the process?  

Let’s take a closer look at each step.
