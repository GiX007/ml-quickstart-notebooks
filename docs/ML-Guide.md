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

![](https://github.com/GiX007/ml-quickstart-notebooks/blob/main/data/images/ml_steps.png)

1. [Problem definition](#1-problem-definition--turn-your-business-problem-into-a-machine-learning-problem)
 — What issue are we trying to solve? How can it be phrased as a machine learning task?  
2. **Data** — What data do we have? Is it structured (tables) or unstructured (text, images)? Static or streaming? How does it match the problem we want to solve?  
3. **Evaluation** — How do we measure success? For example, is a model that’s 95% accurate “good enough”?  
4. **Features** — Which parts of the data will we use? What can we add to help the model learn better?  
5. **Modeling** — Which model should be used? How can it be trained, tested, and compared to others?  
6. **Experimentation** — What else can we try? Does the model behave as expected in practice? How do the results change if we tweak the process?  

Let’s take a closer look at each step.

## 1. Problem Definition — Turn Your Business Problem Into a Machine Learning Problem

The first step is to connect your business problem with a machine learning task. This means restating the problem in a way that can be solved with data and algorithms.  

There are four main types of machine learning: supervised learning, unsupervised learning, transfer learning, and reinforcement learning (there’s also semi-supervised, but we’ll skip that here). In business, the most common are supervised learning, unsupervised learning, and transfer learning.  

---

### Supervised Learning

Supervised learning is called “supervised” because you already have both data and labels (answers). The algorithm learns patterns in the data that connect inputs to outputs.  

For example, imagine predicting heart disease in patients. You may have medical records for 100 patients (inputs) and whether or not each had heart disease (labels).  
The algorithm studies this data to learn patterns that link medical history to the presence of disease.  

Once trained, the model can look at a new patient’s records and predict whether they are likely to have heart disease. The prediction is never 100% certain — it’s usually a probability, such as “70% chance of heart disease.”  

### Unsupervised Learning

Unsupervised learning is when you have data but no labels. For example, an online game store may have customer purchase histories but no categories.  
Here, the algorithm groups customers based on their behavior. This is called clustering.  

You may discover groups like:  
- customers who buy PC games  
- customers who prefer console games  
- customers who mostly buy discounted older games  

The algorithm didn’t provide labels, it only found patterns. Using your domain knowledge, you decide how to interpret these groups.  

### Transfer Learning

Transfer learning is using knowledge from one task and applying it to another. Instead of training a model from scratch, you start with one that’s already been trained and adapt it to your problem.  

This saves time and resources. For example, suppose a car insurance company wants to classify whether an insurance claim is “at fault” or “not at fault.”  
Instead of building a model from zero, you could start with an existing text model (trained on large amounts of text, like Wikipedia). Then, fine-tune it using your company’s insurance claims and their outcomes.  

---

### Main Types of Business ML Problems

Most machine learning problems in business fall into these three categories:  

- **Classification** — Decide if something belongs to one group or another. Example: disease or no disease. (Can also be multi-class or multi-label.)  
- **Regression** — Predict a number. Example: house prices, sales forecasts, or number of visitors next month.  
- **Recommendation** — Suggest items. Example: products to buy, or articles to read based on past behavior.  

---

### Example: Car Insurance

Let’s go back to the car insurance case.  
The company receives thousands of claims per day. Staff must check each one and decide if the claim is “at fault” or “not at fault.” With so many claims, staff can’t keep up.  

This is where machine learning comes in. The task can be rephrased as:  

> *We’re a car insurance company that wants to classify claims as “at fault” or “not at fault.”*  

Notice the keyword: **classify**. This makes it a classification problem.  

It’s important to note that success isn’t guaranteed. Not every problem can be solved with machine learning. But the first step is always to restate the problem in simple, ML-friendly terms. Add details only when needed. 

