---
layout: default
title: Deep Neural Networks 2026/27
---

Links:
* [entry quiz questions](https://docs.google.com/document/d/1gQeiAG-1hZpSuZ0PaLAELCKuHNdGjYn3C2Qf1nHs2g8/edit?usp=sharing)
* [USOS](https://usosweb.mimuw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1000-317bDNN)
<!--* [Moodle](https://moodle.mimuw.edu.pl/course/view.php?id=2971) (for homework submissions)-->


# Lectures

Interactive slides (open in a browser; press `f` for fullscreen) and PDF versions for printing.

* Lecture 1 (Oct 7): [Neural networks: flexible functions, gradient descent, and the generalization question](slides/lectures/01-intro/) ([PDF](slides/pdf/01-intro.pdf))
* Lecture 2 (Oct 14): [PyTorch from scratch — automatic differentiation](slides/pdf/02-Pytorch_from_Scratch.pptx), [[utorch](https://github.com/adendek/utorch)] (Adam Dendek)
* Lecture 3 (Oct 21): optimizers, regularization
* Lecture 4 (Oct 28): log-loss, saturation, BatchNorm, skip connections
* Lecture 5 (Nov 4): flow-matching diffusion (Mateusz Wyszyński)
* Lecture 6 (Nov 13, **Friday**): convolutions, ResNet
* Lecture 7 (Nov 18): UNet, object detection
* Lecture 8 (Nov 25): RNN
* Lecture 9 (Dec 2): Attention, transformers
* Lecture 10 (Dec 9): Transformers continued
* Lecture 11 (Dec 16): Reinforcement Learning, DQN
* Lecture 12 (Dec 23): Policy Gradient
* Lecture 13 (Jan 6): Model-based RL, AlphaGo, ChatGPT


* [Recordings](https://drive.google.com/drive/folders/10w99lp-8kn8r7G58kPuz2t_Ip6KXAMrc).


# Labs

* Lab1: [EDA, linear regression, cross-validation](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/dnn01-EDA.ipynb)
* Lab2: [MSLE, chain rule](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/dnn02-MSLE.ipynb) ([slides](https://mim-ml-teaching.github.io/public-dnn/presentations/backprop-pres/backprop-pres.pdf) on the chain rule)
* Lab3: [Backpropagation](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/dnn03-backprop.ipynb) (see [leaflet](https://mim-ml-teaching.github.io/public-dnn/presentations/backprop-pres/backprop-leaflet.pdf), slides above, and [3blue1brown's video](https://www.youtube.com/watch?v=Ilg3gGewQ5U))
* Coding exercises:
  1. [Periodic boundary conditions](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/ex01-periodic-boundary.ipynb)


# ML Bootcamp

The "ML Bootcamp" consists optional extra sessions in the evenings, open to everyone:<br>
Mondays & Wednesdays, 18:15, for the first four weeks of the semester (you can choose any subset), rooms [3043](https://www.mimuw.edu.pl/pl/dojazd-i-plan/?room=3043)/[3044](https://www.mimuw.edu.pl/pl/dojazd-i-plan/?room=3044).<br>
(In USOS these dates had to be assigned somewhere, so they figure as part of groups 10 & 11, but they are not related in any way to the actual Tuesday 12:15 and 14:15 lab groups).

It is aimed to help you refresh or catch-up on foundations like NumPy, Pandas, gradients or linear regression, which are prerequisites for the DNN course.<br>
You can also just come and ask any questions, in particular about previous labs (of course we also encourage you to ask questions during labs and on Slack).


**Instructors:** Michał Krutul, Maciej Stefaniak

* Notebook 1: [NumPy & Pandas basics](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/bootcamp-ml/Lab_1%262_Numpy_Pandas_student_version.ipynb)

* Notebook 2: [Linear Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/bootcamp-ml/Lab_3_linear_regression_student_version.ipynb)

* Notebook 3: [Logistic Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/bootcamp-ml/Lab_4_logistic_regression_student_version.ipynb)

* Notebook 4: [Softmax Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/bootcamp-ml/Lab_4_softmax_regression_student_version.ipynb)

<!--* Lab5: [Experiment Tracking](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/bootcamp-ml/Lab_5_experiment_tracking_student_version.ipynb)-->


<br>


# Homeworks

Themes & deadlines:
* HW1: ConvNets and vision, deadline: Friday 27 XI.
* HW2: Transformers and Natural Language Processing, deadline: Friday 18 XII.
* HW3: basics of Reinforcement Learning, deadline: Friday 29 I (during the exam session).

Each homework will be announced around 2 weeks before the deadline.

<br>

---

<br>

# Rules
## Grading rules
* 46 points for labs, you need **≥ 30** points to pass:
  * 10 points for the entry quiz (Nov 3-6).
  * 3 × 12 points for homeworks.
* 54 points for the exam, you need ≥ 26 points to pass.
* up to 10 points for extra effort. These points are awarded by lab instructors for extra effort (e.g. solving optional tasks, presenting solved scenarios from previous classes, finding a significantly better solution), up to 3 points at a time.

Total: 110 points possible.
To pass, you need to pass both the homework threshold and the exam.
Grading scale:
* ≥ 90 points: 5.0
* ≥ 80 points: 4.5
* ≥ 70 points: 4.0
* ≥ 60 points: 3.5
* ≥ 50 points: 3.0

Lower thresholds for grades and for passing may be announced after grading is done.

## Entry quiz
The entry quiz (a random subset of 10 questions from the [public list](https://docs.google.com/document/d/1gQeiAG-1hZpSuZ0PaLAELCKuHNdGjYn3C2Qf1nHs2g8/edit?usp=sharing)) will be held at the beginning of the labs on November 3-6.
No notes or other materials allowed.
Please be in the labs on time.
If you're sick or very late, you can take the quiz on the next labs.
No retake allowed.

## Homework rules
There will be three homework assignments, in the form of larger lab-like scenarios to be solved individually at home.
There may be an uneven distribution of points for the three assignments.
You may be asked to explain your solution to homework graders.

### Late submissions
Each homework may be submitted within its announced deadline or late (until the start of the exam session, so by January 22nd, unless a given homework assignment's graders allow later submissions).
Late submissions have their score multiplied by 0.6.
You may submit both on-time and late, but only the last submission's grade will be used (even if it is worse after the 0.6 multiplier).

* If you are late by less than 10 minutes, your homework score will be unchanged.
* If you are late by less than 8 hours, your score will be multiplied by 0.9.
* Otherwise, your score will be multiplied by 0.6.

### Use of LLMs and AI code assistants
TBA...


### Exam pass
There will be a free exam pass (with the highest grade) given to some number of students with the top total scores for homeworks (*excluding* points for lab activity and the entry quiz).
This requires submitting the last homework earlier: details will be announced with HW3.
Last year the threshold was 49.0 / 50.0 points and only a few students got the pass.



<details markdown="1"><summary>Exam rules (click to expand)</summary>

## Exam rules
All the numbers may change slightly.

The exam will have two parts (both take place in the labs):
* Theoretical (18 pt, 30 min): quiz with nine multiple-choice and open questions (unrelated to the entry quiz). No materials allowed (empty paper for calculations is OK; no calculators).
* A short break (10-15 min).
* Practical (18+18 pt, 3 hours): two notebooks. Any communication, AI code assistants, or looking for solutions on the internet is NOT allowed.

<details markdown="1"><summary>Materials allowed (click to expand)</summary>
<ul>
<li>code and notes written earlier by yourself (you will need to upload it to Moodle before the exam; text, markdown, html, python code/notebooks are OK; formats like .docx or .odt are NOT supported)</li>
<li>internet for documentation and definitions only (all relevant packages like PyTorch).</li>
</ul>
</details>

Note that the exam reservation may be longer, just in case of delays.

<!--
<details><summary>Exam organization (click to expand)</summary>
<ul>
  <li>For non-urgent questions, use the Moodle forum.</li>
  <li>Please check that forum regularly, as clarifications will only be announced there (errors in tasks will be announced verbally in each room).</li>
  <li>You will work on lab computers with a restricted Linux (Ubuntu) environment and mostly no internet.</li>
  <li>During the practical part you will have access to VS Code (with Python, Jupyter, Pylance, Ruff extensions).</li>
  <li>You can also use Jupyter Lab for if you prefer (locally, not Colab, but essentially the same setup).</li>
  <li>Instead of the regular documentation for PyTorch and a few other packages, only <a href="https://devdocs.io">devdocs.io</a> will be available. Please familiarize yourself with it before the exam.</li>
  <li>You will get a paper ticket at the entrance to the labs, with a room number and a login/pass for the PC.</li>
  <li>You will need your central account credentials to log in to Moodle during the exam.</li>
</details>
-->



### Retake rules
The retake exam will be similar, except that:
* There might be one practical task instead of two.
* All the thresholds will be different (in particular the threshold required for getting a passing grade).
* It might not be possible to get a grade higher than 4 on the retake.
* You can only retake the exam as a whole (not just the theory or just the practical part).
* If you come for the retake exam, only the retake counts (not the best of both exams), even if you don't submit anything.


</details>
