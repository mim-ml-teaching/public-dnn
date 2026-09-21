---
layout: default
title: Deep Neural Networks 2026/27
---


Links: [lecture slides & recordings](https://drive.google.com/drive/folders/10w99lp-8kn8r7G58kPuz2t_Ip6KXAMrc),
[USOS](https://usosweb.mimuw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazPrzedmiot&kod=1000-317bDNN)

# ML Bootcamp

{% comment %}
<!--
*(open to all students interested in practical ML foundations and with any questions regarding DNN laboratories)*

The bootcamp will take place during the **first 4 weeks of the semester**, on **Mondays and Wednesdays at 18:15 (6:15 PM)** in MiMUW [3043](https://www.mimuw.edu.pl/pl/dojazd-i-plan/?room=3043)/[44](https://www.mimuw.edu.pl/pl/dojazd-i-plan/?room=3044)/[45](https://www.mimuw.edu.pl/pl/dojazd-i-plan/?room=3045) labs.<br>
**Dates:** 6.10, 8.10, 13.10, 15.10, 20.10, 22.10, 27.10, 29.10<br>
**Instructors:** Michał Krutul, Jan Małaśnicki, Maciej Stefaniak

* Lab1&2: [NumPy & Pandas basics](https://colab.research.google.com/github/mim-ml-teaching/public-dnn-2025-26/blob/main/docs/Lab_1%262_Numpy_Pandas_student_version.ipynb)

* Lab3: [Linear Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn-2025-26/blob/main/docs/Lab_3_linear_regression_student_version.ipynb)

* Lab4: [Logistic Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn-2025-26/blob/main/docs/Lab_4_logistic_regression_student_version.ipynb)

* Lab4(extra): [Softmax Regression](https://colab.research.google.com/github/mim-ml-teaching/public-dnn-2025-26/blob/main/docs/Lab_4_softmax_regression_student_version.ipynb)

* Lab5: [Experiment Tracking](https://colab.research.google.com/github/mim-ml-teaching/public-dnn-2025-26/blob/main/docs/Lab_5_experiment_tracking_student_version.ipynb)
-->
{% endcomment %}

# Labs

* Lab1: [linear regression with MSLE, cross-validation](https://colab.research.google.com/github/mim-ml-teaching/public-dnn/blob/main/dnn01-MSLE.ipynb)


# Homeworks

Themes & deadlines:
* HW1: MLPs and backprop, deadline in November.
* HW2: ConvNets and vision, deadline in December.
* HW3: Transformers and Natural Language Processing, deadline in early January.
* HW4: basics of Reinforcement Learning, deadline in late January (during the exam session).

<br>
<br>

---

<br>

# Rules
## Grading rules
* 50 points for homeworks, you need **≥ 35** points to pass.
* 50 points for the exam, you need ≥ 25 points to pass.
* up to 10 points for activity. Activity points are awarded by lab instructors for extra effort (e.g. solving optional tasks, presenting solved scenarios from previous classes, finding a significantly better solution), up to 2.5 points at a time.

Total: 110 points possible. To pass, you need to pass both the homeworks threshold and the exam.
Grading scale:
* ≥ 90 points: 5.0
* ≥ 80 points: 4.5
* ≥ 70 points: 4.0
* ≥ 60 points: 3.5
* ≥ 50 points: 3.0

Lower thresholds for grades and for passing may be announced after grading is done.



## Homework rules
There will be four homework assignments, in the form of larger lab-like scenarios to be solved individually at home.
There may be an uneven distribution of points for the four assignments.
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

<details markdown="1"><summary>Exam rules (click to expand)</summary>

## Exam rules
All the numbers may change slightly.

The exam will have two parts (both take place in the labs):
* Theoretical (16 pt, 30 min): quiz with eight multiple-choice and open questions. No materials allowed (empty paper for calculations is OK; no calculators).
* A short break (10-15 min).
* Practical (17+17 pt, 3 hours): two notebooks. Any communication, AI code assistants, or looking for solutions on the internet is NOT allowed.

<details markdown="1"><summary>Materials allowed (click to expand)</summary>
<ul>
<li>code and notes written earlier by yourself (you will need to upload it to Moodle before the exam; text, markdown, html, python code/notebooks are OK; formats like .docx or .odt are NOT supported)</li>
<li>internet for documentation and definitions only (all relevant packages like PyTorch; read-only Wikipedia).</li>
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


### Exam pass
There will be a free exam pass (with the highest grade) given to some number of students with the top total scores for homeworks (*excluding* points for lab activity).
This requires submitting the last homework earlier: details will be announced with HW4.
Last year the threshold was 49.0 / 50.0 points and only a few students got the pass.

### Retake rules
The retake exam will be similar, except that:
* There might be one practical task instead of two.
* All the thresholds will be different (in particular the threshold required for getting a passing grade).
* It might not be possible to get a grade higher than 4 on the retake.
* You can only retake the exam as a whole (not just the theory or just the practical part).
* If you come for the retake exam, only the retake counts (not the best of both exams), even if you don't submit anything.


</details>
