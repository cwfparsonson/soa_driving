import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Initial Research and Identification of Objectives", Start='2020-09-01', Finish='2020-10-15', Status="Complete"),
    dict(Task="Proposal Writing and Submission", Start='2020-10-15', Finish='2020-10-21', Status="Complete"),
    dict(Task="Run simulations and study code", Start='2020-10-21', Finish='2020-10-28', Status="Complete"),
    dict(Task="Progess Report #1", Start='2020-10-28', Finish='2020-11-04', Status="Complete"),
    dict(Task="Experiment with changing dimensions", Start='2020-11-04', Finish='2020-11-14', Status="Complete"),
    dict(Task="Progress Report #2", Start='2020-11-20', Finish='2020-11-25', Status="Complete"),
    dict(Task="Perfomance Graphs and analysis", Start='2020-12-25', Finish='2021-01-05', Status="In Progress"),
    dict(Task="Interim Report", Start='2020-11-25', Finish='2020-12-23', Status="Complete"),
    dict(Task="Experimentation with cascaded SOAs", Start='2020-12-23', Finish='2021-02-15', Status="In Progress"),
    dict(Task="Gain Profile Measurements", Start='2020-12-23', Finish='2021-02-15', Status="In Progress"),
    dict(Task="Transfer function manipulation", Start='2020-12-30', Finish='2021-01-30', Status="Incomplete"),
    dict(Task="Progress Report #3", Start='2021-01-13', Finish='2021-02-15', Status="Incomplete"),
    dict(Task="Impact statement", Start='2021-02-17', Finish='2021-02-24', Status="Incomplete"),
    dict(Task="Implementation of FPGA as a DAC", Start='2021-02-24', Finish='2021-03-01', Status="Incomplete"),
    dict(Task="Presentation Preparation and improvements", Start='2021-03-01', Finish='2021-03-14', Status="Incomplete"),
    dict(Task="Final Report", Start='2021-03-10', Finish='2021-03-31', Status="Incomplete")
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Status")
fig.update_yaxes(autorange="reversed")
fig.show()