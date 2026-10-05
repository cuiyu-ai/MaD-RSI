<div align="center">

# MaD-RSI: Model-as-Data for Recursive Self-Improvement of Agent Harnesses

**Yu Cui<sup>1</sup> · Hong He<sup>2</sup> · Ruiqing Yue<sup>3,4</sup> · Sicheng Pan<sup>1</sup>**  
**Zhuoyu Sun<sup>1</sup> · Haibin Zhang<sup>5,6</sup> · Cong Zuo<sup>1</sup>**

<sup>1</sup> Beijing Institute of Technology  
<sup>2</sup> Tsinghua University  
<sup>3</sup> Chengdu Institute of Computer Applications, Chinese Academy of Sciences  
<sup>4</sup> University of Chinese Academy of Sciences  
<sup>5</sup> Yangtze Delta Region Institute of Tsinghua University, Zhejiang  
<sup>6</sup> Jiaxing Key Laboratory of Artificial Intelligence and Cyber Resilience

[**Repository**](https://github.com/cuiyu-ai/MaD-RSI) · [**Contact**](mailto:cuiyu@bit.edu.cn)

**Project Lead:** Yu Cui · cuiyu@bit.edu.cn

</div>

> **Treat execution models as sources of behavioral evidence for improving a shared harness.**

## Abstract

**Recursive self-improvement (RSI) of agent harnesses** seeks to improve LLM-agent behavior by iteratively revising prompts, control logic, and tool-use policies while keeping model parameters fixed. Existing approaches typically optimize a harness using feedback from a single LLM and evaluate transfer to other models only after optimization. This model-centric formulation can bias harness updates toward the capabilities and failure modes of the optimization model, limiting their effectiveness across heterogeneous LLMs.

We introduce **MaD-RSI**, a **Model-as-Data** framework that places the shared harness at the center of RSI and incorporates cross-model generalization directly into the improvement process. Rather than treating a single LLM as the fixed optimization target, MaD-RSI uses heterogeneous LLMs as behavioral probes that provide complementary evidence about the same harness. It organizes model responses as a structured set of observations spanning queries, models, and repeated stochastic samples, enabling both cross-model comparison and estimation of within-model variability. MaD-RSI then prioritizes informative queries using cross-model contrasts together with repeated observations, and aggregates trajectories for the same query into comparative feedback for recursive harness refinement. In MaD-RSI, heterogeneous LLMs contribute complementary behavioral evidence to the improvement of a shared harness, and the resulting harness subsequently benefits multiple execution models, including models not observed during optimization.

MaD-RSI is compatible with both general-purpose and safety-oriented harness optimizers. We evaluate it across legal reasoning, code generation, interactive household tasks, and agent safety, measuring task performance, transfer to unseen LLMs, and optimization cost. Our results show that incorporating heterogeneous LLMs and repeated sampling into the RSI process improves cross-model transfer over single-model harness optimization.


<img width="1168" height="627" alt="overview" src="https://github.com/user-attachments/assets/ffa36173-d0db-4884-b207-73a9764ef4ac" />


## Why Model-as-Data?

A model is usually treated as the execution target whose harness should be improved. MaD-RSI additionally treats each model as a **behavioral probe**: its successes, failures, and intermediate actions reveal different aspects of the same task and harness.

Here, *model-as-data* refers to **model-indexed behavioral observations**. It does not mean accessing model parameters, fine-tuning execution models, or replacing the task dataset. Queries provide shared experimental conditions; models provide diverse observations under those conditions.

| Aspect | Single-model improvement setting | MaD-RSI |
|---|---|---|
| Training feedback | Trajectories from one execution model | Structured observations across an execution-model group |
| Evidence organization | Individual runs and their outcomes | Query × model × sampling configuration × repetition |
| Query prioritization | Failures or scores observed on the training model | Cross-model contrasts, repeated outcomes, and evidence quality |
| Feedback for a query | Model-specific success or failure traces | Consolidated same-query evidence across models |
| Improvement target | A harness optimized using one model's feedback | A shared harness optimized using group-level evidence |
| Transfer evaluation | Apply the resulting harness to other models | Evaluate both participating models and held-out models |

The comparison describes the single-model setting used by our baselines, rather than a claim that every existing RSI method uses the same design.


## Cross-Model Evaluation Metrics

We introduce two complementary metrics for evaluating a shared harness across a group of **m execution models**: **pass@m** and **pass^m**.

**pass@m** measures the collective task coverage of the model group. It does not imply that a deployed system can identify the successful model output without an additional selection mechanism.

**pass^m** measures the fraction of queries solved consistently across the entire model group, providing a stricter measure of shared-harness reliability.

| Metric | Success criterion | Interpretation |
|---|---|---|
| `pass@m` | At least one model succeeds | Collective task coverage |
| `pass^m` | Every model succeeds | Consistent success across models |

Unlike conventional pass@k, where \(k\) denotes samples from a model, \(m\) here denotes **distinct execution models**. Both metrics should be reported alongside per-model performance and the exact model-group composition.


## Citation

If you use MaD-RSI in your research, please cite this repository:

```bibtex
@misc{cui2026madrsi,
  author       = {Cui, Yu and He, Hong and Yue, Ruiqing and
                  Pan, Sicheng and Sun, Zhuoyu and
                  Zhang, Haibin and Zuo, Cong},
  title        = {MaD-RSI: Model-as-Data for Recursive Self-Improvement of Agent Harnesses},
  year         = {2026},
  howpublished = {\url{https://github.com/cuiyu-ai/MaD-RSI}},
  note         = {Code repository}
}
```

