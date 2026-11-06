# Papers for the course and the project

Scope: small-sample, model-based experimental design (Bayesian / information-theoretic / adaptive
design, active learning with statistical models, Gaussian processes). No deep-learning methods that
need large data pools. Every project paper can be turned into a **sampler** for the 2AFC task in the
Tutorial 06 form, `sampler(gp, n, rng)` or `sampler(gp, data, n, rng)` (the GP may be replaced inside
the sampler by the paper's own model, fitted to `data`).

Bibliographic details were checked online on 2026-11-06 (✓) or are long-established classics (●).

## Foundations (course background)

| | Reference | Why |
|---|---|---|
| ✓ | Musslick, S., Bartlett, L. K., Chandramouli, S. H., Dubova, M., Gobet, F., Griffiths, T. L., Hullman, J., King, R. D., Kutz, J. N., Lucas, C. G., Mahesh, S., Pestilli, F., Sloman, S. J., & Holmes, W. R. (2025). Automating the practice of science: Opportunities, challenges, and implications. *PNAS, 122*(5), e2401238121. https://doi.org/10.1073/pnas.2401238121 | Why automate science; where the bottlenecks are |
| ✓ | Musslick, S., et al. (2024). AutoRA: Automated Research Assistant for closed-loop empirical research. *Journal of Open Source Software, 9*(104), 6839. https://doi.org/10.21105/joss.06839 | The framework behind the task's loop |
| ✓ | Dubova, M., Moskvichev, A., & Zollman, K. (2026). Against theory-motivated experimentation: Can random experimental choice lead to better theories? *Collective Intelligence*. https://doi.org/10.1177/26339137261421577 (preprint: "Against theory-motivated experimentation in science") | Random experimentation as a serious baseline |
| ✓ | Musslick, S., Hewson, J., Andrew, B., Strittmatter, Y., et al. (2023). An evaluation of experimental sampling strategies for autonomous empirical research in cognitive science. *Proceedings of the 45th Annual Conference of the Cognitive Science Society*. https://escholarship.org/uc/item/5ch569fg | Random vs. model-driven sampling, tested with AutoRA: random often wins |
| ✓ | Rainforth, T., Foster, A., Ivanova, D. R., & Bickford Smith, F. (2024). Modern Bayesian experimental design. *Statistical Science, 39*(1), 100–114. https://doi.org/10.1214/23-STS915 | Review: expected information gain, adaptive design |
| ● | Chaloner, K., & Verdinelli, I. (1995). Bayesian experimental design: A review. *Statistical Science, 10*(3), 273–304. | Classic review |
| ● | Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison. | Overview of active-learning query strategies |

## Project papers (one per group; each becomes a 2AFC sampler)

| | Reference | The sampler on 2AFC |
|---|---|---|
| ● | Watson, A. B., & Pelli, D. G. (1983). QUEST: A Bayesian adaptive psychometric method. *Perception & Psychophysics, 33*(2), 113–120. | Posterior over the threshold on a grid; next trial at the most probable threshold |
| ● | Kontsevich, L. L., & Tyler, C. W. (1999). Bayesian adaptive estimation of psychometric slope and threshold. *Vision Research, 39*(16), 2729–2737. | Psi method: posterior over (threshold, slope); next trial minimises the expected posterior entropy |
| ✓ | Watson, A. B. (2017). QUEST+: A general multidimensional Bayesian adaptive psychometric method. *Journal of Vision, 17*(3), 10. https://doi.org/10.1167/17.3.10 | Psi / QUEST generalised to two stimulus dimensions (ratio × scatteredness) and more parameters |
| ✓ | Cavagnaro, D. R., Myung, J. I., Pitt, M. A., & Kujala, J. V. (2010). Adaptive design optimization: A mutual information-based approach to model discrimination in cognitive science. *Neural Computation, 22*(4), 887–905. | Mutual information between the next answer and *which model* is true (e.g. with vs. without a scatteredness effect) |
| ✓ | Kim, W., Pitt, M. A., Lu, Z.-L., Steyvers, M., & Myung, J. I. (2014). A hierarchical adaptive approach to optimal experimental design. *Neural Computation, 26*(11), 2465–2492. https://doi.org/10.1162/NECO_a_00654 | Adaptive design that borrows strength from earlier players' data (prior from the class's random blocks) |
| ● | MacKay, D. J. C. (1992). Information-based objective functions for active data selection. *Neural Computation, 4*(4), 590–604. | Expected information gain about the model parameters (a logistic or GP model) |
| ● | Cohn, D. A., Ghahramani, Z., & Jordan, M. I. (1996). Active learning with statistical models. *Journal of Artificial Intelligence Research, 4*, 129–145. | Choose the trial that most reduces the model's expected prediction variance over all trials (not at the trial itself) |
| ✓ | Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745. | BALD: the noise vs. not-knowing split of Session 6, for GP classification |
| ● | Seung, H. S., Opper, M., & Sompolinsky, H. (1992). Query by committee. *Proceedings of the 5th Annual Workshop on Computational Learning Theory*, 287–294. | Several models (or bootstrapped fits); trials where they disagree most |
| ✓ | Gardner, J. R., Song, X., Weinberger, K. Q., Barbour, D., & Cunningham, J. P. (2015). Psychophysical detection testing with Bayesian active learning. *Proceedings of the 31st Conference on Uncertainty in Artificial Intelligence (UAI)*. https://auai.org/uai2015/proceedings/papers/213.pdf | GP classification of detection over a stimulus space, actively sampled (page numbers differ between listings) |
| ✓ | Song, X. D., Sukesan, K. A., & Barbour, D. L. (2018). Bayesian active probabilistic classification for psychometric field estimation. *Attention, Perception, & Psychophysics, 80*(3), 798–812. | A whole psychometric *field* over two dimensions with a GP, actively sampled: very close to our task |

Note on Song et al. (2018): the earlier, related paper with Garnett is Song, X. D., Garnett, R., & Barbour, D. L. (2017). Psychometric function estimation by probabilistic classification. *Journal of the Acoustical Society of America, 141*(4), 2513.
