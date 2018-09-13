# Rat-maze-navigation
Maze navigation by model-free choice (habitual action selection) and imagination-based (deliberate / explicit pre-play) planning using a multi-component brain model 

Two main modes of model for reward-seeking navigation:

- IBP: Imagination-based planning by sampling and evaluating internally generated action sequences.
- HAS: Habitual action selection, using state - action mappings of dlS.

Random and perfect navigation strategies can be used for benchmarking.

Components of brain model, based on known anatomical and functional features of the rat reward-seeking navigation system, include:

- prefrontal cortex (PFC)
- hippocampus (HC)
- grid system (GS)
- ventral striatum (vS)
- dorso-lateral striatum (dlS)
- motor cortex (MC)
- visaul cortex (VC)

<br/>

![alt text](https://github.com/davidsamu/Rat-maze-navigation/blob/master/docs/schematics/full_model.png)

<br/>

Places cells in HC are modelled as integrating visual and proprioceptive sensory information (evidence) received from VC and GS (Bayesian inference by evidence integration). Accumulating path integration noise in GS is being continuously corrected by top-down HC feedback during navigation. Potential actions (movements) are sampled by PFC and evaluated by vS and dlS before execution.

Task settings to test navigation strategies and learning methods:
- Learned environment and rewards
- Learned environment but revalued rewards
- Modified environment (shortcuts added), learned rewards
- Modified environment (obstacles added), learned rewards
- Modified environment (shortcuts and obstacles added), learned rewards
- Modified environment (shortcuts added), revalued rewards
- Modified environment (obstacles added), revalued rewards
- Modified environment (shortcuts and obstacles added), revalued reward
- Unknown environment and rewards

Maze can be procedurally generated or pre-set (example below, rectangles: locations, links: paths, numbers in arrows: Q-values):

<br/>

![alt text](https://github.com/davidsamu/Rat-maze-navigation/blob/master/results/village/village.png)

<br/>

