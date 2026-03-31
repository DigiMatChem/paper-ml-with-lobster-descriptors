```{toctree}
:hidden:
essentials/index
```

```{toctree}
:caption: Target Properties
:hidden:
results/index
```


```{toctree}
:caption: Target data extraction
:hidden:
targets/Vibrational_and_thermodynamic_properties/index
targets/Lattice_thermal_conductivity/index
targets/Elasticity/index
```

```{toctree}
:caption: ML Scripts
:hidden:
feature_selector/index
correlation_analysis/index
training/index
explainer/index
t_test/index
```

```{toctree}
:caption: Misc Visualization
:hidden:
misc/index
```

```{toctree}
:caption: API reference
:hidden:
reference/index
```


```{toctree}
:caption: About
:hidden:
about/license
```

# Comprehensive SI

**Date**: {sub-ref}`today`

**Useful links**:
[Github Repository](https://github.com/DigiMatChem/paper-ml-with-lobster-descriptors) 

**Bonding Analysis Dataset: Composition and Structure Coverage**

::::{grid} 1
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: dataset/total
:link-type: doc
:class-header: custom-red-header

**LOBSTER Bonding analysis dataset**  
^^^
Click to see overview
:::

::::

**Target properties datasets: Composition and Structure Coverage**

::::{card-carousel} 3
:class: text-center

:::{card}
:link: dataset/vibrational
:link-type: doc
:class-header: custom-blue-header

**Vibrational & Thermodynamic properties**  
^^^
Click to see overview
:::

:::{card}
:link: dataset/elasticity
:link-type: doc
:class-header: custom-blue-header

**Elastic properties**  
^^^
Click to see overview
:::

:::{card}
:link: dataset/anharmonic
:link-type: doc
:class-header: custom-blue-header

**Anharmonic properties**  
^^^
Click to see overview
:::

::::


`mlproject` is a package that hosts all the utility scripts to reproduce the results from our publication: **A critical assessment of bonding descriptors for predicting materials properties**. 

To keep the publication concise, not all results are included. This repository provides a one-stop access point to all results, along with all code and data required to reproduce them. Results are organized as static HTML pages for easy navigation and are deployed on github pages.


(readme-page)=

```{include} ../README.md
---
start-line: 9
---
```
