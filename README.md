# Predicting Soil Carbon Content to Inform Preservation

## Motivation

Soil is one of the largest carbon reservoirs on Earth, containing approximately 75% of the carbon stored on land — three times more than the amount stored in living plants and animals. Soils perform carbon sequestration, the long term storage of carbon dioxide, which plays a significant role in maintaining a balanced and harmonious global carbon cycle.

However, soils are increasingly being destroyed through harmful practices such as deforestation, urban development, pollution, etc. This leads to the stored carbon being emitted into the atmosphere as CO2 and contributing to the positive feedback loop of climate change.

As a result, it is paramount for us to understand which areas of soil store the most amount of carbon and thus are the most important to preserve. In order to do this, we need to know the soil organic carbon content (SOC) within a unit of soil. However, SOC is difficult to measure: dry combustion, the most accurate form of measurement, requires expensive equipment and trained personnel. Therefore, we need to predict SOC based on metrics that are easier to measure or that we know of already. 

Our project is an ML model that predicts SOC from other soil parameters in order to inform soil preservation. We intend for our model to be flexible to predict SOC based on any number of given parameters. 

## Methods

We are training our model on the Harmonized World Soil Database, which is the world’s most comprehensive collection of soil data. It aggregates information across seven source databases and standardizes them across 20+ soil properties. 

The HWSD2 database:
- HWSD2_LAYERS is the primary table which has 48 parameters including Organic Carbon Content (ORG_CARBON).
  - 408835 rows x 48 columns
- HWSD2_LAYERS_METADATA contains more details about each parameter / column.
- D_* are the lookup tables corresponding to parameter *.

