"""Gandhi Cloth Company MIP Problem

Gandhi Cloth Company is capable of manufacturing three types of clothing:

1. shirts
2. shorts
3. pants

The manufacture of each type of clothing requires that Gandhi have the appropriate type of
machinery available. The machinery needed to manufacture each type of clothing must be rented at
the following weekly rates:

1. shirt machinery: $200 per week
2. shorts machinery: $150 per week
3. pants machinery: $100 per week

The manufacture of each type of clothing also requires cloth and labor as given in the table below.
Each week, the following resources are available:

1. 150 hours of labor
2. 160 sq yd of cloth

The variable unit cost and selling price for each type of clothing are shown below. The goal is to:
Formulate an Integer Program (IP) whose solution maximizes Gandhi’s weekly profits.

Product Data
==============
| Type   | Sales Price ($) | Cost ($) | Labor (hours) | Cloth (sq yd) |
| ------ | --------------- | -------- | ------------- | ------------- |
| Shirt  | 12              | 6        | 3             | 4             |
| Shorts | 8               | 4        | 2             | 3             |
| Pants  | 15              | 8        | 6             | 4             |

Fixed Machinery Rental Costs
=============================
| Product | Weekly Machine Rental Cost |
| ------- | -------------------------- |
| Shirt   | 200                        |
| Shorts  | 150                        |
| Pants   | 100                        |
"""
