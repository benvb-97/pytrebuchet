# Pytrebuchet

A Python-based trebuchet simulation project.

## Description

Pytrebuchet is a physics simulation tool for modeling and analyzing trebuchet mechanics.

The differential equations governing the trebuchet's motion were derived using Langrangian mechanics by E. Constans and A. Constans. They published their equations on [http://www.benchtophybrid.com/TB/TB_Literature.html](http://www.benchtophybrid.com/TB/TB_Literature.html). The code uses scipy's solve_ivp function to solve the system of ODEs.

The prediction of the projectile's trajectory by pytrebuchet has been verified by comparing it to the prediction made by [https://virtualtrebuchet.com/](https://virtualtrebuchet.com/) for the same configuration. 

## License

Pytrebuchet is provided under an MIT license that can be found in the LICENSE file.
